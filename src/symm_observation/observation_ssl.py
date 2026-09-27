"""Stage A: only same-visit local/global/domain modeling and observation reconstruction."""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from .observation_encoder import ThreePhaseObservationEncoder, positions_3d
from .observation_heads import ObservationPredictor, SpatialReadout, ArcFaceHead, DomainHead
from .observation_views import map_query_positions
from .utils import update_ema


def selected_mse(prediction, target, mask):
    if not mask.any():
        raise ValueError("Empty JEPA target mask")
    errors = (prediction.float()-target.float()).square().mean(-1)
    return (errors*mask).sum()/mask.sum()


def masked_l1(prediction, target, valid):
    w = valid.to(prediction.dtype).expand_as(prediction)
    return ((prediction.float()-target.float()).abs()*w).sum()/w.sum().clamp_min(1.)


class ObservationPretrainer(nn.Module):
    def __init__(self, cfg, statistics):
        super().__init__()
        self.cfg = cfg
        self.encoder = ThreePhaseObservationEncoder(cfg.encoder, statistics)
        self.target_encoder = copy.deepcopy(self.encoder).eval().requires_grad_(False)
        self.predictor = ObservationPredictor(cfg.encoder)
        # Deep readout shared between current-observation reconstruction and measured readouts.
        self.readout = SpatialReadout(cfg.encoder, 27)  # 24 VQ + 2 signal differences + 1 segmentation logit
        a = cfg.observation
        self.arcface = ArcFaceHead(cfg.encoder.dim, a.phenotype_classes) if a.arcface_weight else None
        self.domain_head = DomainHead(cfg.encoder.dim, a.domain_classes) if a.dann_weight else None

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, step, exponent=1.0):
        a = self.cfg.observation
        fraction = min(1., step/max(1, self.cfg.training.stage_a_steps))
        decay = a.ema_start + (a.ema_end-a.ema_start)*.5*(1-math.cos(math.pi*fraction))
        decay = decay ** exponent
        update_ema(self.target_encoder, self.encoder, decay)
        return decay

    def forward(self, batch, step=0):
        if set(batch) != {"a", "b", "phenotype", "domain", "visit_ids"}:
            raise ValueError("Stage A accepts only same-visit observation views and current labels")
        cfg = self.cfg.observation
        av, bv = batch["a"], batch["b"]
        kwargs = {"minimum_coverage": cfg.minimum_patch_coverage}
        za = self.encoder(av["context"], av["valid"], av["visible"], **kwargs)
        zb = self.encoder(bv["context"], bv["valid"], bv["visible"], **kwargs)
        with torch.no_grad():
            ha = self.target_encoder(av["clean"], av["valid"], teacher_average=True, **kwargs).dense()
            hb = self.target_encoder(bv["clean"], bv["valid"], teacher_average=True, **kwargs).dense()
        coords = positions_3d(za.grid, av["clean"].device)[None].expand(len(av["clean"]), -1, -1)
        p_aa = self.predictor(za, coords, av["action"], ~av["good"])
        p_bb = self.predictor(zb, coords, bv["action"], ~bv["good"])
        local = .5*(selected_mse(p_aa, ha, av["hidden"]) + selected_mse(p_bb, hb, bv["hidden"]))
        if cfg.global_weight:
            to_b = map_query_positions(av["affine"], bv["affine"], coords)
            to_a = map_query_positions(bv["affine"], av["affine"], coords)
            p_ab = self.predictor(za, to_b, av["action"], ~bv["good"])
            p_ba = self.predictor(zb, to_a, bv["action"], ~av["good"])
            global_loss = .5*(selected_mse(p_ab, hb, bv["good"]) + selected_mse(p_ba, ha, av["good"]))
        else:
            global_loss = local*0
        # A separate CLEAN current-observation path anchors detailed representations.
        # No raw-input skip or future observation is available to the readout.
        clean = self.encoder(av["clean"], av["valid"], **kwargs)
        read = self.readout(clean)
        valid = av["valid"]
        reconstruction = masked_l1(read[:, :24], self.encoder.normalize(av["clean"]), valid)
        total = cfg.local_weight*local + cfg.global_weight*global_loss + cfg.reconstruction_weight*reconstruction
        parts = {"local": local, "global": global_loss, "reconstruction": reconstruction,
                 "arcface": 0., "dann": 0., "phenotype_labels": 0, "domain_labels": 0}
        for name, channels, weight in (("kinetics", slice(24, 26), cfg.kinetics_weight),
                                       ("segmentation", slice(26, 27), cfg.segmentation_weight)):
            present = av[name+"_present"]
            value = total*0
            if weight and present.any():
                pred, truth, mask = read[present, channels], av[name][present], valid[present]
                if name == "kinetics":
                    value = masked_l1(pred, truth, mask)
                else:
                    bce = F.binary_cross_entropy_with_logits(pred.float(), truth.float(), reduction="none")
                    bce = (bce*mask).sum()/mask.sum().clamp_min(1.)
                    prob = pred.float().sigmoid()*mask
                    truth = truth*mask
                    dice = 1-(2*(prob*truth).flatten(1).sum(1)+1)/(prob.flatten(1).sum(1)+truth.flatten(1).sum(1)+1)
                    value = bce+dice.mean()
                total = total + weight*value
            parts[name], parts[name+"_labels"] = value, int(present.sum())
        if step >= cfg.supervised_start_step:
            labels = batch["phenotype"]
            keep = labels >= 0
            if self.arcface is not None and keep.any():
                if (labels[keep] >= cfg.phenotype_classes).any():
                    raise ValueError("Phenotype label outside declared class range")
                arc = F.cross_entropy(self.arcface(clean.pooled()[keep], labels[keep]), labels[keep])
                total = total + cfg.arcface_weight*arc
                parts["arcface"] = arc
                parts["phenotype_labels"] = int(keep.sum())
            labels = batch["domain"]
            keep = labels >= 0
            if self.domain_head is not None and keep.any():
                if (labels[keep] >= cfg.domain_classes).any():
                    raise ValueError("Domain label outside declared scanner/site range")
                logits = self.domain_head(clean.tokens[keep])
                targets = labels[keep, None].expand(-1, logits.shape[1])
                losses = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1), reduction="none").reshape_as(targets)
                good = ~clean.padding[keep]
                dann = (losses*good).sum()/good.sum()
                total = total + cfg.dann_weight*dann
                parts["dann"] = dann
                parts["domain_labels"] = int(keep.sum())
        values = clean.tokens[~clean.padding].float()
        std = values.std(0, unbiased=False).mean()
        if cfg.variance_weight:
            var_loss = F.relu(1.-(values.var(0, unbiased=False)+1e-4).sqrt()).mean()
            total = total + cfg.variance_weight*var_loss
            parts["variance"] = var_loss
        parts["token_std"] = std
        parts["loss"] = total
        return total, {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in parts.items()}
