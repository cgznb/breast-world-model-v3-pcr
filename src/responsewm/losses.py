"""Separate supervised FM paths from source-only stochastic outcome paths.

No endpoint interpolant or ground-truth future is ever passed to forecast().
Every unobserved clinical label has an explicit mask. Proper ensemble scores
preserve a sample axis per patient; they never mix patients' futures.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .legacy.encoder import corrupt_latent, PatientState
from .model import StateSequence
from .contracts import gather_visit
from .flow import make_joint_path


def zero(ref):
    return ref.sum()*0.0


def masked_mean(value, mask):
    mask = mask.to(value)
    return (value*mask).sum()/mask.sum().clamp_min(1)


def bernoulli_nll(logits, label, mask):
    if ((label[mask] != 0)&(label[mask] != 1)).any():
        raise ValueError("Observed pCR labels must be binary")
    safe = torch.where(mask,label,torch.zeros_like(label))
    return masked_mean(F.binary_cross_entropy_with_logits(logits.float(),safe.float(),reduction="none"),mask)


def marginal_bernoulli_nll(logits, label, mask, per_sample=False):
    """Stable -log(1/K sum p_k(y)); NOT mean_k BCE(p_k,y).

    Finite-K log Monte Carlo estimates are biased; evaluation must sweep K.
    Missing labels contribute exactly zero gradient.
    """
    if logits.ndim != 2 or logits.shape[0] != len(label) or logits.shape[1] < 1:
        raise ValueError("Logits must be [B,K>=1]")
    if ((label[mask] != 0)&(label[mask] != 1)).any():
        raise ValueError("Observed pCR labels must be binary")
    y = torch.where(mask,label,torch.zeros_like(label)).float()
    signed = logits.float()*(2*y[:,None]-1)
    log_prob = F.logsigmoid(signed)
    nll = -log_prob.mean(1) if per_sample else -(torch.logsumexp(log_prob,1)-math.log(logits.shape[1]))
    return masked_mean(nll,mask)


def energy_score(samples, target, mask):
    """Unbiased U-statistic energy score in a fixed semantic coordinate system.

    samples [B,K,F,S,d]; target [B,F,S,d]; mask [B,F]. Distances are over
    the observed subtrajectory only, normalized by its number of coordinates.
    Missing entire future: no gradient. K>=2. May be slightly negative at finite K.
    """
    if samples.ndim != 5 or samples.shape[1] < 2 or target.shape != (samples.shape[0],*samples.shape[2:]):
        raise ValueError("Energy score shape mismatch / K<2")
    b,k = samples.shape[:2]
    weight = mask[:,:,None,None].expand_as(target).float()
    x = (samples.float()*weight[:,None]).flatten(2)
    y = (target.float()*weight).flatten(1)
    scale = weight.flatten(1).sum(1).clamp_min(1).sqrt()
    observation = torch.linalg.vector_norm(x-y[:,None],dim=-1).mean(1)/scale
    dispersion = torch.cdist(x,x).sum((1,2))/(k*(k-1)*scale)
    return masked_mean(observation-.5*dispersion,mask.any(1))


def spatial_alignment(pred, target):
    """iREPA-style spatial centering/normalization, then channel cosine distance.

    Target must contain spatial tokens. A global Pillar vector cannot be tiled
    here. For one spatial token the spatial objective is not identifiable.
    """
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1] < 2:
        raise ValueError("Spatial alignment requires matched [B,N>=2,d] tokens")
    def normalize(x):
        x = x.float()
        x = (x-x.mean(1,keepdim=True))/(x.var(1,unbiased=False,keepdim=True)+1e-4).sqrt()
        return F.normalize(x,dim=-1)
    return (1-(normalize(pred)*normalize(target)).sum(-1)).mean()


def variance_covariance(tokens):
    # Spatial token regularity is NOT a claim of independent patient statistics.
    x = tokens.float().reshape(-1,tokens.shape[-1])
    if len(x) < 2:
        return zero(x),zero(x)
    x = x-x.mean(0)
    var = F.relu(1-(x.square().mean(0)+1e-4).sqrt()).mean()
    cov = x.T@x/(len(x)-1)
    off = cov-torch.diag_embed(cov.diag())
    return var,off.square().sum()/x.shape[-1]


def _auxiliary(model,state,sidecars):
    """Only measured / explicitly typed sidecars; no fabricated tumor labels."""
    cfg = model.cfg
    names = ("pillar","dense_teacher","segmentation","kinetics","biomarkers")
    losses = {k:zero(state.dense) for k in names}
    counts = {k:0 for k in names}
    for i,aux in enumerate(sidecars):
        if "pillar" in aux and cfg.loss.pillar:
            target = aux["pillar"].flatten()
            if target.shape != (cfg.network.pillar_dim,):
                raise ValueError("Pillar global feature dimension mismatch")
            pred = model.pillar_projection(state.disease[i].mean(0))
            losses["pillar"] += 1-F.cosine_similarity(pred.float(),target.float(),dim=0)
            counts["pillar"] += 1
        if "dense_teacher" in aux and cfg.loss.dense_teacher:
            target = aux["dense_teacher"]
            if target.shape != state.dense[i].shape:
                raise ValueError("Dense teacher must match the declared spatial token grid")
            losses["dense_teacher"] += spatial_alignment(state.dense[i:i+1],target[None].detach())
            counts["dense_teacher"] += 1
        for name,kind in (("kinetics","kinetics"),("segmentation","segmentation"),("biomarkers","biomarker")):
            if name not in aux or not getattr(cfg.loss,name):
                continue
            if name+"_mask" not in aux:
                raise ValueError(f"Missing explicit validity mask for {name}")
            target,valid = aux[name],aux[name+"_mask"].bool()
            one = PatientState(state.dense[i:i+1],state.anatomy[i:i+1],state.disease[i:i+1],state.grid)
            pred = (model.state_heads.biomarker(one.disease.mean(1))[0] if name == "biomarkers"
                    else model.state_heads.volume(kind,one)[0])
            if target.shape != pred.shape:
                raise ValueError(f"{name} target must be in declared state-head coordinates")
            valid = valid.expand_as(pred)
            if valid.any():
                if name == "segmentation":
                    if ((target[valid]<0)|(target[valid]>1)).any():
                        raise ValueError("Invalid segmentation target")
                    loss = F.binary_cross_entropy_with_logits(pred.float(),target.float(),reduction="none")
                    losses[name] += masked_mean(loss,valid)
                    p = pred.float().sigmoid()
                    losses[name] += 1-(2*(p*target*valid).sum()+1)/((p*valid).sum()+(target*valid).sum()+1)
                else:
                    losses[name] += masked_mean(F.smooth_l1_loss(pred.float(),target.float(),reduction="none"),valid)
                counts[name] += 1
    return {k:v/max(1,counts[k]) for k,v in losses.items()},counts


def real_sequence(model,inp,sup,teacher=False):
    z = torch.cat((inp.observed,sup.future),1)
    days = torch.cat((inp.observed_days,inp.future_days),1)
    mask = torch.cat((inp.observed_mask,sup.future_mask),1)
    return model.encode_sequence(z,days,mask,teacher=teacher)


def representation_loss(model,inp,sup):
    cfg = model.cfg
    # Sample one real visit per patient, including terminal visits as TARGET data.
    images = torch.cat((inp.observed,sup.future),1)
    availability = torch.cat((inp.observed_mask,sup.future_mask),1)
    choices = [int(torch.multinomial(row.float(),1)) for row in availability]
    selected = torch.stack([images[i,j] for i,j in enumerate(choices)])
    sidecars = [sup.auxiliary[i][j] for i,j in enumerate(choices)]
    state = model.encoder(selected)
    with torch.no_grad():
        target = model.target_encoder(selected)
    corrupted,mask = corrupt_latent(selected,cfg.encoder)
    context = model.encoder(corrupted)
    predicted = model.masked_predictor(context)
    v,c = variance_covariance(state.dense)
    losses = {
        "reconstruction":F.smooth_l1_loss(model.state_heads.volume("reconstruction",state).float(),
                                           F.adaptive_avg_pool3d(selected,state.grid).float()),
        "phase_difference":F.smooth_l1_loss(model.state_heads.volume("latent_delta",state).float(),
                                             F.adaptive_avg_pool3d(model.encoder.raw_differences(selected),state.grid).float()),
        "masked_jepa":masked_mean(F.smooth_l1_loss(predicted.float(),target.dense.float(),reduction="none").mean(-1),mask),
        "variance":v,"covariance":c,
    }
    aux,counts = _auxiliary(model,state,sidecars)
    losses.update(aux)
    initial = model.encode_sequence(inp.observed,inp.observed_days,inp.observed_mask)
    memory = model.memory(inp,initial)
    truth = real_sequence(model,inp,sup)
    logits,residual,observed = model.read_trajectory(inp,memory,truth)
    losses["real_pcr"] = bernoulli_nll(logits,sup.label,sup.label_mask)
    losses["observed_pcr"] = bernoulli_nll(observed,sup.label,sup.label_mask)
    losses["residual_l2"] = residual.float().square().mean()
    # Anatomy invariance only on explicitly audited comparable pairs.
    anatomy = zero(state.dense)
    comparable = 0
    t = inp.observed.shape[1]
    for j in range(sup.future.shape[1]):
        valid = sup.future_mask[:,j]&sup.anatomy_comparable[:,j]
        if j:
            valid = valid&sup.future_mask[:,j-1]
            prior = truth.anatomy[:,t+j-1]
        else:
            prior = gather_visit(initial.anatomy,initial.mask)
        if valid.any():
            anatomy += F.smooth_l1_loss(prior[valid].mean(1),truth.anatomy[valid,t+j].mean(1))
            comparable += 1
    losses["anatomy"] = anatomy/max(1,comparable)
    total = sum(getattr(cfg.loss,k)*v for k,v in losses.items())
    metrics = {k:float(v.detach()) for k,v in losses.items()}
    metrics.update({"support/"+k:v for k,v in counts.items()})
    metrics["support/pcr"] = int(sup.label_mask.sum())
    return total,metrics


def flow_loss(model,inp,sup,generator=None,allow_reverse=True):
    """Teacher-forced FM only. It may use true past states for density training.

    A missing intermediate visit skips that adjacent FM pair, not that patient's
    representation/rollout/pCR supervision. Export a direct interval case when
    a nonadjacent pair should be trained explicitly.
    """
    cfg = model.cfg
    with torch.no_grad():
        truth = real_sequence(model,inp,sup,teacher=True)
    losses = {k:zero(next(model.velocity.parameters())) for k in ("fm_image","fm_state","repa")}
    count = 0
    b,t = inp.observed.shape[:2]
    for i in range(b):
        eligible = [j for j in range(sup.future.shape[1]) if bool(sup.future_mask[i,j]) and
                    (j == 0 or bool(sup.future_mask[i,j-1]))]
        if not eligible:
            continue
        j = eligible[int(torch.randint(len(eligible),(),device=inp.observed.device,generator=generator))]
        p = t+j
        earlier_z = (gather_visit(inp.observed[i:i+1],inp.observed_mask[i:i+1]) if j == 0
                     else sup.future[i:i+1,j-1])
        earlier_s = (gather_visit(truth.disease[i:i+1,:t],inp.observed_mask[i:i+1]) if j == 0
                     else truth.disease[i:i+1,p-1])
        later_z,later_s = sup.future[i:i+1,j],truth.disease[i:i+1,p]
        inp_i = type(inp)(**{key:getattr(inp,key)[i:i+1] for key in inp.__dataclass_fields__})
        reverse = allow_reverse and bool(torch.rand((),device=earlier_z.device,generator=generator)<cfg.training.reverse_probability)
        source_day = (gather_visit(inp.observed_days[i:i+1],inp.observed_mask[i:i+1]) if j == 0
                      else inp.future_days[i:i+1,j-1])
        target_day = inp.future_days[i:i+1,j]
        if reverse:
            seq = StateSequence(truth.dense[i:i+1,p:p+1],truth.anatomy[i:i+1,p:p+1],truth.disease[i:i+1,p:p+1],
                                target_day[:,None],torch.ones(1,1,device=earlier_z.device,dtype=torch.bool),
                                torch.zeros(1,1,device=earlier_z.device,dtype=torch.bool))
            from_day,to_day,direction = target_day,source_day,-1
            target_dense = (gather_visit(truth.dense[i:i+1,:t],inp.observed_mask[i:i+1]) if j == 0
                            else truth.dense[i:i+1,p-1])
        else:
            seq = StateSequence(truth.dense[i:i+1,:p],truth.anatomy[i:i+1,:p],truth.disease[i:i+1,:p],
                                truth.days[i:i+1,:p],truth.mask[i:i+1,:p],truth.generated[i:i+1,:p])
            from_day,to_day,direction = source_day,target_day,1
            target_dense = truth.dense[i:i+1,p]
        h = model.memory(inp_i,seq)
        context = model.conditions(h,inp.actions[i:i+1,j],inp.action_mask[i:i+1,j],from_day,to_day,direction)
        z,s,vz,vs,tau = make_joint_path(earlier_z,later_z,earlier_s,later_s,generator=generator)
        pred_z,pred_s,pred_dense = model.velocity(z,s,tau,context)
        losses["fm_image"] += F.mse_loss(pred_z.float(),vz.float())
        losses["fm_state"] += F.mse_loss(pred_s.float(),vs.float())
        if cfg.loss.repa:
            losses["repa"] += spatial_alignment(pred_dense,target_dense.detach())
        count += 1
    losses = {k:v/max(1,count) for k,v in losses.items()}
    total = sum(getattr(cfg.loss,k)*v for k,v in losses.items())
    metrics = {k:float(v.detach()) for k,v in losses.items()}
    metrics["support/fm_pairs"] = count
    return total,metrics


def _image_readout(model,inp,result):
    """Independent re-encoding readout for consistency, not a second generator."""
    initial = result.observed_states
    probabilities = []
    for k in range(result.logits.shape[1]):
        s = result.image_state[:,k]
        # pCR head only uses disease; re-use harmless empty dense/anatomy slots.
        seq = StateSequence(initial.dense,initial.anatomy,
                            torch.cat((initial.disease,s),1),
                            torch.cat((initial.days,inp.future_days),1),
                            torch.cat((initial.mask,inp.future_mask),1),
                            torch.cat((initial.generated,inp.future_mask),1))
        logits,_,_ = model.read_trajectory(inp,result.memory,seq)
        probabilities.append(logits.sigmoid())
    return torch.stack(probabilities,1)


def rollout_outcome_loss(model,inp,sup,stage="joint",generator=None):
    cfg = model.cfg
    # Readout adaptation freezes generation and encoders, not classifier gradients.
    if stage == "readout":
        with torch.no_grad():
            result = model.forecast(inp,samples=cfg.sampling.train_samples,steps=cfg.sampling.train_steps,generator=generator)
        # Re-read detached states with a trainable head.
        logits,residuals = [],[]
        for k in range(result.logits.shape[1]):
            ds = result.image_state[:,k] if cfg.network.readout_source == "reencode" else result.state[:,k]
            seq = result.observed_states
            seq = StateSequence(seq.dense,seq.anatomy,torch.cat((seq.disease,ds),1),
                                torch.cat((seq.days,inp.future_days),1),torch.cat((seq.mask,inp.future_mask),1),
                                torch.cat((seq.generated,inp.future_mask),1))
            logit,residual,obs = model.read_trajectory(inp,result.memory,seq)
            logits.append(logit); residuals.append(residual)
        result.logits = torch.stack(logits,1)
        result.residuals = torch.stack(residuals,1)
        result.observed_logit = obs
    else:
        result = model.forecast(inp,samples=cfg.sampling.train_samples,steps=cfg.sampling.train_steps,generator=generator)
    losses = {
        "marginal_pcr":marginal_bernoulli_nll(result.logits,sup.label,sup.label_mask,cfg.loss.per_sample_bce),
        "observed_pcr":bernoulli_nll(result.observed_logit,sup.label,sup.label_mask),
        "residual_l2":result.residuals.float().square().mean(),
    }
    with torch.no_grad():
        truth = real_sequence(model,inp,sup,teacher=True)
    # Genuine future is used only here as a supervised teacher readout.
    logits,_,_ = model.read_trajectory(inp,result.memory,truth)
    losses["real_pcr"] = bernoulli_nll(logits,sup.label,sup.label_mask)
    if stage != "readout" and inp.future_days.shape[1]:
        err = (result.state.float()-result.image_state.float()).square().mean((-1,-2)).mean(1)
        losses["grounding"] = masked_mean(err,inp.future_mask)
        target = truth.disease[:,inp.observed.shape[1]:]
        losses["energy"] = energy_score(result.image_state,target,sup.future_mask)
        # This is a small readout consistency term, not a proof of clinical truth.
        if cfg.loss.prediction_grounding:
            image_prob = _image_readout(model,inp,result)
            losses["prediction_grounding"] = (result.logits.sigmoid()-image_prob).square().mean()
    total = sum(getattr(cfg.loss,k)*v for k,v in losses.items())
    metrics = {k:float(v.detach()) for k,v in losses.items()}
    metrics["support/pcr"] = int(sup.label_mask.sum())
    return total,metrics


def stage_loss(model,inp,sup,stage,generator=None):
    if stage == "representation":
        return representation_loss(model,inp,sup)
    if stage == "flow":
        return flow_loss(model,inp,sup,generator)
    outcome,metrics = rollout_outcome_loss(model,inp,sup,stage,generator)
    if stage == "joint":
        density,other = flow_loss(model,inp,sup,generator)
        outcome = outcome+density
        metrics.update(other)
    return outcome,metrics
