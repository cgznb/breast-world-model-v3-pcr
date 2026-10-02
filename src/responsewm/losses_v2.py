"""Patient-normalized supervision with explicit paired, free and observed paths."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from .conditioning import IntervalSpec
from .flow import make_joint_path
from .losses import (bernoulli_nll, marginal_bernoulli_nll, energy_score,
                     spatial_alignment, variance_covariance, _auxiliary, zero)
from .representation import deep_jepa_loss
from .rollout import NoiseLedger
from .state import Observation


def _require_homogeneous_batch(inp, sup=None):
    if len(inp.observed) < 2:
        return
    if not bool((inp.as_of == inp.as_of[0]).all()):
        raise ValueError("Batched objectives require the same as_of stage")
    if any(stages != inp.observed_stage_ids[0] for stages in inp.observed_stage_ids[1:]):
        raise ValueError("Batched objectives require the same observed stage history")
    if sup is not None:
        if not bool((sup.future_mask == sup.future_mask[0]).all()):
            raise ValueError("Batched objectives require the same future availability mask")
        if not bool((sup.label_mask == sup.label_mask[0]).all()):
            raise ValueError("Batched objectives require the same label availability mask")


def belief_logits(model, belief):
    return model.observed_logit(belief)


def prefix_objective(model, inp, sup):
    """A uniformly selected landmark estimates the within-patient mean."""
    _require_homogeneous_batch(inp, sup)
    belief = model.initialize(inp)
    logits = belief_logits(model, belief)
    pcr = marginal_bernoulli_nll(logits, sup.label, sup.label_mask)
    a, s = model.cfg.encoder.anatomy_tokens, model.cfg.encoder.disease_tokens
    prediction = model.assimilator.reconstruction(belief.memory[:, :, a:a+s])
    reconstruction = F.mse_loss(prediction.float(), belief.anchor_disease.detach().float())
    return pcr, reconstruction, belief


def representation_objective(model, inp, sup):
    _require_homogeneous_batch(inp, sup)
    cfg = model.cfg
    images = torch.cat((inp.observed, sup.future), 1)
    valid = torch.cat((inp.observed_mask, sup.future_mask), 1)
    indices = [int(torch.multinomial(row.float(), 1)) for row in valid]
    selected = torch.stack([images[i,j] for i,j in enumerate(indices)])
    state = model.encoder(selected)
    jepa, details = deep_jepa_loss(model.encoder, model.target_encoder, model.deep_predictor,
        selected, mix=cfg.multistage.deep_jepa_mix,
        level_weights=cfg.multistage.deep_jepa_weights,
        visible_weight=cfg.multistage.deep_jepa_visible_weight)
    # Regularize spatial tokens within each patient, matching single-patient training.
    regularity = [variance_covariance(tokens[None]) for tokens in state.dense]
    var, cov = (torch.stack(values).mean() for values in zip(*regularity))
    terms = {
        "reconstruction": F.smooth_l1_loss(model.state_heads.volume("reconstruction", state).float(),
                                            F.adaptive_avg_pool3d(selected, state.grid).float()),
        "phase_difference": F.smooth_l1_loss(model.state_heads.volume("latent_delta", state).float(),
                                              F.adaptive_avg_pool3d(model.encoder.raw_differences(selected), state.grid).float()),
        "masked_jepa": jepa, "variance": var, "covariance": cov,
    }
    aux, counts = _auxiliary(model, state, [sup.auxiliary[i][j] for i,j in enumerate(indices)])
    terms.update({name: value * counts[name] / len(selected) for name, value in aux.items()})
    pcr, assimilation, _ = prefix_objective(model, inp, sup)
    total = sum(getattr(cfg.loss, k)*v for k,v in terms.items())
    total = total + cfg.multistage.representation_pcr_weight*pcr + cfg.multistage.assimilation_reconstruction*assimilation
    metrics = {"loss/"+k: float(v.detach()) for k,v in {**terms, **details,
               "real_prefix_pcr": pcr, "assimilation": assimilation}.items()}
    metrics.update({"support/"+k: v for k,v in counts.items()})
    metrics.update({"support/label": int(sup.label_mask.sum()), "support/patients": len(selected),
                    "real_prefix_stage": int(inp.as_of[0])})
    return total, metrics


def edge_objective(model, inp, truth, generator=None):
    """Only noisy FM endpoints see the future truth; conditions see the source."""
    _require_homogeneous_batch(inp)
    belief = model.initialize(inp)
    if model.action_dim and truth.kind == "bridge_auxiliary":
        raise ValueError("Nonadjacent action aggregation needs an audited interval policy")
    interval = IntervalSpec(truth.source_stage, truth.target_stage,
                            inp.actions[:,0] if inp.actions.shape[1] else None,
                            inp.action_mask[:,0] if inp.action_mask.shape[1] else None,
                            mode="direct_aux" if truth.kind == "bridge_auxiliary" else "sequential")
    return edge_from_belief(model,belief,truth.latent,interval,generator)


def edge_from_belief(model,belief,target_latent,interval,generator=None):
    cfg=model.cfg
    if belief.samples != 1:
        raise ValueError("Paired FM uses one explicit observed posterior")
    with torch.no_grad():
        target=model.target_encoder(target_latent)
    context = model.conditions(belief.memory[:,0], belief.clinical, belief.clinical_mask, interval)
    z,s,vz,vs,tau = make_joint_path(belief.anchor_latent[:,0], target_latent,
                                  belief.anchor_disease[:,0], target.disease, generator=generator)
    pz,ps,pdense = model.velocity(z,s,tau,context)
    terms = {"fm_image": F.mse_loss(pz.float(), vz.float()),
             "fm_state": F.mse_loss(ps.float(), vs.float()),
             "repa": spatial_alignment(pdense, target.dense) if cfg.loss.repa else zero(pdense)}
    total = sum(getattr(cfg.loss,k)*v for k,v in terms.items())
    edge = str(interval.src_stage)+str(interval.dst_stage)
    metrics = {"loss/"+k+"/"+edge: float(v.detach()) for k,v in terms.items()}
    metrics["support/edge"+edge] = len(target_latent)
    return total, metrics


def interval_plan(inp):
    origin=int(inp.as_of[0])
    return tuple(IntervalSpec(origin+j,origin+j+1,inp.actions[:,j],inp.action_mask[:,j],known_at=origin)
                 for j in range(inp.actions.shape[1]))


def distribution_objectives(model, trace, inp, sup):
    """Ground every simulated stage, but score truth only where it exists."""
    _require_homogeneous_batch(inp, sup)
    stages = tuple(trace.output_stages)
    values, masks = [], []
    with torch.no_grad():
        for stage in stages:
            col = stage-int(inp.as_of[0])-1
            valid = sup.future_mask[:,col]
            z = sup.future[:,col]
            disease = z.new_zeros(len(z), model.cfg.encoder.disease_tokens, model.cfg.encoder.dim)
            if valid.any():
                encoded = model.target_encoder(z[valid]).disease
                disease = disease.to(encoded).index_copy(0, valid.nonzero(as_tuple=True)[0], encoded)
            values.append(disease)
            masks.append(valid)
    if not values:
        loss = zero(trace.logits)
        return loss, {"support/distribution": 0}
    target, mask = torch.stack(values,1), torch.stack(masks,1)
    grounding = (trace.state.float()-trace.image_state.float()).square().mean((0,1,3,4))
    joint = energy_score(trace.image_state, target, mask)
    scores = [energy_score(trace.image_state[:,:,j:j+1], target[:,j:j+1], mask[:,j:j+1])
              for j in range(len(stages)) if mask[:,j].any()]
    per_stage = torch.stack(scores).mean() if scores else zero(trace.image_state)
    m, cfg = model.cfg.multistage, model.cfg
    total = cfg.loss.grounding*grounding.mean()+cfg.loss.energy*(m.distribution_stage_mix*per_stage+m.distribution_joint_mix*joint)
    metrics = {"loss/energy_joint": float(joint.detach()), "loss/energy_stage": float(per_stage.detach()),
               "support/distribution": int(mask.sum()), "support/distribution_patients": int(mask.any(1).sum())}
    for j,stage in enumerate(stages):
        metrics[f"loss/grounding/T{stage}"] = float(grounding[j].detach())
        metrics[f"support/target_T{stage}"] = int(mask[:,j].sum())
    return total, metrics


def free_rollout_objective(model, inp, sup, *, seed, max_hops=3, marginal_weight=0.0):
    _require_homogeneous_batch(inp, sup)
    cfg = model.cfg
    belief = model.initialize(inp)
    stages = tuple(range(belief.stage+1, min(3, belief.stage+max_hops)+1))
    if not stages:
        return zero(belief_logits(model,belief)), {"rollout_depth": 0}
    trace = model.forecast(belief, output_stages=stages, samples=cfg.sampling.train_samples,
                           steps=cfg.sampling.train_steps, noise_ledger=NoiseLedger(seed),
                           through_stage=stages[-1],plan=interval_plan(inp))
    loss, metrics = distribution_objectives(model, trace, inp, sup)
    if marginal_weight:
        if stages[-1] != 3:
            raise ValueError("Main marginal pCR requires complete remaining future")
        marginal = marginal_bernoulli_nll(trace.logits, sup.label, sup.label_mask)
        loss = loss+marginal_weight*marginal
        metrics["loss/marginal_pcr"] = float(marginal.detach())
        metrics["support/marginal_label"] = int(sup.label_mask.sum())
    metrics.update(rollout_depth=len(stages), K=cfg.sampling.train_samples,
                   heun_steps=cfg.sampling.train_steps, origin_stage=belief.stage)
    return loss, metrics


def observed_update_objective(model, inp, sup, *, seed):
    """A separate model-prior-to-real-MRI training task, never a free-path input."""
    _require_homogeneous_batch(inp, sup)
    belief = model.initialize(inp)
    candidates = [j for j in range(sup.future.shape[1]) if bool(sup.future_mask[:,j].all())]
    if not candidates:
        return zero(belief.memory), {"support/assimilation_updates": 0}
    j = candidates[int(torch.randint(len(candidates), ()))]
    target_stage = int(inp.as_of[0])+j+1
    prior = belief
    plan=interval_plan(inp)
    for stage in range(belief.stage+1,target_stage+1):
        prior = model.advance(prior, plan[stage-belief.stage-1], noise_ledger=NoiseLedger(seed),
                              steps=model.cfg.sampling.train_steps).final_state
    observation = Observation(target_stage, sup.future[:,j],
                               tuple(f"training-observation-T{target_stage}-{b}" for b in range(len(inp.observed))))
    posterior = model.observe(prior, observation)
    pcr = marginal_bernoulli_nll(belief_logits(model,posterior), sup.label, sup.label_mask)
    a,s = model.cfg.encoder.anatomy_tokens,model.cfg.encoder.disease_tokens
    reconstructed = model.assimilator.reconstruction(posterior.memory[:,:,a:a+s])
    assimilate = F.mse_loss(reconstructed.float(),posterior.anchor_disease.detach().float())
    loss=model.cfg.loss.real_pcr*pcr+model.cfg.multistage.assimilation_reconstruction*assimilate
    metrics={
        "loss/posterior_pcr": float(pcr.detach()), "loss/assimilation_update": float(assimilate.detach()),
        "support/assimilation_updates": len(inp.observed), "posterior_stage": target_stage}
    if target_stage<3 and bool(sup.future_mask[:,j+1].all()) and model.stage!="readout":
        continuation,details=edge_from_belief(model,posterior,sup.future[:,j+1],plan[j+1])
        loss=loss+continuation
        metrics.update({"posterior_continuation/"+key:value for key,value in details.items()})
    return loss,metrics
