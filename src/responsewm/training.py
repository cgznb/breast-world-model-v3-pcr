"""Four stages, audited data identity, validation selection and exact step-boundary resume."""
from __future__ import annotations
from pathlib import Path
import math
import time
import torch
from .config import from_dict
from .data import ManifestStore,PatientSampler
from .model import ResponseWorldModel
from .losses import stage_loss,representation_loss,flow_loss,marginal_bernoulli_nll,real_sequence
from .metrics import classification_metrics,patient_bootstrap
from .io import read_json,write_json,save_checkpoint,load_checkpoint,rng_state,restore_rng,seed_all,autocast,stable_hash
from .checkpoints import migrate_v2

STAGES = ("representation","flow","readout","joint")


def initialize_run(store,cfg,root):
    if store.manifest["time_basis"] != cfg.network.time_basis:
        raise ValueError("Dataset time basis differs from model configuration")
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    path = root/"metadata.json"
    if path.exists():
        meta = read_json(path)
        if meta["config_digest"] != cfg.digest or meta["manifest_digest"] != store.manifest_digest:
            raise ValueError("Run configuration or manifest changed. Use a new output directory")
        if meta["asset_signatures"] != store.asset_signature():
            raise ValueError("An input latent/sidecar file changed since this run was created")
        store.set_statistics(meta["statistics"])
        return meta
    if not store.by_split["val"]:
        raise ValueError("A patient-disjoint validation split is required for checkpoint selection")
    stats = store.fit_statistics()
    meta = {"schema":"responsewm_run_v1","config":cfg.to_dict(),"config_digest":cfg.digest,
            "manifest_digest":store.manifest_digest,"statistics":stats,"asset_signatures":store.asset_signature(),
            "data_contract":{k:store.manifest[k] for k in ("clinical_features","action_features","phase_order","latent_shape","vq_identity","time_basis")},
            "audit":store.audit(),"synthetic":store.manifest.get("synthetic",False),"clinical_validation":False}
    write_json(path,meta)
    return meta


def build_model(cfg,store):
    model = ResponseWorldModel(cfg,store.c,store.a)
    st = store.statistics
    model.encoder.set_normalization(st["latent_mean"],st["latent_std"])
    model.target_encoder.set_normalization(st["latent_mean"],st["latent_std"])
    return model


def _previous(root,stage):
    previous = STAGES[STAGES.index(stage)-1]
    folder = Path(root)/previous
    last = folder/"last.pt"
    if not last.exists() or not load_checkpoint(last).get("completed",False):
        raise ValueError(f"Complete {previous} before starting {stage}")
    return folder/"best.pt" if (folder/"best.pt").exists() else last


def learning_rate(step,total,warmup):
    warmup = min(warmup,max(1,total//5))
    if step < warmup:
        return (step+1)/max(1,warmup)
    return .01+.99*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))


def _validation_indices(store,limit):
    # Round-robin across patients before taking additional landmarks.
    groups = {}
    for i in store.by_split["val"]:
        groups.setdefault(store.cases[i]["patient_id"],[]).append(i)
    result = []; depth = 0
    while len(result)<limit:
        added = [group[depth] for _,group in sorted(groups.items()) if len(group)>depth]
        if not added:
            break
        result.extend(added); depth += 1
    return result[:limit]


@torch.no_grad()
def validate(model,store,cfg,stage):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = _validation_indices(store,cfg.training.validation_cases)
    values = []; y=[]; p=[]; distances=[]
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.training.seed+100003)
        if devices:
            torch.cuda.manual_seed_all(cfg.training.seed+100003)
        generator = torch.Generator(device=device).manual_seed(cfg.training.seed+100003)
        for index in ids:
            inp,sup = store.batch([index],device)
            with autocast(device,cfg.training.precision):
                if stage == "representation":
                    _,terms = representation_loss(model,inp,sup)
                    values.append(terms["reconstruction"]+terms["masked_jepa"]+terms["real_pcr"])
                elif stage == "flow":
                    value,_ = flow_loss(model,inp,sup,generator,allow_reverse=False)
                    values.append(float(value))
                else:
                    result = model.forecast(inp,samples=cfg.training.validate_samples,steps=cfg.sampling.inference_steps,generator=generator)
                    if bool(sup.label_mask[0]):
                        y.append(int(sup.label[0])); p.append(float(result.probability[0]))
                    if stage == "joint":
                        density,_ = flow_loss(model,inp,sup,generator,allow_reverse=False)
                        values.append(float(density))
                        if inp.future_days.numel():
                            distance = (result.state.float()-result.image_state.float()).square().mean((-1,-2)).mean(1)
                            distances.append(float((distance*inp.future_mask).sum()/inp.future_mask.sum().clamp_min(1)))
    model.train(was_training)
    if stage in {"representation","flow"}:
        return {"selection_score":sum(values)/max(1,len(values)),"selection_metric":stage+"_validation_objective",
                "split":"val","cases":len(ids),"synthetic":store.manifest.get("synthetic",False)}
    if not y:
        raise ValueError("Readout/joint validation needs at least one labelled validation patient")
    result = classification_metrics(y,p)
    generation = (sum(values)/max(1,len(values)))+(sum(distances)/max(1,len(distances)))
    result.update(selection_score=result["nll"]+cfg.training.selection_generation_weight*generation,
                  selection_metric="marginal_NLL_plus_weighted_generation",generation_objective=generation,
                  split="val",cases=len(ids),synthetic=store.manifest.get("synthetic",False))
    return result


def train_stage(store,cfg,root,stage,*,resume=False,init_v2=None,stop_after=None):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    seed_all(cfg.training.seed,cfg.training.threads,cfg.training.strict_determinism)
    if cfg.training.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; select an explicit CPU smoke configuration")
    meta = initialize_run(store,cfg,root)
    folder = Path(root)/stage; folder.mkdir(parents=True,exist_ok=True)
    last_path = folder/"last.pt"
    if last_path.exists() and not resume:
        raise FileExistsError("Stage checkpoint exists; use --resume rather than overwrite")
    total_steps = getattr(cfg.training,stage+"_steps")
    if total_steps < 1:
        raise ValueError("A requested stage must have a positive optimization budget")
    if stage in {"readout","joint"} and not any(store.cases[i]["target"]["pcr"] is not None for i in store.by_split["train"]):
        raise ValueError("No training pCR labels for outcome training")
    if stage in {"flow","joint"}:
        paired = False
        for i in store.by_split["train"]:
            future = store.cases[i]["target"]["future"]
            paired |= any(v is not None and (j == 0 or future[j-1] is not None) for j,v in enumerate(future))
        if not paired:
            raise ValueError("No eligible adjacent FM pair; export direct-interval cases for nonadjacent observations")
    if init_v2 and (stage != "representation" or resume):
        raise ValueError("V2 warm start is allowed only at a fresh representation stage")
    model = build_model(cfg,store)
    migration = None
    if stage != "representation" and not resume:
        payload = load_checkpoint(_previous(root,stage))
        model.load_state_dict(payload["model"],strict=True)
        if stage == "flow":
            model.freeze_representation()
    elif not resume:
        if init_v2:
            migration = migrate_v2(model,init_v2)
            write_json(Path(root)/"v2_migration.json",migration)
        prior = store.fit_prior(model)
        write_json(Path(root)/"clinical_prior.json",prior)
    model.to(cfg.training.device)
    params = model.configure_stage(stage)
    lr = cfg.training.joint_lr if stage == "joint" else cfg.training.lr
    optimizer = torch.optim.AdamW(params,lr=lr,weight_decay=cfg.training.weight_decay)
    sampler = PatientSampler(store,cfg.training.seed+17,eligible_only=(stage == "flow"))
    start,best = 0,float("inf")
    if resume:
        if not last_path.exists():
            raise FileNotFoundError("No checkpoint to resume")
        payload = load_checkpoint(last_path)
        if payload["config_digest"] != cfg.digest or payload["manifest_digest"] != store.manifest_digest or payload["stage"] != stage:
            raise ValueError("Resume contract mismatch")
        model.load_state_dict(payload["model"],strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start,best = payload["step"],payload["best_score"]
        sampler.generator.set_state(payload["sampler_rng"])
        restore_rng(payload["rng"])
        if payload.get("completed",False):
            return {"stage":stage,"completed":True,"steps":start,"checkpoint":str(last_path)}
    def snapshot(step,completed=False):
        return {"schema":"responsewm_checkpoint_v1","stage":stage,"step":step,"completed":completed,
                "config":cfg.to_dict(),"config_digest":cfg.digest,"manifest_digest":store.manifest_digest,
                "metadata":meta,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "best_score":best,"sampler_rng":sampler.generator.get_state(),"rng":rng_state()}
    start_time = time.monotonic()
    for step in range(start,total_steps):
        optimizer.zero_grad(set_to_none=True)
        factor = learning_rate(step,total_steps,cfg.training.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr*factor
        aggregate={}; total_loss=0.
        for _ in range(cfg.training.accumulation):
            indices = sampler.sample(cfg.training.batch_size)
            inp,sup = store.batch(indices,cfg.training.device)
            with autocast(cfg.training.device,cfg.training.precision):
                loss,terms = stage_loss(model,inp,sup,stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss at step {step+1}")
            (loss/cfg.training.accumulation).backward()
            total_loss += float(loss.detach())/cfg.training.accumulation
            for k,v in terms.items():
                aggregate[k] = aggregate.get(k,0.)+v/cfg.training.accumulation
        norm = torch.nn.utils.clip_grad_norm_(params,cfg.training.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        model.update_target()
        getattr(model,stage+"_ready").fill_(True)
        done = step+1
        if done%cfg.training.log_every == 0 or done == total_steps:
            record = {"stage":stage,"step":done,"loss":total_loss,"gradient_norm":float(norm),
                      "lr":lr*factor,"elapsed_seconds":time.monotonic()-start_time,**aggregate}
            with (folder/"training.jsonl").open("a",encoding="utf-8") as handle:
                import json
                handle.write(json.dumps(record,allow_nan=False)+"\n")
            print(f"{stage} step={done} loss={total_loss:.5f} grad_norm={float(norm):.5f}",flush=True)
        if done%cfg.training.validation_every == 0 or done == total_steps:
            result = validate(model,store,cfg,stage)
            write_json(folder/"validation_last.json",result)
            if result["selection_score"] < best:
                best = result["selection_score"]
                save_checkpoint(folder/"best.pt",snapshot(done,done == total_steps))
                write_json(folder/"validation_best.json",result)
        if done%cfg.training.checkpoint_every == 0 or done == total_steps or (stop_after is not None and done >= stop_after):
            save_checkpoint(last_path,snapshot(done,done == total_steps))
        if stop_after is not None and done >= stop_after:
            break
    return {"stage":stage,"completed":done == total_steps,"steps":done,"checkpoint":str(last_path)}


def load_trained(path,device="cpu"):
    payload = load_checkpoint(path)
    if payload.get("schema") != "responsewm_checkpoint_v1":
        raise ValueError("Expected responsewm checkpoint, not a raw V2 checkpoint")
    cfg = from_dict(payload["config"])
    contract = payload["metadata"]["data_contract"]
    model = ResponseWorldModel(cfg,len(contract["clinical_features"]),len(contract["action_features"]))
    model.load_state_dict(payload["model"],strict=True)
    model.stage = payload["stage"]
    model.to(device).eval().requires_grad_(False)
    return model,payload


@torch.no_grad()
def evaluate(path,manifest,output,*,split="test",device="cpu",samples=8,steps=None,bootstrap=0,allow_synthetic=False):
    import numpy as np
    model,payload = load_trained(path,device)
    if not bool(model.readout_ready or model.joint_ready):
        raise ValueError("Generated-trajectory readout has not received training updates")
    store = ManifestStore(manifest,allow_synthetic)
    if store.manifest_digest != payload["manifest_digest"]:
        raise ValueError("Evaluation manifest differs from training contract; create an audited external-cohort adapter")
    store.set_statistics(payload["metadata"]["statistics"])
    if not store.by_split[split]:
        raise ValueError("Requested split has no cases")
    if samples < 1:
        raise ValueError("samples must be positive")
    seed_all(model.cfg.training.seed,model.cfg.training.threads)
    rng = torch.Generator(device=device).manual_seed(model.cfg.training.seed+200003)
    rows=[]; y=[]; probs=[]; pids=[]; per_draw=[]; per_origin={}
    for i in store.by_split[split]:
        # Crucially: prediction loads observed arrays only. Labels read AFTER prediction.
        inp = store.batch([i],device,supervised=False)
        with autocast(device,model.cfg.training.precision):
            result = model.forecast(inp,samples=samples,steps=steps,generator=rng)
        case = store.cases[i]
        p = float(result.probability[0]); label = case["target"]["pcr"]
        rows.append({"case_hash":stable_hash(case["id"]),"probability":p,"label":label,
                     "landmark_day":case["input"]["landmark_day"],"observed_visits":len(case["input"]["observed"]),
                     "trajectory_probability_std":float(result.logits[0].sigmoid().std(unbiased=False))})
        per_draw.append(result.logits[0].float().sigmoid().cpu().numpy())
        if label is not None:
            y.append(label); probs.append(p); pids.append(case["patient_id"])
            key = str(len(case["input"]["observed"]))+"_observed_visits"
            per_origin.setdefault(key,[]).append((label,p,case["patient_id"]))
    report = {"split":split,"samples":samples,"steps":steps or model.cfg.sampling.inference_steps,
              "synthetic":store.manifest.get("synthetic",False),"clinical_validation":False,
              "independence":"Patient-disjoint manifest checked; prior pretraining/development exposure still requires auditing",
              "aggregate_metrics":classification_metrics(y,probs) if y else None,
              "by_observed_prefix":{k:classification_metrics([a[0] for a in v],[a[1] for a in v]) for k,v in per_origin.items()},
              "rows":rows}
    if bootstrap and y:
        report["patient_bootstrap_95ci"] = patient_bootstrap(y,probs,pids,bootstrap,model.cfg.training.seed)
    write_json(output,report)
    np.savez_compressed(Path(output).with_suffix(".npz"),probabilities=np.asarray([r["probability"] for r in rows]),
                        trajectory_probabilities=np.stack(per_draw),labels=np.asarray([-1 if r["label"] is None else r["label"] for r in rows]))
    return report
