"""Measure complete real-data optimizer updates in an isolated GPU process."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
import torch
from symm_observation.config import load_config
from symm_observation.data import VisitStore, PairStore, PatientSampler
from symm_observation.conditioning import ConditionSchema
from symm_observation.observation_encoder import ThreePhaseObservationEncoder
from symm_observation.observation_ssl import ObservationPretrainer
from symm_observation.observation_views import observation_batch
from symm_observation.world_model import ThreePhaseWorldModel
from symm_observation.utils import read_json, write_json, seed_all, autocast_context, update_ema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.training.seed, cfg.training.cpu_threads, cfg.training.strict_determinism)
    data = Path(args.data)
    stats = read_json(data/"statistics.json")
    store = VisitStore(data/"visits.json") if args.stage == "A" else PairStore(data/"pairs.json")
    sampler = PatientSampler(store.records("train"), cfg.training.seed)
    result = {"stage": args.stage, "batch": args.batch, "updates": args.updates,
              "gpu": torch.cuda.get_device_name(), "precision": cfg.training.precision,
              "checkpoint_blocks": cfg.encoder.checkpoint_blocks if args.stage == "A" else cfg.velocity.checkpoint_blocks}
    try:
        if args.stage == "A":
            model = ObservationPretrainer(cfg, stats).cuda()
            teacher = None
            lr = cfg.training.lr_a
        else:
            encoder = ThreePhaseObservationEncoder(cfg.encoder, stats)
            model = ThreePhaseWorldModel(cfg, encoder, ConditionSchema.fit(store.records("train"))).cuda()
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            lr = cfg.training.lr_b
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=cfg.training.weight_decay)
        torch.cuda.reset_peak_memory_stats()
        seconds, losses = [], []
        for step in range(args.updates):
            torch.cuda.synchronize()
            started = time.perf_counter()
            records = sampler.batch(step, args.batch)
            batch = (observation_batch(store, records, cfg, step, "cuda") if args.stage == "A"
                     else store.batch(records, "cuda"))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context("cuda", cfg.training.precision):
                loss, parts = model(batch, step) if args.stage == "A" else model.loss(batch, step)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite profile loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.training.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if teacher is None:
                model.update_target(step+1)
            else:
                update_ema(teacher, model, cfg.training.ema_decay)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter()-started)
            losses.append(float(loss.detach()))
            del batch, loss
        result.update(passed=True, losses=losses, seconds=seconds,
                      examples_per_second=args.batch/(sum(seconds[2:])/len(seconds[2:])),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                      device_total_mib=torch.cuda.get_device_properties(0).total_memory/1024**2)
    except torch.OutOfMemoryError:
        result.update(passed=False, reason="cuda_out_of_memory")
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
