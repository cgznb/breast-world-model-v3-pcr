"""Run the independent physical-ROI, causal response-model optimization study."""

from __future__ import annotations

import argparse
import copy
import fcntl
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gpu_admission(cfg):
    from src.single_phase_response_data import check_pause, progress
    while True:
        free = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free >= cfg["minimum_gpu_free_mib"]:
            return
        progress(cfg, "waiting_for_gpu0_memory", free_mib=free, required_mib=cfg["minimum_gpu_free_mib"], existing_jobs_preserved=True)
        check_pause(cfg)
        time.sleep(20)


def run_tasks(cfg, tasks, stage):
    from src.single_phase_response_data import check_pause, progress
    from src.single_phase_response_training import run_task, task_complete
    pending = [t for t in tasks if not task_complete(cfg, t)]
    completed = len(tasks) - len(pending)
    progress(cfg, stage, completed=completed, total=len(tasks))
    if not pending:
        return
    adapting = any(cfg["arms"][t["arm"]].get("lora") for t in pending)
    if adapting and not all(cfg["arms"][t["arm"]].get("lora") for t in pending):
        raise ValueError("Adapter and CPU-head tasks must be scheduled separately")
    workers = 1 if adapting else cfg["training_jobs"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        queue, running = iter(pending), set()

        def submit():
            task = next(queue, None)
            if task is not None:
                check_pause(cfg)
                if adapting:
                    gpu_admission(cfg)
                running.add(pool.submit(run_task, cfg, task))

        for _ in range(min(workers, len(pending))):
            submit()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                running.remove(future)
                future.result()
                completed += 1
                submit()
            progress(cfg, stage, completed=completed, total=len(tasks), concurrent_tasks=workers)
            check_pause(cfg)


def smoke(cfg, study):
    import numpy as np
    import pandas as pd
    import torch
    from src.first_post_pcr_data import read_json, write_json
    from src.single_phase_fmbcmri_data import load_encoder
    from src.single_phase_response_data import extract_real, frozen_tokens, load_data, progress, real_input, sample_volume, subset
    from src.single_phase_response_models import ResponseModel
    from src.single_phase_response_training import fit, predict
    root = Path(cfg["output_dir"])
    if (root / "SMOKE_COMPLETE.json").exists():
        return read_json(root / "SMOKE_COMPLETE.json")
    metadata = pd.read_csv(root / "metadata.csv", dtype={"pid": str}).set_index("pid")
    observed_t0 = {r["canonical_patient_id"] for r in study["records"] if r["timepoint"] == 0}
    ids = []
    for label in (0, 1):
        ids.extend([p for p in study["split"]["train"]
                    if metadata.loc[p, "pCR"] == label and p in observed_t0 and p not in study["conflicts"]][:4])
    gpu_admission(cfg)
    extract_real(cfg, study, "train", patient_ids=ids)
    data = load_data(cfg, study, "context", "train", patient_ids=ids)
    record = next(r for r in study["records"] if r["canonical_patient_id"] == ids[0] and r["timepoint"] == 0)
    loc = study["localization"][f"{ids[0]}_T0"]
    array, center, spacing = real_input(cfg, record, loc)
    volume = sample_volume(array, center, spacing, loc["sides_mm"]["context"])[0][None].to("cuda:0")
    encoder = load_encoder(cfg, "cuda:0")
    with torch.no_grad():
        tokens, feature = frozen_tokens(encoder, volume)
        direct = encoder.forward_features(volume)[:, 0].float().cpu()
        from_cache = encoder.norm(encoder.blocks[-1](tokens.to("cuda:0")))[:, 0].float().cpu()
    token_error = float(max((feature - direct).abs().max(), (feature - from_cache).abs().max()))
    encoder_check = dict(parameters=sum(p.numel() for p in encoder.parameters()),
                         trainable_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                         training=encoder.training, token_shape=list(tokens.shape),
                         feature_shape=list(feature.shape), token_forward_error=token_error)
    if token_error > 1e-6 or encoder_check["trainable_parameters"] or encoder.training:
        raise ValueError("Frozen encoder token decomposition differs from direct inference")
    del encoder, tokens, feature, direct, from_cache, volume
    torch.cuda.empty_cache()
    train, val = subset(data, ids[:3] + ids[4:7]), subset(data, [ids[3], ids[7]])
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["training"].update(epochs=2, patience=2, batch_size=4)
    smoke_cfg["adaptation"].update(epochs=2, batch_size=2)
    checks = []
    for name in ("image_linear_context", "image_gru_context", "fusion_gru_context", "fusion_gru_context_lora"):
        arm = smoke_cfg["arms"][name]
        device = "cuda:0" if arm.get("lora") else "cpu"
        encoder = load_encoder(cfg, "cpu") if arm.get("lora") else None
        if arm.get("lora"):
            gpu_admission(cfg)
        a, first = fit(train, None, smoke_cfg, arm, 142, root / "smoke" / name / "continuous", fixed_epochs=2, encoder=encoder, device=device)
        try:
            fit(train, None, smoke_cfg, arm, 142, root / "smoke" / name / "recovered", fixed_epochs=2, encoder=encoder, device=device, pause_after=1)
        except InterruptedError:
            pass
        b, resumed = fit(train, None, smoke_cfg, arm, 142, root / "smoke" / name / "recovered", fixed_epochs=2, encoder=encoder, device=device)
        for kind in ("head", "adapter"):
            if first["state"][kind] is not None:
                if not all(torch.equal(v, resumed["state"][kind][k]) for k, v in first["state"][kind].items()):
                    raise ValueError("Real-data optimizer/RNG recovery is not exact")
        if first["history"] != resumed["history"]:
            raise ValueError("Resumed real-data learning curves differ")
        pa = predict(a, val, first["clinical"], device, 2)[0]
        pb = predict(b, val, resumed["clinical"], device, 2)[0]
        reload = ResponseModel(arm, smoke_cfg["training"], first["prevalence"], encoder, smoke_cfg["adaptation"]).to(device)
        reload.load_portable_state(first["state"])
        pc = predict(reload, val, first["clinical"], device, 2)[0]
        error = float(max(np.abs(pa - pb).max(), np.abs(pa - pc).max()))
        if error > 1e-6:
            raise ValueError("Real-data portable model replay failed")
        checks.append(dict(arm=name, exact_recovery=True, prediction_replay_error=error, gradient_audit=first["gradient_audit"]))
        del a, b, reload, encoder
        torch.cuda.empty_cache()
    result = dict(passed=True, checks=checks, encoder=encoder_check, development_patients=8, holdout_read=False)
    write_json(root / "SMOKE_COMPLETE.json", result)
    progress(cfg, "real_data_smoke_passed", checks=len(checks), includes_fold_local_lora=True)
    return result


def workflow(configs, studies, mode):
    import torch
    from src.first_post_pcr_data import identity, now, write_json
    from src.single_phase_fmbcmri_data import unchanged_json
    from src.single_phase_response_data import extract_real, progress
    from src.single_phase_response_quality import quality
    for cfg, study in zip(configs, studies):
        quality(cfg, study)
    if mode == "prepare":
        return
    for cfg, study in zip(configs, studies):
        gpu_admission(cfg)
        torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"], 0)
        smoke(cfg, study)
    if mode == "smoke":
        return
    for cfg, study in zip(configs, studies):
        gpu_admission(cfg)
        extract_real(cfg, study, "train")
    source_files = [Path(__file__), *sorted((ROOT / "src").glob("single_phase_response_*.py"))]
    unchanged_json(Path(configs[0]["study_dir"]) / "runtime_sources.json",
                   {str(p.relative_to(ROOT)): identity(p) for p in source_files})
    from src.single_phase_response_training import formal_tasks, inner_tasks
    from src.single_phase_response_reporting import freeze_models, report
    from src.single_phase_response_generated import extract_generated
    # Both branches finish their frozen-image controls before any adapter comparison.
    for cfg, study in zip(configs, studies):
        names = [name for name, arm in cfg["arms"].items() if not arm.get("lora")]
        run_tasks(cfg, [t for t in inner_tasks(cfg) if t["arm"] in names], "training_frozen_inner")
        run_tasks(cfg, formal_tasks(cfg, names), "training_frozen_formal")
        report(cfg, study, arm_names=names)
    for cfg, study in zip(configs, studies):
        names = [name for name, arm in cfg["arms"].items() if arm.get("lora")]
        run_tasks(cfg, [t for t in inner_tasks(cfg) if t["arm"] in names], "training_lora_inner")
        run_tasks(cfg, formal_tasks(cfg, names), "training_lora_formal")
        refs = freeze_models(cfg)
        write_json(Path(cfg["output_dir"]) / "TRAINING_COMPLETE.json", dict(complete=True, formal_models=len(refs), inner_fits=len(inner_tasks(cfg)), holdout_used=False))
    for cfg, study in zip(configs, studies):
        gpu_admission(cfg)
        extract_real(cfg, study, "val")
        extract_generated(cfg, study)
        report(cfg, study, include_holdout=True)
        write_json(Path(cfg["output_dir"]) / "COMPLETE.json", dict(complete=True, completed_utc=now(),
                   bootstrap=False, branches_independent=True, sources=["real", "symm", "bifm", "copy"]))
        progress(cfg, "complete")
    from src.single_phase_fmbcmri_data import BRANCHES
    if all((Path(configs[0]["study_dir"]) / branch / "COMPLETE.json").exists() for branch in BRANCHES):
        write_json(Path(configs[0]["study_dir"]) / "COMPLETE.json", dict(complete=True, completed_utc=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_response_pcr_v2.yaml")
    parser.add_argument("--branch", choices=("both", "registered_dce0", "unregistered_first_post"), default="both")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "run", "detach"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from src.first_post_pcr_data import identity, now, write_json
    from src.single_phase_fmbcmri_data import BRANCHES
    from src.single_phase_fmbcmri_training import deterministic_runtime
    from src.single_phase_response_data import check_pause, load_config, prepare, progress
    deterministic_runtime()
    configs = [load_config(args.config, b) for b in (BRANCHES if args.branch == "both" else (args.branch,))]
    output = Path(configs[0]["study_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", configs[0]["config_path"], "--branch", args.branch, "--run"]
        if args.resume:
            command.append("--resume")
        with (output / "workflow.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
        write_json(output / "launch.json", dict(pid=process.pid, gpu=0, created_utc=now(), command=command,
                   existing_jobs_preserved=True, log=str(output / "workflow.log")))
        print(f"Launched response study PID {process.pid} on GPU0; {output / 'workflow.log'}", flush=True)
        return
    with (output / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An independent response-study controller is already running")
        def pause(signum, frame):
            for cfg in configs:
                write_json(Path(cfg["output_dir"]) / "PAUSE_REQUESTED", dict(signal=signum, updated_utc=now()))
        signal.signal(signal.SIGTERM, pause)
        signal.signal(signal.SIGINT, pause)
        if args.resume:
            for cfg in configs:
                (Path(cfg["output_dir"]) / "PAUSE_REQUESTED").unlink(missing_ok=True)
        mode_name = "prepare" if args.prepare else "smoke" if args.smoke else "run"
        try:
            write_json(output / "controller_status.json", dict(stage="running", mode=mode_name, pid=os.getpid(), updated_utc=now()))
            for cfg in configs:
                check_pause(cfg)
            studies = [prepare(cfg) for cfg in configs]
            workflow(configs, studies, mode_name)
            write_json(output / "controller_status.json", dict(stage=f"{mode_name}_complete", pid=os.getpid(), updated_utc=now()))
        except BaseException as error:
            write_json(output / "controller_status.json", dict(stage="paused" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                       error_type=type(error).__name__, error=str(error), updated_utc=now()))
            raise


if __name__ == "__main__":
    main()
