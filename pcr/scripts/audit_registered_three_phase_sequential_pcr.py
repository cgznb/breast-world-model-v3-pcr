"""Audit saved sequential forecasts, source boundaries and pCR aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src import registered_three_phase_sequential_pcr as study
from src.first_post_pcr_data import identity, now, read_json, write_json
from src.single_phase_repeat_pcr import pillar_identity

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/registered_three_phase_roi32_sequential_pcr_v1_20260918"
DIRECT = ROOT / "results/registered_three_phase_roi32_generated_pcr_v1_20260918"
PCR = ROOT / "results/registered_three_phase_roi32_pcr_v1_20260917"


def audit_generation(cfg, routes):
    protocol = read_json(OUTPUT / "protocol.json")
    for section in ("sources", "inputs"):
        for path, expected in protocol[section].items():
            assert identity(path) == expected
    for ref in protocol["classifiers"]:
        assert identity(ref["path"]) == ref["identity"]
    assert pillar_identity() == read_json(PCR / "input_contract.json")["pillar_files"]
    by_key = {(r["patient_id"], r["timepoint"]): r for r in routes}
    seen, counts, target_differences = set(), {"shared_T1": 0, "rollout": 0, "previous_real": 0}, []
    for route in routes:
        pid, tp = route["patient_id"], route["timepoint"]
        for mode in study.MODES:
            path = study.case_path(cfg, mode, route)
            if path in seen:
                continue
            seen.add(path)
            record = study.case_record(cfg, mode, route)
            assert record is not None
            latent = torch.load(record["latent_path"], map_location="cpu", weights_only=True)
            assert latent.shape == study.LATENT_SHAPE and torch.isfinite(latent).all()
            with np.load(record["image_path"], allow_pickle=False) as image:
                samples, foreground = image["samples"], image["source_foreground"]
                assert samples.shape == (4, 3, 32, 128, 128) and np.isfinite(samples).all()
                assert foreground.shape == samples.shape[1:] and foreground.reshape(3, -1).any(axis=1).all()
            t1_route = by_key[pid, 1]
            with np.load(DIRECT / "volumes" / (t1_route["key"] + ".npz"), allow_pickle=False) as t1:
                assert np.array_equal(foreground, t1["source_foreground"])
            guard = record["source_read_audit"]
            assert guard["passed"] and guard["target_image_or_latent_reads"] == 0 and not guard["target_mask_used"]
            allowed_names = {pid + "_T0.npy"}
            if mode == "previous_real":
                allowed_names.add(pid + f"_T{tp - 1}.npy")
            assert {Path(p).name for p in guard["real_latent_files_opened"]} <= allowed_names
            if tp == 1:
                assert record["source_kind"] == "real_T0" and record["shared_T1_volume_replay_max_error"] == 0
            elif mode == "rollout":
                parent = study.case_record(cfg, mode, by_key[pid, tp - 1])
                assert record["parent_generated_latent"] == identity(parent["latent_path"])
                assert record["source_kind"] == "previous_generated_same_draw"
                other = study.case_record(cfg, "previous_real", route)
                other_latent = torch.load(other["latent_path"], map_location="cpu", weights_only=True)
                target_differences.append(float((latent - other_latent).abs().mean()))
            else:
                assert record["source_kind"] == "real_previous_visit" and record["parent_generated_latent"] is None
            counts[record["mode"]] += 1
    assert counts == {"shared_T1": 100, "rollout": 176, "previous_real": 176}
    assert min(target_differences) > 0
    return {"generated_case_counts": counts, "shared_T1_volume_replay_max_error": 0.,
            "rollout_vs_previous_real_latent_mae_min": min(target_differences),
            "source_boundaries_verified": True, "frozen_models_and_sources_unchanged": True}


def audit_features(cfg, routes):
    vectors = 0
    for route in routes:
        for mode in study.MODES:
            for draw in range(4):
                assert study.feature_complete(cfg, mode, route, draw)
                path = study.feature_path(cfg, mode, route, draw)
                vector = torch.load(path, map_location="cpu", weights_only=True)
                assert vector.shape == (1152,) and torch.isfinite(vector).all() and vector.norm() > 0
                if route["timepoint"] == 1:
                    old = DIRECT / "embeddings" / f"draw_{draw}" / route["patient_id"] / (route["key"] + ".pt")
                    assert torch.equal(vector, torch.load(old, map_location="cpu", weights_only=True))
                vectors += 1
    assert vectors == 2208
    return {"generated_feature_count": vectors, "shared_T1_features_equal": True}


def audit_scores():
    directory = OUTPUT / "evaluation"
    folds = pd.read_csv(directory / "fold_predictions.csv", dtype={"probability": np.float32})
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"probability": np.float32})
    keys = ["source", "temporal_depth", "seed", "patient_id", "label"]
    assert len(folds) == 285600 and len(predictions) == 69360
    assert not folds.duplicated(keys + ["fold"]).any() and not predictions.duplicated(keys).any()
    assert (folds.groupby(keys).fold.nunique() == 5).all()
    reproduced = folds.groupby(keys).probability.mean()
    stored = predictions.set_index(keys).probability
    fold_error = float((reproduced - stored.loc[reproduced.index]).abs().max())
    assert fold_error == 0
    for mode in ("direct", *study.MODES):
        draws = predictions[predictions.source.str.startswith(mode + "_draw_")]
        assert (draws.groupby(keys[1:]).source.nunique() == 4).all()
        mean = draws.groupby(keys[1:]).probability.mean()
        ensemble = predictions[predictions.source == mode + "_mc4"].set_index(keys[1:]).probability
        assert float((mean - ensemble).abs().max()) == 0
    old = pd.read_csv(DIRECT / "evaluation/predictions.csv", dtype={"probability": np.float32})
    for source, new_source in (("real", "real"), ("copy_T0", "copy_T0"), ("generated_mc4", "direct_mc4")):
        original = old[old.source == source].set_index(keys[1:]).probability
        current = predictions[predictions.source == new_source].set_index(keys[1:]).probability
        assert len(original) == len(current) == 4080 and float((original - current).abs().max()) == 0
    t0 = predictions[predictions.temporal_depth == "T0"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    assert float(t0.sub(t0.real, axis=0).abs().max().max()) == 0
    t1 = predictions[predictions.temporal_depth == "T0-T1"].pivot(index=["seed", "patient_id"], columns="source", values="probability")
    for mode in study.MODES:
        for suffix in ("mc4", "draw_0", "draw_1", "draw_2", "draw_3"):
            assert float((t1[f"{mode}_{suffix}"] - t1[f"direct_{suffix}"]).abs().max()) == 0
    heldout = set(read_json(PCR / "cohort.json")["split"]["val"])
    rows = []
    for (source, depth, seed), frame in predictions.groupby(["source", "temporal_depth", "seed"]):
        assert set(frame.patient_id) == heldout and len(frame) == 102 and frame.label.sum() == 32
        rows.append({"source": source, "depth": depth, "seed": seed, "auroc": roc_auc_score(frame.label, frame.probability)})
    scored = pd.DataFrame(rows).groupby(["source", "depth"]).auroc.agg(["mean", "std"])
    saved = pd.read_csv(directory / "summary.csv")
    saved = saved[saved.threshold_policy == "fixed_0.5"].set_index(["source", "depth"])
    error = float(max((scored["mean"] - saved.auroc_mean).abs().max(), (scored["std"] - saved.auroc_std).abs().max()))
    assert error < 1e-12
    differences = pd.read_csv(directory / "paired_auroc_differences.csv")
    for row in differences.itertuples():
        left, right = row.comparison.split("_minus_")
        assert abs(scored.loc[(left, row.depth), "mean"] - scored.loc[(right, row.depth), "mean"] - row.auroc_difference) < 1e-12
    return {"fold_rows": len(folds), "patient_seed_window_arm_rows": len(predictions),
            "fold_and_MC4_probability_replay_max_error": fold_error,
            "all_T0_and_shared_T1_prediction_error": 0., "auroc_summary_replay_max_error": error}


def audit_intervals(cfg):
    directory = OUTPUT / "evaluation"
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"probability": np.float32})
    differences = pd.read_csv(directory / "paired_auroc_differences.csv")
    assert len(differences) == 28
    rng, errors = np.random.default_rng(cfg["bootstrap_seed"]), []
    for depth, frame in predictions.groupby("temporal_depth"):
        ids = sorted(frame.patient_id.unique())
        labels = frame.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
        npos, nneg = int(labels.sum()), int((1 - labels).sum())
        weights = np.ones((cfg["bootstrap_samples"] + 1, len(ids)), dtype=np.float64)
        weights[1:, labels == 1] = rng.multinomial(npos, np.full(npos, 1 / npos), size=cfg["bootstrap_samples"])
        weights[1:, labels == 0] = rng.multinomial(nneg, np.full(nneg, 1 / nneg), size=cfg["bootstrap_samples"])
        contrasts = differences[differences.depth == depth]
        arms = {arm for contrast in contrasts.comparison for arm in contrast.split("_minus_")}
        scored = {}
        for arm in arms:
            matrix = frame[frame.source == arm].pivot(index="seed", columns="patient_id", values="probability")[ids].to_numpy()
            samples = []
            for probabilities in matrix:
                # Recompute weighted rank AUC with tied scores grouped, independently of the pairwise kernel.
                order = np.argsort(probabilities, kind="stable")
                starts = np.r_[0, np.flatnonzero(np.diff(probabilities[order]) != 0) + 1]
                positive = np.add.reduceat(weights[:, order] * labels[order], starts, axis=1)
                negative = np.add.reduceat(weights[:, order] * (1 - labels[order]), starts, axis=1)
                auc = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(axis=1) / (npos * nneg)
                assert abs(auc[0] - roc_auc_score(labels, probabilities)) < 1e-12
                samples.append(auc)
            scored[arm] = np.mean(samples, axis=0)
        for row in contrasts.itertuples():
            assert row.bootstrap_samples == cfg["bootstrap_samples"] and row.patients == len(ids)
            left, right = row.comparison.split("_minus_")
            delta = scored[left] - scored[right]
            actual = np.r_[delta[0], np.quantile(delta[1:], [.025, .975])]
            expected = np.array([row.auroc_difference, row.ci95_low, row.ci95_high])
            errors.append(float(np.abs(actual - expected).max()))
    assert max(errors) < 1e-12
    return {"paired_bootstrap_contrasts_recomputed": len(errors),
            "paired_bootstrap_point_and_interval_max_error": max(errors)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = study.load_config("configs/registered_three_phase_sequential_pcr_v1.yaml")
    routes = read_json(OUTPUT / "routes.json")["routes"]
    report = audit_generation(cfg, routes)
    if not args.generation_only:
        report.update(audit_features(cfg, routes))
        report.update(audit_scores())
        report.update(audit_intervals(cfg))
    report.update(passed=True, checked_utc=now())
    name = "generation_audit.json" if args.generation_only else "evaluation/independent_audit.json"
    write_json(OUTPUT / name, report)
    print(json.dumps(report, indent=2))
