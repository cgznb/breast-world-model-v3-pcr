"""Freeze the development-supported residual-penalty candidate, then evaluate once."""

from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from scripts.run_full978_anti_overfit import _atomic_csv
from scripts.verify_registered_three_phase_overfit_ablation import auc
from src import first_post_optimization as opt
from src import registered_three_phase_fixed_pcr as fixed
from src import registered_three_phase_fixed_report as reporter
from src import registered_three_phase_optimization as shared
from src import registered_three_phase_overfit_ablation as ablation
from src.first_post_pcr_data import identity, now, public, read_json, write_json


def verify(cfg, summary):
    output = Path(cfg["output_dir"]) / "evaluation/holdout"
    raw = pd.read_csv(output / "draw_fold_predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
    mc4 = pd.read_csv(output / "mc4_fold_predictions.csv", dtype={"patient_id": str}, float_precision="round_trip")
    metrics = pd.read_csv(output / "fold_metrics.csv", float_precision="round_trip")
    keys = ["model", "depth", "seed", "outer", "patient_id", "label", "source"]
    old = pd.read_csv(Path(cfg["reference_run"]) / "evaluation/holdout/draw_fold_predictions.csv",
                      dtype={"patient_id": str}, float_precision="round_trip")
    old = old[(old.model == "reference") & (old.depth == 4)].set_index(keys).sort_index()
    reference = raw[raw.model == "reference"].set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index, reference.index)
    reference_error = float(np.max(np.abs(old.probability - reference.probability)))
    assert reference_error < 1e-12
    errors = []
    base_keys = keys[:-1]
    for mode in ("direct", "rollout", "previous_real"):
        draws = raw[raw.source.str.startswith(mode + "_draw_")].pivot(
            index=base_keys, columns="source", values="probability").sort_index()
        assert draws.shape == (10200, 4) and not draws.isna().any().any()
        expected = draws.to_numpy(dtype=float).sum(axis=1) / 4
        actual = mc4[mc4.source == mode + "_mc4"].set_index(base_keys).sort_index().probability
        pd.testing.assert_index_equal(draws.index, actual.index)
        errors.append(float(np.max(np.abs(expected - actual.to_numpy()))))
    metric_keys = ["model", "depth", "seed", "outer", "source"]
    saved = metrics.set_index(metric_keys)
    for key, part in mc4.groupby(metric_keys):
        errors.append(abs(auc(part.label, part.probability) - saved.loc[key, "auroc"]))
    for row in summary.itertuples():
        part = metrics[(metrics.model == row.model) & (metrics.depth == row.depth)
                       & (metrics.seed == row.seed) & (metrics.source == row.source)]
        assert set(part.outer) == set(range(5)) and len(part) == 5
        errors.extend([abs(float(part.auroc.mean()) - row.auroc_mean),
                       abs(float(part.auroc.std(ddof=1)) - row.auroc_fold_sd)])
    assert max(errors) < 1e-12
    return {"raw_prediction_rows": len(raw), "mc4_prediction_rows": len(mc4),
            "fold_metrics": len(metrics), "seed_summaries": len(summary),
            "reference_predictions_match_previous_evaluation": reference_error,
            "independent_auc_mc4_and_summary_max_error": max(errors)}


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    study = ablation.load_config("configs/registered_three_phase_overfit_ablation_v4.yaml")
    parent = Path(study["output_dir"])
    output = parent / "candidate_followup"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "COMPLETE.json").exists():
            shared.verify_inventory(read_json(output / "FROZEN_MODELS.json")["artifacts"])
            shared.verify_inventory(read_json(output / "input_inventory.json"))
            print("Completed candidate evaluation verified; no inference repeated.")
            return
        assert read_json(parent / "independent_verification.json")["passed"]
        shared.verify_inventory(read_json(parent / "input_inventory.json"))
        shared.verify_inventory(read_json(parent / "FROZEN_MODELS.json")["artifacts"])
        development = pd.read_csv(parent / "paired_comparisons.csv", float_precision="round_trip")
        chosen = development[development.arm == "residual_penalty"]
        assert len(chosen) == 10 and (chosen.delta_validation_auroc > 0).all()
        assert (chosen.delta_validation_logloss < 0).all()
        cfg = fixed.load_config(study["reference_config"])
        cfg.update(output_dir=str(output), reference_run=study["reference_run"], depths=[4],
                   bootstrap_samples=study["bootstrap_samples"], bootstrap_seed=study["bootstrap_seed"])
        refs = []
        for ref in read_json(parent / "model_references.json"):
            if ref["arm"] in ("reference", "residual_penalty"):
                refs.append({**ref, "arm": "primary" if ref["arm"] == "residual_penalty" else "reference", "depth": 4})
        assert len(refs) == 100
        protocol = {
            "primary": "T0-T3 residual_penalty 0.1, otherwise identical to the fixed40 reference",
            "decision": "Follow-up chosen after the four-arm development ablation; ten seeds improve validation AUC and logloss.",
            "not_improved": "Training-validation AUC gap increased; overfitting is not established as resolved.",
            "all_seeds_retained": list(range(42, 52)), "depths": [4], "reference": study["reference_run"],
            "model_selection_for_this_followup_uses_new_holdout_results": False,
            "historical_reuse": "These102patients were generator validation and repeatedly analyzed; exploratory only.",
            "new_training_or_image_generation": False, "no_further_tuning_after_this_evaluation": True,
            "config": public(cfg), "development_evidence": str(parent / "paired_comparisons.csv"),
        }
        shared.frozen_json(output / "protocol.json", protocol)
        shared.frozen_json(output / "nested_folds.json", read_json(parent / "nested_folds.json"))
        shared.frozen_json(output / "model_references.json", refs)
        inventory = read_json(Path(study["reference_run"]) / "evaluation_input_inventory.json")
        inventory.update(read_json(parent / "input_inventory.json"))
        for path in (Path(__file__), repo_path_verifier(), parent / "paired_comparisons.csv",
                     parent / "independent_verification.json", output / "protocol.json",
                     output / "model_references.json", output / "nested_folds.json"):
            inventory[str(path)] = identity(path)
        shared.frozen_json(output / "input_inventory.json", inventory)
        shared.verify_inventory(inventory)
        artifacts = {str(Path(r["path"]) / "model.pt"): identity(Path(r["path"]) / "model.pt") for r in refs}
        shared.frozen_json(output / "FROZEN_MODELS.json", {"holdout_used_for_selection": False, "artifacts": artifacts})
        shared.frozen_json(output / "TRAINING_COMPLETE.json", {"training_complete": True,
                            "new_fits_in_this_stage": 0, "development_audit": str(parent / "independent_verification.json")})
        summary = reporter.evaluate_holdout(cfg, refs)
        audit = verify(cfg, summary)
        intervals = pd.read_csv(output / "evaluation/holdout/paired_seed_intervals.csv")
        comparison = intervals[(intervals.left_source == intervals.right_source)
                               & (intervals.left_model == "primary") & (intervals.right_model == "reference")]
        lines = ["# Fixed Residual-Penalty Candidate: Historical Follow-up", "",
                 "Candidate chosen using the completed development ablation, before this follow-up evaluation.",
                 "T0-T3 only, 40 epochs, five models per seed, same102patients/32positives and original MC4 inputs.",
                 "Primary means residual L2 penalty0.1 with the original balanced loss; reference means the matched fixed40 model.",
                 "The original v1 peak-selected model is a different comparator and its result files remain unchanged.", ""]
        for source in fixed.SOURCES:
            lines += [f"## {source}", "", "| Seed | Fixed40 reference | Residual penalty | AUC difference | Paired95% interval |",
                      "|---|---:|---:|---:|---:|"]
            for seed in cfg["formal_seeds"]:
                part = summary[(summary.seed == seed) & (summary.source == source)].set_index("model")
                contrast = comparison[(comparison.seed == seed) & (comparison.left_source == source)].iloc[0]
                ref, candidate = part.loc["reference"], part.loc["primary"]
                lines.append(f"| {seed} | {ref.auroc_mean:.4f} +/- {ref.auroc_fold_sd:.4f} | "
                             f"{candidate.auroc_mean:.4f} +/- {candidate.auroc_fold_sd:.4f} | "
                             f"{contrast.mean_fold_auc_difference:+.4f} | [{contrast.ci95_low:+.4f}, {contrast.ci95_high:+.4f}] |")
        lines += ["", "## Interpretation", "",
                  "These are mean-of-five-model AUCs, not AUC after averaging five probabilities.",
                  "MC4 averages four generated sequence probabilities within each model before scoring.",
                  "Intervals condition on fitted models and omit training uncertainty/multiplicity adjustment.",
                  "The candidate's development training-validation gap did not improve. AUC gains do not prove overfitting is cured.",
                  "Historical generator-validation reuse precludes an untouched end-to-end test claim.",
                  "No seed is selected and these outcomes will not trigger more tuning in this experiment.", "",
                  "[All fold scores](evaluation/holdout/fold_metrics.csv) | [Per-seed summary](evaluation/holdout/seed_fivefold_summary.csv)",
                  "[Probability ensemble, separate statistic](evaluation/holdout/probability_ensemble_metrics.csv)", ""]
        (output / "README.md").write_text("\n".join(lines))
        _atomic_csv(output / "candidate_vs_reference.csv", comparison)
        with pd.ExcelWriter(output / "historical_candidate.xlsx") as workbook:
            summary.to_excel(workbook, sheet_name="Per Seed Five Models", index=False)
            pd.read_csv(output / "evaluation/holdout/fold_metrics.csv").to_excel(workbook, sheet_name="Individual Models", index=False)
            comparison.to_excel(workbook, sheet_name="Paired Comparisons", index=False)
        shared.verify_inventory(inventory)
        shared.verify_inventory(artifacts)
        write_json(output / "COMPLETE.json", {"passed": True, "completed_utc": now(), **audit,
                   "new_training": False, "new_image_generation": False, "historical_exploratory_evaluation": True})
        print(output / "README.md")
        print(audit)


def repo_path_verifier():
    return Path(__file__).resolve().with_name("verify_registered_three_phase_overfit_ablation.py")


if __name__ == "__main__":
    main()
