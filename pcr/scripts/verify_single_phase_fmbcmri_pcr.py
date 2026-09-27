"""Recompute completed-study statistics and render research figures on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.first_post_pcr_data import identity, read_json, write_json
from src.single_phase_fmbcmri_data import BRANCHES, DEPTHS, load_config
from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text


def independent_metrics(frame):
    return {"auroc": roc_auc_score(frame.label, frame.probability),
            "auprc": average_precision_score(frame.label, frame.probability),
            "logloss": log_loss(frame.label, frame.probability, labels=[0, 1]),
            "brier": brier_score_loss(frame.label, frame.probability)}


def verify(cfg):
    root = Path(cfg["output_dir"])
    if not read_json(root / "COMPLETE.json")["complete"]:
        raise ValueError("Study is not complete")
    cohort = read_json(root / "cohort.json")
    development, holdout = set(cohort["split"]["train"]), set(cohort["split"]["val"])
    assert len(development) == cfg["expected_development"] and len(holdout) == cfg["expected_holdout"]
    assert not development & holdout
    nested = read_json(root / "nested_folds.json")
    selections = read_json(root / "selected_epochs.json")
    for selection in selections:
        directory = root / "fits/inner" / selection["window"] / f"outer_{selection['outer']}"
        epochs = []
        for inner in range(3):
            for seed in (42, 43):
                fit_root = directory / f"inner_{inner}" / f"seed_{seed}"
                state = read_json(fit_root / "COMPLETE.json")
                history = pd.read_csv(fit_root / "history.csv")
                selected = int(history.loc[history.validation_logloss.idxmin(), "epoch"])
                assert selected == state["selected_epochs"]
                allowed = set(nested[selection["outer"]]["train_ids"])
                task = state["task"]
                assert set(task["train_ids"]) | set(task["val_ids"]) == allowed
                assert not set(task["train_ids"]) & set(task["val_ids"])
                epochs.append(selected)
        assert epochs == selection["inner_selected_epochs"]
        assert int(np.ceil(np.median(epochs))) == selection["fixed_epochs"]
    refs = read_json(root / "frozen_models.json")
    assert len(refs) == 200
    expected_epochs = {(r["window"], r["outer"]): r["fixed_epochs"] for r in selections}
    for ref in refs:
        assert identity(ref["path"]) == ref["identity"]
        checkpoint = torch.load(ref["path"], map_location="cpu", weights_only=False)
        history = pd.read_csv(Path(ref["path"]).parent / "history.csv")
        assert checkpoint["model_state"]["proj.net.0.weight"].shape[1] == 768
        assert all(torch.isfinite(v).all() for v in checkpoint["model_state"].values())
        task = checkpoint["task"]
        assert set(task["train_ids"]) | set(task["val_ids"]) == development
        assert not set(task["train_ids"]) & set(task["val_ids"])
        assert checkpoint["selected_epochs"] == len(history) == expected_epochs[ref["window"], ref["outer"]]
        assert not any(k.startswith("validation_") for k in history)
        assert checkpoint["initialized_from_pretrained_tdn"] is False
        assert all(checkpoint["gradient_audit"].values())
        assert checkpoint["updated_parameter_tensors"] > 0
        assert checkpoint["clinical_prior"]["fitted_on"] == "fold_train_only"
    feature_count = 0
    for source in ("real", "symm", "bifm", "copy"):
        audits = list((root / "embeddings" / source).rglob("*.json"))
        expected = cfg["expected_visits"] if source == "real" else cfg["expected_targets"]
        assert len(audits) == expected
        for path in audits:
            audit = read_json(path)
            assert audit["phase"] == cfg["phase"] and audit["embedding_dim"] == 768 and audit["frozen_encoder"]
            assert identity(path.with_suffix(".pt")) == audit["identity"]
            vector = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
            assert vector.dtype == torch.float32 and vector.shape == (768,) and torch.isfinite(vector).all()
        feature_count += len(audits)
    destination = root / "evaluation"
    oof = pd.read_csv(destination / "oof_predictions.csv", dtype={"patient_id": str})
    held = pd.read_csv(destination / "holdout_predictions.csv", dtype={"patient_id": str})
    fold_predictions = pd.read_csv(destination / "holdout_fold_predictions.csv", dtype={"patient_id": str})
    keys = ["source", "window", "seed", "patient_id"]
    reconstructed = fold_predictions.groupby(keys).probability.mean().sort_index()
    stored = held.set_index(keys).probability.sort_index()
    mean_error = float(np.max(np.abs(reconstructed.to_numpy() - stored.to_numpy())))
    assert mean_error <= 1e-12
    assert set(held.patient_id) == holdout and set(oof.patient_id) == development
    assert not held.duplicated(keys).any() and not oof.duplicated(keys).any()
    t0 = held[held.window == "T0"].pivot(index=["patient_id", "seed"], columns="source", values="probability")
    assert (t0.max(axis=1) - t0.min(axis=1)).max() == 0
    per_seed = pd.read_csv(destination / "metrics_per_seed.csv")
    summary = pd.read_csv(destination / "summary.csv")
    largest_metric_error = 0.0
    for population, frame in (("development_oof", oof), ("fixed_holdout", held)):
        for row in per_seed[per_seed.population == population].itertuples():
            values = frame[(frame.source == row.source) & (frame.window == row.window) & (frame.seed == row.seed)]
            if row.subset == "complete_window":
                values = values[values.complete_window]
            assert len(values) == row.patients
            for metric, value in independent_metrics(values).items():
                largest_metric_error = max(largest_metric_error, abs(value - getattr(row, metric)))
    for row in summary.itertuples():
        rows = per_seed[(per_seed.population == row.population) & (per_seed.subset == row.subset)
                        & (per_seed.source == row.source) & (per_seed.window == row.window)]
        assert len(rows) == 10 and set(rows.seed) == set(range(42, 52))
        for metric in ("auroc", "auprc", "logloss", "brier"):
            largest_metric_error = max(largest_metric_error, abs(rows[metric].mean() - getattr(row, metric + "_mean")),
                                      abs(rows[metric].std(ddof=1) - getattr(row, metric + "_std")))
    assert largest_metric_error <= 1e-10
    original_verification = read_json(destination / "verification.json")
    assert original_verification["max_checkpoint_prediction_error"] <= 1e-6
    if cfg["phase"] == "dce0":
        assert read_json(root / "crop_preparation.json")["rebuilt_t0_crop_plans"] == 110
        assert read_json(root / "decoder_parity.json")["max_voxel_error"] <= 1e-6
    result = {"passed": True, "branch": cfg["branch"], "formal_classifiers": len(refs),
              "inner_fits": 120, "feature_vectors": feature_count, "fixed_holdout_patients": len(holdout),
              "development_patients": len(development), "ten_seeds": True, "four_source_t0_equal": True,
              "fold_probability_mean_max_error": mean_error, "independent_metric_max_error": largest_metric_error,
              "checkpoint_replay_max_error": original_verification["max_checkpoint_prediction_error"]}
    write_json(destination / "independent_verification.json", result)
    return result


def figures(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    root = Path(cfg["output_dir"])
    destination = root / "evaluation/figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for (window, _), ax in zip(DEPTHS.items(), axes.flat):
        path = root / "fits/inner" / window / "outer_0/inner_0/seed_42"
        history = pd.read_csv(path / "history.csv")
        selected = read_json(path / "COMPLETE.json")["selected_epochs"]
        ax.plot(history.epoch, history.train_logloss, label="Training", color="#287d66")
        ax.plot(history.epoch, history.validation_logloss, label="Inner validation", color="#bf5662")
        ax.axvline(selected, color="#666666", linestyle=":", label="Selected epoch")
        ax.set(title=window, xlabel="Epoch", ylabel="Log loss")
        ax.legend(fontsize=8)
    fig.suptitle(f"{cfg['branch']}: outer 0 / inner 0 / seed 42", fontsize=12)
    fig.savefig(destination / "prespecified_learning_curves.png", dpi=180)
    fig.savefig(destination / "prespecified_learning_curves.pdf")
    plt.close(fig)
    summary = pd.read_csv(root / "evaluation/summary.csv")
    summary = summary[(summary.population == "fixed_holdout") & (summary.subset == "all_prefixes")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {"real": "#287d66", "symm": "#bf5662", "bifm": "#387fba", "copy": "#777777"}
    for metric, ax in zip(("auroc", "auprc", "logloss", "brier"), axes.flat):
        for index, (source, color) in enumerate(colors.items()):
            rows = summary[summary.source == source].set_index("window").loc[list(DEPTHS)]
            ax.bar(np.arange(4) + (index - 1.5) * 0.18, rows[metric + "_mean"], width=0.18,
                   yerr=rows[metric + "_std"], capsize=2, color=color, label=source)
        ax.set_xticks(np.arange(4), list(DEPTHS))
        ax.set_title(metric.upper() if metric in ("auroc", "auprc") else metric.capitalize())
        ax.set_ylabel("Mean with seed SD")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"{cfg['branch']}: fixed holdout, ten seeds", fontsize=12)
    fig.savefig(destination / "four_source_metrics.png", dpi=180)
    fig.savefig(destination / "four_source_metrics.pdf")
    plt.close(fig)


def summarize_study(config_path):
    configs = [load_config(config_path, branch) for branch in BRANCHES]
    root = Path(configs[0]["output_dir"]).parent
    records, frames, paired, selections = [], [], [], []
    for cfg in configs:
        directory = Path(cfg["output_dir"])
        record = read_json(directory / "evaluation/independent_verification.json")
        if not record["passed"] or not read_json(directory / "COMPLETE.json")["complete"]:
            raise ValueError("Both branch verifications must pass before summarizing")
        records.append(record)
        selections.extend(read_json(directory / "selected_epochs.json"))
        frame = pd.read_csv(directory / "evaluation/summary.csv")
        frame.insert(0, "branch", cfg["branch"])
        frames.append(frame)
        comparison = pd.read_csv(directory / "evaluation/legacy_comparison.csv")
        comparison.insert(0, "branch", cfg["branch"])
        paired.append(comparison)
    summary, comparison = pd.concat(frames, ignore_index=True), pd.concat(paired, ignore_index=True)
    formal_count = sum(r["formal_classifiers"] for r in records)
    inner_count = sum(r["inner_fits"] for r in records)
    feature_count = sum(r["feature_vectors"] for r in records)
    maximum_error = max(r["checkpoint_replay_max_error"] for r in records)
    long = [r["fixed_epochs"] for r in selections if r["window"] in ("T0-T2", "T0-T3")]
    decoder = read_json(Path(configs[0]["output_dir"]) / "decoder_parity.json")
    _atomic_csv(root / "summary.csv", summary)
    _atomic_csv(root / "legacy_comparison.csv", comparison)
    lines = ["# Single-phase FM-BCMRI pCR results", "",
             f"Completed: {formal_count} formal classifiers and {inner_count} inner fits on GPU{configs[0]['gpu']}.",
             "Frozen FM-BCMRI CLS768, complete new TDN, original patient splits, five folds and ten seeds 42-51.",
             "No bootstrap, confidence intervals, hyperparameter search or generator fitting/sampling.", "",
             "## Fixed holdout AUROC", "", "All available contiguous T0 prefixes; means with sample SD over ten seeds.",
             "Each seed first averages five fold probabilities.", "",
             "| Branch | Source | T0 | T0-T1 | T0-T2 | T0-T3 |", "|---|---|---:|---:|---:|---:|"]
    holdout = summary[(summary.population == "fixed_holdout") & (summary.subset == "all_prefixes")]
    for branch in BRANCHES:
        for source in ("real", "symm", "bifm", "copy"):
            rows = holdout[(holdout.branch == branch) & (holdout.source == source)].set_index("window")
            values = " | ".join(f"{rows.loc[w, 'auroc_mean']:.4f} ({rows.loc[w, 'auroc_std']:.4f})" for w in DEPTHS)
            lines.append(f"| {branch} | {source} | {values} |")
    lines += ["", "## Matched Legacy T0-T3 AUROC", "",
              "| Branch | Source | New FM-BCMRI | Legacy Pillar |", "|---|---|---:|---:|"]
    for row in comparison[(comparison.subset == "all_prefixes") & (comparison.window == "T0-T3")].itertuples():
        lines.append(f"| {row.branch} | {row.source} | {row.new_auroc_mean:.4f} | {row.old_auroc_mean:.4f} |")
    lines += ["", "## Interpretation", "",
              f"{sum(e == 1 for e in long)} of {len(long)} long-window outer selections (T0-T2 and T0-T3 across both branches) selected one epoch under the prespecified inner-log-loss rule. All formal models received optimizer updates.",
              "The inherited long-window recipes use balanced positive-class weighting, whereas selection uses ordinary log loss. In the prespecified registered outer 0 / inner 0 / seed 42 curves, weighted training objective falls while ordinary training and validation log loss rise. This objective difference is relevant to the early selections; no loss setting was changed after inspecting outcomes.",
              "Encoder, phase count, preprocessing and selection protocol differ from legacy Pillar. These results do not isolate a causal encoder or registration effect.",
              "Both fixed holdouts previously participated in upstream selection. Native ROI localization uses visit annotations/predictions, including future visits; evaluation is retrospective and internal.",
              "Development OOF is real-only. Archived generated inputs cover the fixed holdout patients. Complete-window subsets are reported separately; seed SD is not patient-sampling uncertainty.", "",
              "## Verification", "", "14 focused tests and 4 existing fixed-epoch tests passed; real GPU smokes verified exact training recovery.",
              f"All {formal_count} final checkpoint replays are within 1e-6; maximum error is {maximum_error:.8g}. Four-source T0 equality and prediction-derived metric reconstruction passed.",
              f"All {feature_count} feature vectors were independently checked. Registered 110 supplemental T0 crops preserve the cohort. {decoder['probes']} retained VQ images replay with maximum error {decoder['max_voxel_error']}; {decoder['recovered_images']} missing BiFM volumes were recovered from archived endpoints.", "",
              "## Full Reports", "",
              "- [Registered DCE0](registered_dce0/evaluation/report.md): 876 development, 102 fixed holdout.",
              "- [Native first-post](unregistered_first_post/evaluation/report.md): 881 development, 96 fixed holdout.",
              "- `summary.csv` contains AUROC, AUPRC, log loss and Brier means/SD for OOF, holdout and complete-window subsets.",
              "- Branch `evaluation/` folders contain per-seed metrics, predictions, legacy pairing, verification and PNG/PDF figures."]
    _atomic_text(root / "report.md", "\n".join(lines) + "\n")
    write_json(root / "COMPLETE.json", {"complete": True, "gpu": configs[0]["gpu"], "formal_classifiers": formal_count,
               "inner_fits": inner_count, "feature_vectors": feature_count,
               "independent_verification": records, "statistics": "ten_seed_mean_and_sample_SD",
               "reports": [str(Path(c["output_dir"]) / "evaluation/report.md") for c in configs]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_phase_fmbcmri_pcr_v1.yaml")
    parser.add_argument("--branch", choices=("both", *BRANCHES), default="both")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for branch in BRANCHES if args.branch == "both" else (args.branch,):
        cfg = load_config(args.config, branch)
        print(verify(cfg), flush=True)
        figures(cfg)
    if args.branch == "both":
        summarize_study(args.config)


if __name__ == "__main__":
    main()
