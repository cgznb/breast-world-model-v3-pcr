"""Diagnose saved real-input pCR fits without loading or fitting any model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/registered_three_phase_roi32_pcr_v1_20260917"
OUTPUT = RUN / "diagnostics/overfitting"
DEPTHS = ["T0", "T0-T1", "T0-T2", "T0-T3"]


def main():
    refs = json.loads((RUN / "frozen_models.json").read_text())["models"]
    folds = {row["fold"]: row for row in json.loads((RUN / "folds.json").read_text())}
    holdout = set(json.loads((RUN / "cohort.json").read_text())["split"]["val"])
    assert len(refs) == 200 and len(holdout) == 102
    assert len({(r["depth"], r["seed"], r["fold"]) for r in refs}) == 200
    histories, rows, errors, input_stats = {d: [] for d in DEPTHS}, [], [], {}
    for ref in refs:
        checkpoint = Path(ref["path"])
        stat = checkpoint.stat()
        assert {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} == ref["identity"]
        folder = checkpoint.parent
        for name in ("best.pt", "summary.json", "TRAINING_COMPLETE.json", "history.csv", "train_predictions.csv", "val_predictions.csv"):
            path = folder / name
            stat = path.stat()
            input_stats[path] = (stat.st_size, stat.st_mtime_ns)
        saved = json.loads((folder / "summary.json").read_text())
        complete = json.loads((folder / "TRAINING_COMPLETE.json").read_text())
        config, selection = complete["effective_config"], saved["selection"]
        assert complete["complete"] and saved["checkpoint_policy"] == "best_validation"
        assert (saved["temporal_depth"], saved["seed"], saved["fold"]) == (ref["depth"], ref["seed"], ref["fold"])
        assert saved["test_embeddings_or_labels_loaded_during_training"] is False
        history = pd.read_csv(folder / "history.csv")
        assert list(history.epoch) == list(range(len(history)))
        assert len(history) == selection["epochs_trained"]
        selected = history.iloc[selection["best_epoch_zero_based"]]
        last = history.iloc[-1]
        row = {"depth": ref["depth"], "seed": ref["seed"], "fold": ref["fold"]}
        for role, name, metric_key, ids_key in (("train", "train", "train_metrics", "train_ids"),
                                              ("validation", "val", "validation_metrics", "val_ids")):
            frame = pd.read_csv(folder / (name + "_predictions.csv"),
                                dtype={"probability": np.float32, "clinical_prior_probability": np.float32})
            assert not frame.patient_id.duplicated().any()
            assert set(frame.patient_id) == set(folds[ref["fold"]][ids_key])
            assert not set(frame.patient_id) & holdout
            assert np.isfinite(frame.probability).all() and frame.probability.between(0, 1).all()
            auc = roc_auc_score(frame.label, frame.probability)
            errors.append(abs(auc - saved[metric_key]["auroc"]))
            row.update({role + "_auc": auc,
                        role + "_logloss": log_loss(frame.label, frame.probability, labels=[0, 1]),
                        role + "_clinical_prior_auc": roc_auc_score(frame.label, frame.clinical_prior_probability)})
        assert abs(row["validation_auc"] - selected.validation_auroc) < 1e-12
        row.update(auc_gap=row["train_auc"] - row["validation_auc"],
                   parameters=saved["parameters"], selected_epoch=int(selected.epoch) + 1,
                   epochs_trained=len(history), epoch_budget=config["epochs"], patience=config["patience"],
                   early_stopped=len(history) < config["epochs"],
                   first_training_loss=float(history.iloc[0].train_loss),
                   selected_training_loss=float(selected.train_loss), final_training_loss=float(last.train_loss),
                   final_validation_auc=float(last.validation_auroc),
                   validation_auc_drop=float(selected.validation_auroc - last.validation_auroc),
                   loss_down_and_auc_down=bool(last.train_loss < selected.train_loss - 1e-6 and
                                               last.validation_auroc < selected.validation_auroc - 1e-6))
        rows.append(row)
        histories[ref["depth"]].append(history)
    assert max(errors) < 1e-12
    frame = pd.DataFrame(rows)
    assert (frame.groupby("depth").size() == 50).all()
    summary = frame.groupby("depth").agg(
        fits=("seed", "size"), train_auc=("train_auc", "mean"), validation_auc=("validation_auc", "mean"),
        auc_gap=("auc_gap", "mean"), train_logloss=("train_logloss", "mean"), validation_logloss=("validation_logloss", "mean"),
        train_clinical_prior_auc=("train_clinical_prior_auc", "mean"),
        validation_clinical_prior_auc=("validation_clinical_prior_auc", "mean"),
        selected_epoch_median=("selected_epoch", "median"), selected_epoch_min=("selected_epoch", "min"),
        selected_epoch_max=("selected_epoch", "max"), epochs_median=("epochs_trained", "median"),
        early_stopped=("early_stopped", "sum"), final_validation_auc=("final_validation_auc", "mean"),
        validation_auc_drop=("validation_auc_drop", "mean"),
        selected_training_loss=("selected_training_loss", "mean"), final_training_loss=("final_training_loss", "mean"),
        loss_down_and_auc_down=("loss_down_and_auc_down", "sum"), parameters=("parameters", "first")).reindex(DEPTHS)
    common_rows = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for index, depth in enumerate(DEPTHS):
        histories_for_depth = histories[depth]
        limit = min(len(h) for h in histories_for_depth)
        for column, (key, label, color) in enumerate((("train_loss", "Training objective", "#0072B2"),
                                                     ("validation_auroc", "Validation AUROC", "#D55E00"))):
            values = np.stack([h[key].to_numpy()[:limit] for h in histories_for_depth])
            epochs = np.arange(1, limit + 1)
            low, high = np.quantile(values, [.25, .75], axis=0)
            axis = axes[index, column]
            axis.fill_between(epochs, low, high, color=color, alpha=.14)
            axis.plot(epochs, values.mean(0), color=color, linewidth=1.8)
            axis.set(title=f"{depth}: {label}", xlabel="Epoch", ylabel=label)
            axis.grid(axis="y", alpha=.2)
            for epoch, mean, q25, q75 in zip(epochs, values.mean(0), low, high):
                common_rows.append({"depth": depth, "epoch": int(epoch), "metric": key,
                                    "fits": 50, "mean": mean, "q25": q25, "q75": q75})
    fig.suptitle("Real-input pCR training: the same 50 fits in every plotted epoch", y=.997, fontsize=14)
    fig.text(.5, .036, "Lines: mean; shading: interquartile range across fits, not confidence intervals. Curves stop when the first fit stops.",
             ha="center", fontsize=9)
    fig.text(.5, .017, "Training objectives differ across window recipes and are not validation log loss. Axis scales vary; no models were retrained.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .97), h_pad=1.5)
    for path, expected in input_stats.items():
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == expected
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "fit_details.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv")
    pd.DataFrame(common_rows).to_csv(OUTPUT / "common_epoch_curves.csv", index=False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / ("training_curves." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    verification = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(), "fits": len(refs),
                    "selected_checkpoint_aucs_recomputed": len(errors), "auc_replay_max_error": max(errors),
                    "frozen_checkpoints_match_references": True, "input_artifacts_unchanged": True,
                    "fold_membership_and_holdout_disjointness_verified": True, "model_inference": False,
                    "optimizer_updates": 0, "common_epochs": {d: min(len(h) for h in histories[d]) for d in DEPTHS}}
    (OUTPUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    lines = ["# Real-Input pCR Overfitting Diagnostic", "",
             "Analysis of all 200 existing fits, with 50 seed/fold models per temporal window. No model inference or fitting.",
             "Training and validation scores below are unweighted means of the same 50 selected-checkpoint model scores.",
             "Validation is the development fold used to select checkpoints, not the 102-patient historical holdout.",
             "Those holdout results use five-fold probability ensembles and are not directly interchangeable with fold-mean validation AUC.", "",
             "| Window | Training AUROC | Validation AUROC | Gap | Selected epoch median | Stopping epoch median |",
             "|---|---:|---:|---:|---:|---:|"]
    for depth, row in summary.iterrows():
        lines.append(f"| {depth} | {row.train_auc:.5f} | {row.validation_auc:.5f} | {row.auc_gap:.5f} | {row.selected_epoch_median:g} | {row.epochs_median:g} |")
    lines += ["", "## Interpretation", "",
              "The strongest overfitting evidence is in T0-T3: the selected models retain a large training-validation AUC gap, and later optimization reduces training loss while validation ranking degrades.",
              "All 50 T0-T3 fits have lower training loss and lower validation AUC at stopping than at their selected epoch. Selected-to-final validation AUC decreases from 0.78740 to 0.74620 on average.",
              "Because checkpoint selection maximizes validation AUC, a subsequent decline alone is not independent proof of overfitting. The gap and the chronological curves provide additional evidence.",
              "Across the same 50 T0-T3 fits, mean training loss falls from 0.62626 at epoch40 to 0.54995 at epoch51 while mean validation AUC falls from 0.76205 to 0.75349. This comparison does not align curves by their selected peak or change the participating fits.",
              "Shorter windows show smaller selected-checkpoint gaps (about 0.041-0.047); those gaps alone do not prove severe memorization. There is no logged per-epoch training AUROC or validation loss, so no such curves are inferred.",
              "T0 and T0-T1 train for all 200 epochs (patience201) but retain the validation-selected weights, not the final epoch. T0-T2 stops early in 48/50 fits and T0-T3 in 50/50.",
              "T0-T3 has 88610 trainable parameters versus 40418/40418/40978 in T0/T0-T1/T0-T2. Its projection/signature widths are64/32 versus32/16, and matrix weight decay is0.0005 versus0.0007 (projection weight decay remains0.002).",
              "T0/T0-T1 bound the residual logit at1.0. T0-T2/T0-T3 leave it unbounded; all four have residual L2 weight0 and prefix dropout0, with dropout0.25 and embedding dropout0.1. These are plausible contributors, not isolated causal explanations.",
              "The clinical-prior-only train and validation AUROCs, plus unweighted selected-checkpoint log loss, are in summary.csv. Training objectives include different positive-class-weight recipes; their absolute values must not be compared across windows.",
              "Validation scores are optimistic because they select epochs. Fold repetitions and seeds share patients, and the historical holdout has already been used in generator validation and repeated analyses. These results do not establish performance on an untouched external cohort.",
              "Generated-input degradation cannot be attributed solely to this overfitting: generation and encoder/input distribution differences remain separate explanations.", "",
              "## Next Research Decision", "",
              "Preserve this reference. A separate development-only comparison could shrink the T0-T3 head, bound or regularize its residual, and tune earlier stopping using training/inner-validation data. Do not select these changes by the existing 102-patient results.",
              "Lowering patience alone shortens training after a peak; it does not automatically change the previously selected checkpoint or remove its generalization gap.", "",
              "[Training curves](training_curves.png) | [PDF](training_curves.pdf) | [Summary](summary.csv) | [All 200 fits](fit_details.csv) | [Verification](verification.json)", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    print(summary.to_string())
    print(json.dumps(verification, indent=2))
    print(OUTPUT / "README.md")


if __name__ == "__main__":
    main()
