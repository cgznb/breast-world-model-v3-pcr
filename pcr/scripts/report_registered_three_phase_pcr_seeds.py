"""Export and plot separate classifier seeds from completed pCR predictions."""

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from plot_registered_three_phase_sequential_pcr import ARMS, DEPTHS, OUTPUT


def load_verified_metrics():
    directory = OUTPUT / "evaluation"
    assert json.loads((OUTPUT / "COMPLETE.json").read_text())["passed"]
    assert json.loads((directory / "independent_audit.json").read_text())["passed"]
    metrics = pd.read_csv(directory / "seed_metrics.csv")
    metrics = metrics[(metrics.threshold_policy == "fixed_0.5") & metrics.source.isin(ARMS)]
    keys = ["source", "depth", "seed"]
    expected = pd.MultiIndex.from_product([list(ARMS), DEPTHS, range(42, 52)], names=keys)
    assert len(metrics) == 200 and not metrics.duplicated(keys).any()
    assert metrics.set_index(keys).index.sort_values().equals(expected.sort_values())
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"probability": np.float32})
    predictions = predictions[predictions.source.isin(ARMS)].rename(columns={"temporal_depth": "depth"})
    assert len(predictions) == 20400 and not predictions.duplicated(keys + ["patient_id"]).any()
    heldout = set(predictions.patient_id)
    saved = metrics.set_index(keys).auroc
    errors = []
    for key, frame in predictions.groupby(keys):
        assert len(frame) == 102 and set(frame.patient_id) == heldout and frame.label.sum() == 32
        errors.append(abs(roc_auc_score(frame.label, frame.probability) - saved.loc[key]))
    summary = pd.read_csv(directory / "summary.csv")
    summary = summary[(summary.threshold_policy == "fixed_0.5") & summary.source.isin(ARMS)]
    summary = summary.set_index(["source", "depth"])
    aggregates = metrics.groupby(["source", "depth"]).auroc.agg(["mean", "std"])
    summary_error = max(float((aggregates["mean"] - summary.auroc_mean).abs().max()),
                        float((aggregates["std"] - summary.auroc_std).abs().max()))
    assert max(errors) < 1e-12 and summary_error < 1e-12
    report = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "classifier_seeds": list(range(42, 52)), "metric_groups": len(metrics),
              "patients_per_group": 102, "positives_per_group": 32,
              "patient_prediction_rows_checked": len(predictions),
              "auc_replay_max_error": max(errors), "summary_replay_max_error": summary_error,
              "auc_comparison_tolerance": 1e-12,
              "within_seed_aggregation": "five_fold_mean_probability; generated_arms_also_average_four_draw_probabilities",
              "cross_seed_aggregation_in_comparison_table": False, "optimizer_updates": 0}
    return metrics, report


def main():
    metrics, verification = load_verified_metrics()
    directory = OUTPUT / "evaluation"
    index = pd.MultiIndex.from_product([DEPTHS, range(42, 52)], names=["depth", "seed"])
    table = metrics.pivot(index=["depth", "seed"], columns="source", values="auroc").reindex(index=index, columns=list(ARMS))
    table.to_csv(directory / "seed_comparison.csv")
    lines = ["# pCR Results By Classifier Seed", "",
             "Seeds 42-51 are pCR classifier training seeds, not the generator's noise draws.",
             "Every cell is one seed's AUROC on the same 102 patients (32 positives). No averaging across classifier seeds occurs in these tables.",
             "Within each seed, average the five fold probabilities before calculating AUROC. Generated arms also average four complete-sequence probabilities (MC4).",
             "Earlier headline results were means of these ten AUROCs, not AUROC after pooling all ten seeds' probabilities.",
             "Real-previous generation uses true preceding follow-up images; rollout uses its own previous generated latent.",
             "These patients were used for generator validation; the ten seeds are not independent patient cohorts.", ""]
    for depth in DEPTHS:
        lines += ["## " + depth, "", "| Seed | Real | Copy T0 | Direct T0 | Rollout | Real previous |",
                  "|---|---:|---:|---:|---:|---:|"]
        for seed, row in table.loc[depth].iterrows():
            lines.append("| " + str(seed) + " | " + " | ".join(f"{value:.5f}" for value in row) + " |")
        lines.append("")
    lines += ["## Seed Variation", "", "Descriptive ranges across the ten existing seeds, without selecting a seed for deployment.", "",
              "| Window | Arm | Minimum AUROC | Maximum AUROC | Seed SD |",
              "|---|---|---:|---:|---:|"]
    for depth in DEPTHS:
        for arm, (label, _) in ARMS.items():
            values = table.loc[depth, arm]
            lines.append(f"| {depth} | {label} | {values.min():.5f} | {values.max():.5f} | {values.std():.5f} |")
    lines += ["", "## Per-Seed Differences", "", "Counts compare paired AUROCs within each classifier seed; differences within 1e-12 are ties. No significance test across seeds is implied.", "",
              "| Window | Comparison | Higher | Equal | Lower |", "|---|---|---:|---:|---:|"]
    for depth in DEPTHS[2:]:
        for left, right in (("rollout_mc4", "direct_mc4"), ("previous_real_mc4", "rollout_mc4"),
                            ("previous_real_mc4", "real")):
            delta = table.loc[depth, left] - table.loc[depth, right]
            lines.append(f"| {depth} | {ARMS[left][0]} minus {ARMS[right][0]} | {(delta > 1e-12).sum()} | {(delta.abs() <= 1e-12).sum()} | {(delta < -1e-12).sum()} |")
    lines += ["", "[All-seed AUROC CSV](seed_comparison.csv) | [Plot PNG](seed_comparison.png) | [Plot PDF](seed_comparison.pdf)",
              "", "All 200 AUROCs were independently recomputed from saved patient predictions; verification is in seed_comparison_verification.json.", ""]
    (directory / "seed_comparison.md").write_text("\n".join(lines))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    markers = {"real": "o", "copy_T0": "x", "direct_mc4": "s", "rollout_mc4": "^", "previous_real_mc4": "D"}
    for axis, depth in zip(axes.flat, DEPTHS):
        for arm, (label, color) in ARMS.items():
            axis.plot(range(42, 52), table.loc[depth, arm], color=color, label=label,
                      marker=markers[arm], markersize=4, linewidth=1.5,
                      linestyle="--" if arm == "copy_T0" else "-")
        title = depth
        if depth == "T0":
            title += "\nAll five arms coincide"
        elif depth == "T0-T1":
            title += "\nAll three generated arms coincide"
        axis.set(title=title, xticks=list(range(42, 52)), ylabel="AUROC", xlabel="Classifier seed")
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=5, frameon=False, fontsize=9)
    fig.suptitle("pCR AUROC by classifier seed", y=.995, fontsize=15)
    fig.text(.5, .04, "Each point is one classifier seed; five-fold probability averaging and generated MC4 are retained. Panel y-axis scales differ.",
             ha="center", fontsize=9)
    fig.text(.5, .015, "Same 102 historical holdout patients (32 positives). Real-previous uses observed follow-up information. No retraining or seed selection.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91), h_pad=2.2)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / ("seed_comparison." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    (directory / "seed_comparison_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    print(directory / "seed_comparison.md")


if __name__ == "__main__":
    main()
