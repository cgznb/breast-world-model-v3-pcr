"""Plot audited pCR results without rerunning or selecting any model."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/registered_three_phase_roi32_sequential_pcr_v1_20260918"
DEPTHS = ["T0", "T0-T1", "T0-T2", "T0-T3"]
ARMS = {
    "real": ("Real images", "#222222"),
    "copy_T0": ("Copy T0", "#888888"),
    "direct_mc4": ("Direct from T0", "#0072B2"),
    "rollout_mc4": ("Generated-state rollout", "#D55E00"),
    "previous_real_mc4": ("Real previous visit", "#009E73"),
}
CONTRASTS = {
    "rollout_mc4_minus_direct_mc4": "Rollout - direct T0",
    "rollout_mc4_minus_real": "Rollout - real",
    "rollout_mc4_minus_copy_T0": "Rollout - copy T0",
    "previous_real_mc4_minus_rollout_mc4": "Real previous - rollout",
    "previous_real_mc4_minus_direct_mc4": "Real previous - direct T0",
    "previous_real_mc4_minus_real": "Real previous - real",
    "previous_real_mc4_minus_copy_T0": "Real previous - copy T0",
}


def main():
    directory = OUTPUT / "evaluation"
    assert json.loads((OUTPUT / "COMPLETE.json").read_text())["passed"]
    assert json.loads((directory / "independent_audit.json").read_text())["passed"]
    summary = pd.read_csv(directory / "summary.csv")
    summary = summary[summary.threshold_policy == "fixed_0.5"].set_index(["source", "depth"])
    differences = pd.read_csv(directory / "paired_auroc_differences.csv").set_index(["depth", "comparison"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.55, wspace=.75)
    overview = fig.add_subplot(grid[0, :])
    means = []
    for arm, (label, color) in ARMS.items():
        values = [summary.loc[(arm, depth), "auroc_mean"] for depth in DEPTHS]
        means.extend(values)
        overview.plot(range(4), values, marker="o", markersize=5, color=color, label=label,
                      linewidth=2, linestyle="--" if arm == "copy_T0" else "-")
    overview.set(xticks=range(4), xticklabels=DEPTHS, ylabel="Mean AUROC across 10 classifier seeds",
                 ylim=(max(.45, min(means) - .035), min(1., max(means) + .035)),
                 title="pCR prediction with real or generated three-phase MRI")
    overview.grid(axis="y", alpha=.2)
    overview.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=5, frameon=False, fontsize=9)
    extrema = differences[["ci95_low", "ci95_high"]].to_numpy()
    low, high = min(-.02, float(extrema.min()) - .02), max(.02, float(extrema.max()) + .02)
    for column, depth in enumerate(DEPTHS[2:]):
        axes = fig.add_subplot(grid[1, column])
        for index, (contrast, label) in enumerate(CONTRASTS.items()):
            row = differences.loc[(depth, contrast)]
            left, right = contrast.split("_minus_")
            assert np.isclose(row.auroc_difference,
                              summary.loc[(left, depth), "auroc_mean"] - summary.loc[(right, depth), "auroc_mean"],
                              atol=1e-12, rtol=0)
            color = ARMS[left][1]
            axes.plot([row.ci95_low, row.ci95_high], [index, index], color=color, linewidth=2)
            axes.plot(row.auroc_difference, index, "o", color=color, markersize=5)
        axes.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axes.set(yticks=range(len(CONTRASTS)), yticklabels=list(CONTRASTS.values()),
                 ylim=(len(CONTRASTS) - .5, -.5), xlim=(low, high),
                 xlabel="AUROC difference (paired patient 95% CI)", title=depth)
        axes.grid(axis="x", alpha=.15)
        axes.tick_params(axis="y", length=0, labelsize=9)
    fig.text(.5, .045, "Same 102 patients (32 pCR positive), 200 frozen classifiers; no retraining. Generated arms average 4 sequence probabilities.",
             ha="center", fontsize=9)
    fig.text(.5, .025, "2,000 paired stratified patient bootstraps, conditional on fitted models and draws. Real-previous uses observed follow-up information.",
             ha="center", fontsize=9)
    fig.text(.5, .005, "Retrospective known timing/availability; this historical holdout was used for generator validation.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.18, right=.98, top=.93, bottom=.15)
    for suffix in ("png", "pdf"):
        path = directory / ("auroc_comparison." + suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
