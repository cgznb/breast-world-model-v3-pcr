"""Export paired per-seed real-input results without fitting or inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full978_anti_overfit import _atomic_csv, _atomic_text
from src import first_post_optimization as opt
from src import registered_three_phase_optimization as study
from src.first_post_pcr_data import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/registered_three_phase_pcr_optimization_v2.yaml")
    cfg = study.load_config(parser.parse_args().config)
    output = Path(cfg["output_dir"])
    assert read_json(output / "evaluation/independent_audit.json")["passed"]
    metrics = pd.read_csv(output / "evaluation/holdout/seed_metrics.csv")
    frame = metrics[(metrics.source == "real") & metrics.model.isin(["historical_original", "baseline", "selected"])]
    wide = frame.pivot(index=["depth", "seed"], columns="model", values=["auroc", "logloss", "brier"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    assert len(wide) == 40 and not wide.isna().any().any()
    for depth in cfg["depths"]:
        assert wide[wide.depth == depth].seed.tolist() == cfg["formal_seeds"]
    for metric in ("auroc", "logloss", "brier"):
        for model in ("historical_original", "baseline", "selected"):
            for depth in cfg["depths"]:
                expected = frame[(frame.depth == depth) & (frame.model == model)].set_index("seed")[metric]
                actual = wide[wide.depth == depth].set_index("seed")[f"{metric}_{model}"]
                assert (actual - expected.reindex(actual.index)).abs().max() < 1e-12
    wide["auroc_selected_minus_original"] = wide.auroc_selected - wide.auroc_historical_original
    wide["auroc_selected_minus_nested_baseline"] = wide.auroc_selected - wide.auroc_baseline
    _atomic_csv(output / "evaluation/real_per_seed_comparison.csv", wide)
    lines = ["# Paired Classifier Seed Results", "",
             "Real-input historical102-patient holdout; each seed retains fivefold probability averaging.",
             "All seeds42-51 are reported. No seed or candidate is selected using these outcomes.", "",
             "| Window | Seed | Original AUROC | Matched nested baseline | Inner-selected AUROC | Selected - original |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in wide.itertuples():
        lines.append(f"| {opt.DEPTH_NAMES[r.depth]} | {r.seed} | {r.auroc_historical_original:.5f} | {r.auroc_baseline:.5f} | {r.auroc_selected:.5f} | {r.auroc_selected_minus_original:+.5f} |")
    _atomic_text(output / "evaluation/real_per_seed_comparison.md", "\n".join(lines) + "\n")
    print("Verified and exported40paired window/seed rows.")


if __name__ == "__main__":
    main()
