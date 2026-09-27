import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import registered_three_phase_generated_pcr as study


def test_source_guard_blocks_future_data_and_resets_after_failure(tmp_path):
    baseline, future = tmp_path / "T0.npy", tmp_path / "T1.npy"
    baseline.write_bytes(b"baseline")
    future.write_bytes(b"future")
    with pytest.raises(RuntimeError, match="T0-only"):
        with study.source_read_guard([baseline], [], [tmp_path]) as audit:
            assert baseline.read_bytes() == b"baseline"
            future.read_bytes()
    assert audit["opened"] == {str(baseline)}
    assert future.read_bytes() == b"future"
    with study.source_read_guard([], [future], []):
        with pytest.raises(RuntimeError, match="T0-only"):
            future.read_bytes()


def route_inputs():
    visits = [("a", 0), ("a", 1), ("a", 3), ("b", 0), ("c", 0), ("c", 1), ("c", 2)]
    cohort = {"split": {"val": ["a", "b", "c"]},
              "visits": [{"patient_id": p, "timepoint": t} for p, t in visits]}
    records = []
    for p, t in visits:
        if t:
            records.append({"patient_id": p, "pair_id": f"{p}:T0-T{t}", "split": "val",
                            "earlier_stage": "T0", "later_stage": f"T{t}",
                            "earlier_visit_id": f"{p}:T0", "later_visit_id": f"{p}:T{t}",
                            "delta_days": t * 30, "interval_missing": False, "interval_source": "test",
                            "baseline_clinical": {"age": 50}, "treatment": {"treatment_arm": "test"},
                            "pCR": 1, "target_mask": "must_not_be_copied"})
    return cohort, {"pairs": records}


def test_routes_keep_patients_with_missing_visits_and_remove_target_metadata():
    cohort, manifest = route_inputs()
    routes, counts = study.routes_for(cohort, manifest, 100)
    assert counts == {"T0": 3, "T0-T1": 2, "T0-T2": 1, "T0-T3": 0}
    assert [(r["patient_id"], r["timepoint"]) for r in routes] == [("a", 1), ("c", 1), ("c", 2)]
    assert [r["noise_seed"] for r in routes] == [100, 102, 103]
    assert all("pCR" not in r["record"] and "target_mask" not in r["record"] for r in routes)
    manifest["pairs"].pop(0)
    with pytest.raises(ValueError, match="direct-T0"):
        study.routes_for(cohort, manifest, 100)


def test_replacements_cover_future_slots_without_changing_observed_inputs():
    real = {"pids": ["a", "b"], "embs": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
            "masks": np.array([[1, 1, 1, 0], [1, 0, 0, 0]], bool),
            "days": np.array([[0, 30, 60, 0], [0, 0, 0, 0]]),
            "clinical": np.ones((2, 5)), "labels": np.array([0, 1])}
    untouched = copy.deepcopy(real)
    routes = [{"patient_id": "a", "timepoint": t} for t in (1, 2)]
    copied = study.replaced_split(real, routes, "copy_T0")
    np.testing.assert_array_equal(copied["embs"][0, 1:3], np.tile(real["embs"][0, 0], (2, 1)))

    class Store:
        def load(self, pid, tp, require_finite):
            assert pid == "a" and require_finite
            return np.full(3, -tp, dtype=np.float32)

    generated = study.replaced_split(real, routes, "draw_0", Store())
    np.testing.assert_array_equal(generated["embs"][0, 1:3], [[-1] * 3, [-2] * 3])
    for result in (copied, generated):
        for field in ("masks", "days", "clinical", "labels"):
            np.testing.assert_array_equal(result[field], real[field])
        np.testing.assert_array_equal(result["embs"][:, 0], real["embs"][:, 0])
        np.testing.assert_array_equal(result["embs"][~real["masks"]], real["embs"][~real["masks"]])
    for field in real:
        np.testing.assert_array_equal(real[field], untouched[field])
    for invalid in (routes[:1], routes + [routes[0]]):
        with pytest.raises(ValueError, match="exactly once"):
            study.replaced_split(real, invalid, "copy_T0")


def test_auc_kernel_matches_weighted_sklearn_bootstraps_with_ties():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([[.1, .5, .5, .9, .6, .8], [.2, .2, .9, .1, .3, .8]])
    kernel = study.mean_auc_kernel(y, p)
    assert kernel.mean() == pytest.approx(np.mean([roc_auc_score(y, row) for row in p]))
    for weights in (np.array([1, 1, 1, 1, 1, 1]), np.array([0, 2, 2, 1, 1, 0])):
        value = weights[y == 1] @ kernel @ weights[y == 0] / (weights[y == 1].sum() * weights[y == 0].sum())
        expected = np.mean([roc_auc_score(y, row, sample_weight=weights) for row in p])
        assert value == pytest.approx(expected)


def test_paired_bootstrap_preserves_identical_arms_and_known_difference(tmp_path):
    rows = [{"source": arm, "temporal_depth": "T0-T1", "seed": seed, "patient_id": str(i),
             "label": label, "probability": prob}
            for arm in ("real", "copy_T0", "draw_0", "generated_mc4") for seed in (42, 43)
            for i, (label, prob) in enumerate(zip([0, 1, 0, 1], [.1, .6, .4, .9]))]
    frame = pd.DataFrame(rows)
    frame.loc[frame.source == "real", "probability"] = 1 - frame.loc[frame.source == "real", "probability"]
    cfg = {"output_dir": str(tmp_path), "bootstrap_seed": 7, "bootstrap_samples": 100}
    differences = study.paired_bootstrap(cfg, frame).set_index("comparison")
    for arm in ("generated_mc4", "draw_0"):
        for column in ("auroc_difference", "ci95_low", "ci95_high"):
            assert differences.loc[f"{arm}_minus_copy_T0", column] == 0
            assert differences.loc[f"{arm}_minus_real", column] == 1
