import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src import registered_three_phase_sequential_pcr as study


def routes_and_manifest():
    direct, pairs = [], []
    for tp, elapsed in enumerate((20, 70, 150), 1):
        record = {"patient_id": "a", "pair_id": f"a:T0-T{tp}", "split": "val",
                  "earlier_stage": "T0", "later_stage": f"T{tp}", "earlier_visit_id": "a:T0",
                  "later_visit_id": f"a:T{tp}", "delta_days": elapsed, "interval_missing": False,
                  "interval_source": "test", "baseline_clinical": {"age": 50}, "treatment": {"arm": "a"}}
        direct.append({"patient_id": "a", "timepoint": tp, "key": f"a_T{tp}", "noise_seed": 1700 + tp, "record": record})
        adjacent = {**record, "pair_id": f"a:T{tp - 1}-T{tp}", "earlier_stage": f"T{tp - 1}",
                    "earlier_visit_id": f"a:T{tp - 1}", "delta_days": (20, 50, 80)[tp - 1],
                    "pCR": 1, "target_mask": "not_permitted"}
        pairs.append(adjacent)
    return direct, {"pairs": pairs}


def test_adjacent_conditions_change_stages_and_intervals_without_copying_targets():
    direct, manifest = routes_and_manifest()
    routes = study.adjacent_routes(direct, manifest)
    assert [r["record"]["earlier_stage"] for r in routes] == ["T0", "T1", "T2"]
    assert [r["record"]["delta_days"] for r in routes] == [20, 50, 80]
    assert [r["elapsed_days_from_T0"] for r in routes] == [20, 70, 150]
    assert [r["noise_seed"] for r in routes] == [r["noise_seed"] for r in direct]
    assert all("pCR" not in r["record"] and "target_mask" not in r["record"] for r in routes)
    assert study.canonical_mode("rollout", routes[0]) == study.canonical_mode("previous_real", routes[0]) == "shared_T1"


def test_adjacent_routes_reject_missing_predecessors_or_changed_intervals():
    routes, manifest = routes_and_manifest()
    with pytest.raises(ValueError, match="adjacent prefix"):
        study.adjacent_routes(routes[1:], manifest)
    manifest["pairs"][1]["delta_days"] = 51
    with pytest.raises(ValueError, match="conditions disagree"):
        study.adjacent_routes(routes, manifest)


def test_rollout_carries_each_generated_draw_and_never_reads_real_followup():
    routes = study.adjacent_routes(*routes_and_manifest())
    t0 = torch.zeros(1, 2, 1, 1, 1)
    generated = torch.arange(8).reshape(4, 2, 1, 1, 1).float()

    def forbidden(_):
        raise AssertionError("Real follow-up must not enter rollout")

    for route in routes[1:]:
        result = study.source_batch("rollout", route, t0, generated, forbidden, 4)
        assert result is generated
        assert torch.equal(result.flatten(), torch.arange(8))
    with pytest.raises(ValueError, match="each previous generated draw"):
        study.source_batch("rollout", routes[1], t0, generated.mean(0, keepdim=True), forbidden, 4)


def test_real_previous_reads_exactly_the_previous_visit_and_both_modes_share_t0():
    routes = study.adjacent_routes(*routes_and_manifest())
    t0, called = torch.ones(1, 2, 1, 1, 1), []

    def read(visit_id):
        called.append(visit_id)
        return torch.full((2, 1, 1, 1), int(visit_id[-1]) + 5.)

    for mode in study.MODES:
        torch.testing.assert_close(study.source_batch(mode, routes[0], t0, None, read, 4), t0.repeat(4, 1, 1, 1, 1))
    assert called == []
    for route in routes[1:]:
        result = study.source_batch("previous_real", route, t0, torch.zeros(4, 2, 1, 1, 1), read, 4)
        assert (result == route["timepoint"] + 4.).all()
    assert called == ["a:T1", "a:T2"]


def test_noise_matches_the_direct_generator_draw_order():
    source = torch.zeros(4, 2, 1, 2, 3)
    generator = torch.Generator().manual_seed(123)
    expected = torch.cat([torch.randn(source[:1].shape, generator=generator) for _ in range(4)])
    assert torch.equal(study.sampling_noise(source, 123), expected)
    assert not torch.equal(expected[0], expected[1])


def test_real_predecessor_guard_still_blocks_target_latent(tmp_path):
    files = {tp: tmp_path / f"T{tp}.npy" for tp in range(3)}
    for tp, path in files.items():
        path.write_bytes(str(tp).encode())
    with study.direct.source_read_guard([files[0], files[1]], [], [tmp_path]) as audit:
        assert files[1].read_bytes() == b"1"
        with pytest.raises(RuntimeError):
            files[2].read_bytes()
    assert audit["opened"] == {str(files[1])}


def test_fold_then_complete_sequence_probability_ensemble():
    rows = [{"source": f"{mode}_draw_{draw}", "temporal_depth": depth, "seed": seed,
             "fold": fold, "patient_id": pid, "label": label,
             "probability": np.float32(.05 + .15 * draw + .01 * fold)}
            for mode in ("direct", *study.MODES) for draw in range(4) for depth in study.original.DEPTHS
            for seed in (42, 43) for fold in range(5) for pid, label in (("a", 0), ("b", 1))]
    frame = pd.DataFrame(rows)
    predictions = study.ensemble_predictions(frame, ["a", "b"], [42, 43])
    averaged = predictions[predictions.source.str.endswith("mc4")]
    np.testing.assert_allclose(averaged.probability, .295, atol=1e-7)
    assert len(averaged) == 3 * 4 * 2 * 2
    with pytest.raises(ValueError, match="four distinct"):
        study.ensemble_predictions(frame[frame.source != "rollout_draw_3"], ["a", "b"], [42, 43])


def test_paired_differences_keep_identical_modes_and_pair_patients(tmp_path):
    rows = [{"source": arm, "temporal_depth": "T0-T3", "seed": seed, "patient_id": str(i),
             "label": label, "probability": probability}
            for arm in ("real", "copy_T0", "direct_mc4", "rollout_mc4", "previous_real_mc4") for seed in (42, 43)
            for i, (label, probability) in enumerate(zip([0, 1, 0, 1], [.1, .6, .4, .9]))]
    frame = pd.DataFrame(rows)
    frame.loc[frame.source == "direct_mc4", "probability"] = 1 - frame.loc[frame.source == "direct_mc4", "probability"]
    cfg = {"output_dir": str(tmp_path), "bootstrap_seed": 7, "bootstrap_samples": 100}
    result = study.paired_differences(cfg, frame).set_index("comparison")
    assert len(result) == 7
    for column in ("auroc_difference", "ci95_low", "ci95_high"):
        assert result.loc["rollout_mc4_minus_direct_mc4", column] == 1
        assert result.loc["previous_real_mc4_minus_rollout_mc4", column] == 0
