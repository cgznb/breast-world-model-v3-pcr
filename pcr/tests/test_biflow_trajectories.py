from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import generate_biflow_full978_trajectories as trajectory_runner
from src.biflow_trajectories import (
    DIRECT_T0,
    ROLLOUT_GENERATED_DCE0_REAL_SER,
    TrajectoryRoute,
    batched,
    build_trajectory_routes,
    compose_rollout_source_mri,
    index_registered_source_assets,
    parse_registered_timepoints,
    shared_target_seed_schedule,
    validate_locked102_routes,
    previous_real_routes,
)


def _row(timepoints: str, days: tuple[float, float, float, float]):
    return {
        "registered_timepoints": timepoints,
        **{f"days_T{stage}": value for stage, value in enumerate(days)},
    }


def test_routes_distinguish_direct_t0_from_gap_aware_rollout() -> None:
    patient_ids = ["ISPY2-100001", "ISPY2-100002", "ISPY2-100003"]
    rows = {
        "100001": _row("T0;T1;T2;T3", (0, 30, 80, 140)),
        "100002": _row("T0;T2;T3", (0, np.nan, 90, 150)),
        "100003": _row("T0;T1;T3", (0, 35, np.nan, 145)),
    }
    conditions = {key: ("clinical", "treatment") for key in rows}

    routes = build_trajectory_routes(patient_ids, rows, conditions)

    direct = {
        (route.patient_id, route.target_stage): (route.source_stage, route.delta_days)
        for route in routes[DIRECT_T0]
    }
    rollout = {
        (route.patient_id, route.target_stage): (route.source_stage, route.delta_days)
        for route in routes[ROLLOUT_GENERATED_DCE0_REAL_SER]
    }
    assert direct[("ISPY2-100001", 3)] == (0, 140)
    assert rollout[("ISPY2-100001", 3)] == (2, 60)
    assert rollout[("ISPY2-100002", 2)] == (0, 90)
    assert rollout[("ISPY2-100003", 3)] == (1, 110)
    assert all(route.source_dce0 == "real" for route in routes[DIRECT_T0])
    assert all(route.source_ser == "real" for values in routes.values() for route in values)


def test_target_seed_is_shared_between_strategies() -> None:
    rows = {"100001": _row("T0;T1;T2", (0, 30, 80, np.nan))}
    routes = build_trajectory_routes(
        ["ISPY2-100001"], rows, {"100001": ("clinical", "treatment")}
    )

    seeds = shared_target_seed_schedule(routes, base_seed=22026)

    assert seeds == {("ISPY2-100001", 1): 22026, ("ISPY2-100001", 2): 22027}
    assert {
        seeds[route.target_key]
        for route in routes[DIRECT_T0]
        if route.target_stage == 1
    } == {
        seeds[route.target_key]
        for route in routes[ROLLOUT_GENERATED_DCE0_REAL_SER]
        if route.target_stage == 1
    }


def test_previous_real_preserves_gaps_and_changes_only_source_policy():
    rows = {"100001": _row("T0;T1;T3", (0, 30, np.nan, 140))}
    routes = build_trajectory_routes(
        ["ISPY2-100001"], rows, {"100001": ("clinical", "treatment")}
    )
    previous = previous_real_routes(routes[ROLLOUT_GENERATED_DCE0_REAL_SER])
    assert [(r.source_stage, r.target_stage, r.delta_days) for r in previous] == [
        (0, 1, 30), (1, 3, 110)
    ]
    assert all(r.strategy == "adjacent_real" and r.source_dce0 == "real" for r in previous)
    assert previous[-1].target_key == routes[ROLLOUT_GENERATED_DCE0_REAL_SER][-1].target_key
    with pytest.raises(ValueError, match="modality policy"):
        replace(previous[-1], source_dce0="generated")
    with pytest.raises(ValueError, match="gap-aware"):
        previous_real_routes(routes[DIRECT_T0])


def test_rollout_feedback_replaces_only_dce0_and_masks_outside_fov() -> None:
    real = torch.zeros((2, 96, 256, 256), dtype=torch.float16)
    real[0].fill_(3)
    real[1].fill_(7)
    generated = torch.full((1, 96, 256, 256), 5, dtype=torch.float16)
    valid = torch.zeros((96, 256, 256), dtype=torch.bool)
    valid[:, 10:20, 30:40] = True

    output = compose_rollout_source_mri(generated, real, valid)

    assert output.dtype == torch.float16
    assert torch.equal(output[1], real[1])
    assert torch.all(output[0, valid] == 5)
    assert torch.all(output[0, ~valid] == 0)


def test_registered_asset_index_normalizes_prefixes_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    dce0 = tmp_path / "case_T0_dce_aqc_0.nii.gz"
    ser = tmp_path / "case_T0_ser.nii.gz"
    meta = tmp_path / "case_T0_meta.json"
    for path in (dce0, ser, meta):
        path.touch()
    selected = tmp_path / "selected.csv"
    source = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "ISPY2-123456",
                "visit": "T0",
                "dce_paths": str(dce0),
                "meta_path": str(meta),
            }
        ]
    ).to_csv(selected, index=False)
    pd.DataFrame(
        [
            {
                "patient_id": "ACRIN-6698-123456",
                "visit": "T0",
                "dce_paths": str(dce0),
                "ser_path": str(ser),
                "meta_path": str(meta),
            }
        ]
    ).to_csv(source, index=False)

    indexed = index_registered_source_assets(
        selected, [source], patient_ids=["ISPY2-123456"]
    )

    assert indexed[("123456", 0)].patient_id == "ISPY2-123456"
    assert indexed[("123456", 0)].ser_path == ser.resolve()

    conflicting_ser = tmp_path / "conflicting_ser.nii.gz"
    conflicting_ser.touch()
    conflict = tmp_path / "conflict.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "ISPY2-123456",
                "visit": "T0",
                "dce_paths": str(dce0),
                "ser_path": str(conflicting_ser),
                "meta_path": str(meta),
            }
        ]
    ).to_csv(conflict, index=False)
    with pytest.raises(ValueError, match="conflict"):
        index_registered_source_assets(
            selected, [source, conflict], patient_ids=["ISPY2-123456"]
        )


def test_registered_timepoints_accept_ordered_gaps() -> None:
    assert parse_registered_timepoints("T0;T2;T3") == (0, 2, 3)


def test_rollout_feedback_rejects_wrong_dtypes() -> None:
    generated = torch.zeros((1, 96, 256, 256), dtype=torch.float16)
    real = torch.zeros((2, 96, 256, 256), dtype=torch.float16)
    valid = torch.ones((96, 256, 256), dtype=torch.bool)

    with pytest.raises(ValueError, match="tensor contract"):
        compose_rollout_source_mri(generated.to(torch.int16), real, valid)
    with pytest.raises(ValueError, match="tensor contract"):
        compose_rollout_source_mri(generated, real.float(), valid)
    with pytest.raises(ValueError, match="tensor contract"):
        compose_rollout_source_mri(generated, real, valid.to(torch.uint8))


def test_locked_route_validation_rejects_extra_strategy_and_patient_drift() -> None:
    patterns = (
        ["T0;T1;T2;T3"] * 94
        + ["T0;T1;T2"] * 3
        + ["T0;T1;T3"] * 2
        + ["T0;T2;T3"]
        + ["T0;T1"] * 2
    )
    patient_ids = [f"ISPY2-{index:06d}" for index in range(100000, 100102)]
    rows = {
        str(index): _row(pattern, (0, 30, 80, 140))
        for index, pattern in zip(range(100000, 100102), patterns, strict=True)
    }
    conditions = {
        str(index): ("clinical", "treatment") for index in range(100000, 100102)
    }
    routes = build_trajectory_routes(patient_ids, rows, conditions)

    validate_locked102_routes(patient_ids, routes)
    with pytest.raises(ValueError, match="strategies changed"):
        validate_locked102_routes(patient_ids, {**routes, "unexpected": ()})

    changed_direct = list(routes[DIRECT_T0])
    changed_direct[0] = replace(
        changed_direct[0],
        patient_id="ISPY2-999999",
        source_visit_id="ISPY2-999999:T0",
        target_visit_id=f"ISPY2-999999:T{changed_direct[0].target_stage}",
    )
    with pytest.raises(ValueError, match="inventory changed"):
        validate_locked102_routes(
            patient_ids,
            {**routes, DIRECT_T0: tuple(changed_direct)},
        )


@pytest.mark.parametrize("value", ["T1;T2", "T0;T2;T1", "T0;T0", "T0;T4", ""])
def test_registered_timepoints_reject_invalid_sequences(value: str) -> None:
    with pytest.raises(ValueError):
        parse_registered_timepoints(value)


def test_batched_preserves_order_and_validates_size() -> None:
    assert list(batched([0, 1, 2, 3, 4], 2)) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        list(batched([1], 0))


def test_latent_validator_rejects_extra_fields_and_unknown_provenance(
    tmp_path: Path,
) -> None:
    route = TrajectoryRoute(
        strategy=DIRECT_T0,
        patient_id="ISPY2-123456",
        source_stage=0,
        target_stage=1,
        source_visit_id="ISPY2-123456:T0",
        target_visit_id="ISPY2-123456:T1",
        delta_days=30,
        clinical_text="clinical",
        treatment_text="treatment",
        source_dce0="real",
        source_ser="real",
    )
    path = tmp_path / "latent.pt"
    payload = {
        "schema": trajectory_runner.LATENT_SCHEMA,
        **trajectory_runner._artifact_identity(route, 22026, 20),
        "source_input_provenance": "locked_strict_a_cache",
        "target_latent_read": False,
        "predicted_normalized_latent": torch.zeros(
            trajectory_runner.LATENT_SHAPE, dtype=torch.float16
        ),
    }

    torch.save(payload, path)
    assert trajectory_runner._valid_latent(path, route, 22026, 20)

    torch.save({**payload, "unexpected": True}, path)
    assert not trajectory_runner._valid_latent(path, route, 22026, 20)

    payload["source_input_provenance"] = "unknown"
    torch.save(payload, path)
    assert not trajectory_runner._valid_latent(path, route, 22026, 20)


def test_main_preserves_explicit_zero_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "generation": {
            "output_dir": str(tmp_path / "output"),
            "solver_steps": 20,
            "base_seed": 22026,
            "batch_size": 2,
        }
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(trajectory_runner, "_load_config", lambda _: config)

    def capture_generate(_config, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(trajectory_runner, "generate", capture_generate)

    assert (
        trajectory_runner.main(
            [
                "--config",
                "unused.yaml",
                "--device",
                "cpu",
                "--solver-steps",
                "0",
                "--batch-size",
                "0",
            ]
        )
        == 0
    )
    assert captured["solver_steps"] == 0
    assert captured["batch_size"] == 0
