"""Process-level GPU leases are tested without allocating CUDA tensors."""
import multiprocessing
import os
import queue

import pytest

from responsewm.io import read_json, write_json
from responsewm.runtime_v3 import process_identity, stage_gpu_lease


def _hold_lease(root, stage, acquired, release):
    os.environ["RESPONSEWM_GPU_LEASE_DIR"] = str(root)
    with stage_gpu_lease(stage, stage, "cuda"):
        acquired.put((stage, os.getpid()))
        if not release.wait(15):
            raise TimeoutError("Test failed to release the stage lease")


def _finish(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(5)


def test_two_representation_workers_share_but_flow_waits_for_both(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Queue()
    releases = [ctx.Event() for _ in range(3)]
    processes = [ctx.Process(target=_hold_lease, args=(tmp_path, stage, acquired, release))
                 for stage, release in zip(("representation", "representation", "flow"), releases)]
    try:
        for process in processes[:2]:
            process.start()
        assert [acquired.get(timeout=5)[0] for _ in range(2)] == ["representation"] * 2
        holders = read_json(tmp_path / "holders.json")
        assert len(holders) == 2 and sum(h["units"] for h in holders) == 2
        processes[2].start()
        with pytest.raises(queue.Empty):
            acquired.get(timeout=.2)
        releases[0].set()
        processes[0].join(5)
        assert processes[0].exitcode == 0
        with pytest.raises(queue.Empty):
            acquired.get(timeout=1.2)
        releases[1].set()
        assert acquired.get(timeout=5)[0] == "flow"
        holders = read_json(tmp_path / "holders.json")
        assert len(holders) == 1 and holders[0]["units"] == 2
        releases[2].set()
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        assert read_json(tmp_path / "holders.json") == []
    finally:
        for process, release in zip(processes, releases):
            if process.is_alive():
                release.set()
        _finish([p for p in processes if p.pid is not None])


def test_exclusive_stage_blocks_representation_and_crash_does_not_leak_lease(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Queue()
    releases = [ctx.Event(), ctx.Event()]
    exclusive = ctx.Process(target=_hold_lease, args=(tmp_path, "readout", acquired, releases[0]))
    waiting = ctx.Process(target=_hold_lease, args=(tmp_path, "representation", acquired, releases[1]))
    try:
        exclusive.start()
        assert acquired.get(timeout=5)[0] == "readout"
        waiting.start()
        with pytest.raises(queue.Empty):
            acquired.get(timeout=1.2)
        exclusive.kill()
        exclusive.join(5)
        assert acquired.get(timeout=5)[0] == "representation"
        releases[1].set()
        waiting.join(5)
        assert waiting.exitcode == 0
        assert read_json(tmp_path / "holders.json") == []
    finally:
        # A killed Event waiter must never be notified: Condition.notify would
        # wait for an acknowledgement from a process that no longer exists.
        for process, release in zip((exclusive, waiting), releases):
            if process.is_alive():
                release.set()
        _finish([p for p in (exclusive, waiting) if p.pid is not None])


def test_stale_process_identity_is_discarded_and_exception_releases(tmp_path, monkeypatch):
    monkeypatch.setenv("RESPONSEWM_GPU_LEASE_DIR", str(tmp_path))
    write_json(tmp_path / "holders.json", [{"pid": os.getpid(), "identity": "stale-start-time", "units": 2}])
    with pytest.raises(RuntimeError, match="training failure"):
        with stage_gpu_lease("joint", "example", "cuda"):
            holders = read_json(tmp_path / "holders.json")
            assert len(holders) == 1 and holders[0]["identity"] == process_identity(os.getpid())
            raise RuntimeError("training failure")
    assert read_json(tmp_path / "holders.json") == []


def test_cpu_jobs_do_not_take_gpu_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("RESPONSEWM_GPU_LEASE_DIR", str(tmp_path))
    with stage_gpu_lease("representation", "example", "cpu"):
        assert not (tmp_path / "holders.json").exists()
