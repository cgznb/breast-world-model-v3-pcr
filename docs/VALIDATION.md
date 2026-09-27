# Release Validation

Date: 2026-09-27. Local Python 3.12.13, PyTorch 2.12.1+cu130. Tests and generator
smoke ran on CPU with CUDA hidden; no formal study was retrained.

| Check | Result |
| --- | --- |
| Generator tests | 81 passed across initial run and focused rerun |
| `world.py smoke` | Exit 0; synthetic A/B training, observation extraction and generation after deleting the synthetic target |
| PCR release adapter tests in this repository | 2 passed |
| Identical shared PCR budget/input/generated-PCR source in the V2 repository | 20 passed, 1 skipped |

The initial generator run had 80 passes and one environment error because a
subprocess invoked `python` outside an activated environment. Repeating that
test with the interpreter directory on PATH passed. No assertion was removed.

The skipped shared-PCR integration test needs private historical folds and
baseline recipe artifacts. It is not a passed real-cohort replay. Other legacy
PCR tests were not run as a full suite. Actual Pillar weight loading, full-size
patient MRI generation, CUDA training and trained clinical prediction were not
repeated. Synthetic smoke does not establish clinical performance.
