> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# Physical Batch Eight and Independent Seeds

The new experiments use physical batch size 8 and gradient accumulation 1.
One optimizer update therefore uses eight sampled patients per objective
branch. The earlier experiment uses batch size 1 and accumulation 4 and keeps
running from its unchanged remote source and configuration.

## Batching

The trainer passes the entire compatible batch to each objective. Landmark
batches share the current stage, complete observed-stage history, future
availability mask, and label availability mask. Groups are selected by their
probability mass under the original patient-uniform, then landmark-uniform
sampling distribution. Samples within a group are drawn with replacement
using their conditional probabilities. Small groups can contain repeated
patients; they are not duplicated permanently in the dataset.

Paired-flow batches share source and target stages and observed-stage history.
The original edge probabilities and importance weights are retained. Grouping
by complete history matters because missing visits change the elapsed-stage
input to the observation update.

Representation variance/covariance regularization remains per patient. The
batched objectives reject incompatible histories and availability masks.
Training logs record the actual minimum and maximum physical batch sizes, in
addition to configured batch size and accumulation.

## New Experiment Plan

- Seeds: 20261001, 20261002, 20261003.
- Each seed independently trains A, B, C, and D from fresh model initialization.
- The original frozen VQ codec and patient-disjoint 764/102 split are shared.
- Batch size: 8; accumulation: 1; learning rates and stage budgets unchanged.
- Maximum concurrently running new seeds: 2; remaining seeds wait in the queue.
- Remote experiment root: `/root/autodl-tmp/breast_multistage_bs8_20260930`.
- Remote frozen source snapshot: `<experiment-root>/source`.
- Each seed has `<experiment-root>/seed_<seed>/config.yaml` and `run/`.
- `queue_status.json` records queued, running, completed, or failed jobs.
- A failed seed does not prevent the queue from starting the remaining seeds.
- Re-running the same launch command resumes incomplete seeds from their own
  checkpoints and rejects conflicting plans or still-running processes.

Because effective batch size changes from 4 to 8, the earlier experiment is a
separate configuration. Compare the three new seeds with each other when
reporting seed variability; do not pool the old result into that estimate.

## Verification

The full suite passed 121 tests, with two optional MONAI tests skipped. New
checks cover batch shapes, sampling marginals, missing visits, loss and
gradient equivalence, four-stage execution, and exact checkpoint resume.

The remote RTX 5090 completed full-shape forward, backward, and optimizer
updates in all four stages with physical batch 8. B and D used three rollout
intervals, two trajectories per patient, and 20 Heun steps per interval.
Peak allocated memory was about 7.16 GiB; peak reserved memory was about
8.96 GiB. These are isolated-process measurements while the earlier run also
remained active, not a guarantee of constant memory use throughout training.

Reports are stored in the experiment root (`preflight.json`) and the local
artifact directory `/data1/cxx/breast_multistage_bs8_20260930`.
