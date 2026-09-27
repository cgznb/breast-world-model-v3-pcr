# Reduced registered three-phase run

Local target: `/path/to/research/MAM/symm-fm-threephase-observation-v3`.
This is a separate extraction of the supplied v3 ZIP. Original training code,
patient arrays, VQ weights and previous experiment outputs are preserved.

## Data

- Source: `../MeWM-ISPY2/runs/registered_three_phase_roi32_v1`.
- MRI order: pre, first post, metadata-selected late; shape `[3,32,128,128]`.
- Existing raw continuous VQ arrays: `[24,8,32,32]`, float16 on disk.
- Original partitions: 764/102 train/validation patients; 2657/382 visits;
  3462/538 forward longitudinal pairs. The validation cohort is a development
  cohort already used in previous generator experiments, not an independent test.
- A uses `data/registered_roi32/visits.json`, independently sampled by patient
  then visit. B uses the separate `pairs.json`, sampled by patient then pair.
- Normalization is fitted on unique training visits only and matches the
  existing cache. B fits its clinical vocabulary on training pairs only.
- Registration metadata and positive clinical intervals were replayed. Complete
  ROI support is assumed because no measured coverage masks were imported.
- The matching, weights-only-readable VQ is reused from the previous local
  export. Its v3 decoding was compared with the previous decoder, with zero
  maximum absolute difference on a real ROI32 example.

## Adaptation

The original default latent crop `[16,32,32]` exceeds the reduced latent depth.
A uses two different same-visit `[6,24,24]` crops and `[2,4,4]` patches, giving
108 tokens per crop. B uses the full `[24,8,32,32]` latent, giving 256 observation
tokens before its state adapter. Relative crop positions use the same array's
ZYX lattice; physical geometry is preserved as metadata without guessing an
affine convention.

Production architecture is retained: 128-wide phase mixer, 384-wide 12-layer
spatial encoder, 6-layer predictor, 4-layer readout; MONAI velocity widths
`[64,128,256,256]`. BF16 and actual large physical batches are used. Activation
checkpointing is disabled in this recipe. Independent phase-attention rows
are chunked at 8192 to bound CUDA launch dimensions at large batches.

A uses local JEPA + global same-visit prediction + 0.25 reconstruction. It has
no treatment encoder, future predictor or pCR supervision. B receives the
frozen EMA target encoder from A's selected checkpoint and trains its clinical
encoder, state adapter and velocity network using the supplied velocity loss.
The default conditional direction remains forward; reverse probability and
semantic endpoint loss remain zero. Unsupported measured readouts remain off.

Reference budgets from `cache_only.yaml` are preserved as sample counts:
160000 A visits and 800000 B pairs, reference effective batch 8. Actual optimizer
steps, warmup, checkpoint frequency and validation frequency are converted to
the selected physical batches. EMA decay is converted per consumed sample.
This is not an identical-gradient reproduction of batch 8. Learning rates
remain 0.0002 for A and 0.0001 for B. Each validation selects one fixed visit or
pair for each of the 102 patients, with B using MC4 Euler20 as supplied.

## Launch and monitor

```bash
python -B scripts/launch_registered_roi32.py \
  --config configs/registered_roi32_5090.yaml \
  --data data/registered_roi32 \
  --output runs/registered_roi32_20260920 \
  --wait-for-pid 71766 --detach
```

The existing five-seed controller is identified by PID and process start time.
The pipeline waits for both that controller to exit and the GPU to become idle.
It does not interrupt the existing experiment.

After the GPU is free, each batch candidate executes six real optimizer updates
in a separate process, including optimizer states and EMA copies. Selection
requires allocator peak reservation <=93% of GPU memory and throughput >=95%
of the best safe candidate, then prefers the largest measured allocation.
CUDA context overhead remains outside the allocator report. This maximizes
useful memory occupancy with measured headroom; allocating unused tensors is
not part of the method. Candidate OOM is contained in its profiling process.

The pipeline then runs three full-architecture real-data updates in A and B,
validation, checkpoint readback and frozen A-to-B tensor comparisons. Formal
training starts only after those checks pass, and automatically progresses A
then B. A failed check stops the pipeline with a recorded reason.

- `pipeline_status.json`: waiting, profiling, smoke, training, complete or failed.
- `pipeline.log`: pipeline errors and startup messages.
- `preflight/selection.json`: measured batches, throughput and memory.
- `selected_config.json`: actual formal configuration.
- `controller.log`, `A/metrics.jsonl`, `B/metrics.jsonl`: training metrics.
- `A/best.pt`, `B/best.pt`: selected checkpoints; `last.pt` includes recovery state.
- `A/progress.json`, `B/progress.json`: optimizer progress and memory peaks.

Use the same launcher command with `--resume --detach` after a stopped run.
An output-directory lock rejects duplicate pipelines and a separate lock guards
the trainer. Signals to the pipeline are forwarded to its current child; a
training child saves its last complete optimizer state before exiting.
Runtime source files, resolved config, manifests and source assets are bound
using paths, byte sizes and modification times. No checksums are recorded.
Keep bound files unchanged for resume; new recipes use a separate output path.

## Verification

`scripts/verify_registered_roi32.py` audits actual shapes, partitions, same-visit
crop positions, absent A labels, training statistics and VQ decoding. Focused
tests cover sample budgets, EMA scaling, patient-balanced validation, preloaded
data equivalence, exact resume, frozen teacher handoff, changed-asset rejection
and batch selection. GPU results are reported separately in each run's preflight
directory; CPU tests alone do not establish a successful GPU launch.
