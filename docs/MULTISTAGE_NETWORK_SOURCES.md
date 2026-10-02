# Multistage network provenance

Audit date: 2026-09-29. These are commit IDs, not Git blob IDs.

| Repository | Commit | Source | License | Local use |
| --- | --- | --- | --- | --- |
| https://github.com/Project-MONAI/MONAI | `9c6d819f97e37f36c72f3bdfad676b455bd2fa0d` (1.5.1) | `monai/networks/nets/diffusion_model_unet.py` | Apache-2.0 | Actual optional runtime dependency; `CoupledVelocityV2._monai` traverses original residual/attention/downsample modules and inserts bridges before downsampling. No MONAI weights downloaded. |
| https://github.com/facebookresearch/DiT | `ed81ce2229091fd4ecc9a223645f95cf379d582b` | `models.py`, `DiTBlock` and `FinalLayer` | CC BY-NC 4.0 | Existing adaLN adaptation extended to separate self/history/action/FFN residuals with 12 modulation vectors. PyTorch MHA replaces timm attention. No DiT weights downloaded. |
| https://github.com/facebookresearch/vjepa2 | `204698b45b3712590f06245fbfba32d3be539812` | `app/vjepa_2_1/models/predictor.py`, `app/vjepa_2_1/train.py` | MIT | Conceptual source for hierarchical target features and separately weighted visible/hidden predictions. `representation.py` is a task-specific implementation sharing the existing four-layer masked predictor; no upstream source or pretrained video weights copied. |

The original encoder and native image parameter provenance remains in
`SOURCES.md`. This extension preserves their parameter names and shapes.
`PhaseStateEncoder.forward(return_pyramid=True)` returns the existing state
plus an `EncoderPyramid` with unpooled stage features and projected canonical
tokens. All online tensors retain gradients. Pooling stage-2 depth 1 to canonical
depth 2 remains the original alignment operation and adds no physical detail.

The deep JEPA teacher sees the same unmasked visit. The online encoder sees a
mask applied to the input latent before any convolution. Level weights normalize
to one, and `mix` interpolates fused and deep objectives instead of adding a
second full-strength loss. Frozen teacher encoding of generated images is a
different path and remains differentiable with respect to the image.

Source verification used GitHub's commit and contents APIs for the exact pinned
files and licenses above, plus inspection of installed MONAI 1.5.1. Git retrieval
failed and raw HTTP was delayed, so the contents API provided the first verified
copy. No moving branch URL is used as a version identifier.

## Verification

`tests/test_backbones_v2.py` checks native zero-bridge numerical equality and
parameter-layout compatibility, influence of the earliest bridge on deeper
features and decoder output, all-parameter checkpoint gradient parity,
deterministic vector-field evaluations in training mode, delayed gradient
opening through zero-initialized semantic heads and attention gates, MONAI
1.5.1 equality against the actual unsplit backend, optional encoder-pyramid
compatibility, masked predictor gradients, and frozen-teacher input gradients.

On 2026-09-29, `/home/cxx/ispy2-symmflow3d/.venv/bin/python -m pytest
tests/test_backbones_v2.py -q` passed all 7 tests including MONAI 1.5.1.
The separate `/home/cxx/gastric_world_model_codex/.venv/bin/python` environment
passed 6 with 1 correctly skipped because MONAI is absent there.

A full-size native field forward/backward on NVIDIA GeForce RTX 5090 used
PyTorch 2.12.1+cu130, BF16 autocast and non-reentrant checkpointing. Shapes were
image `[1,48,8,32,32]`, semantic `[1,16,192]`, and dense `[1,32,192]`; loss and
all populated gradients were finite. The one-evaluation check took 1.13 seconds
and peaked at 697.8 MiB allocated CUDA memory. This excludes complete trajectory
rollout, optimizer state and encoder costs; it is not a long-run memory estimate.
