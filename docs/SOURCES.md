# Source-level provenance and licensing

Audit date: **2026-09-28**. This document distinguishes adapted source, actual
runtime dependencies, and conceptual references. It does not claim to reproduce
published benchmark numbers, download every repository, or use pretrained
weights that were not provided.

## 1. User's V2 code — adapted source and compatibility target

Repository: https://github.com/cgznb/breast-world-model-v2-pcr

Audited commit: `11220d38dd076439951d08928335d5fe4fa880b7`.

| Audited file | Git blob SHA | Use here |
|---|---|---|
| `src/symm_world/encoder.py` | `5fba5f3ef1773f06d70edaa2c2749014393f3651` | `legacy/encoder.py`: same encoder parameter layout; no deterministic future-prior reuse |
| `src/symm_world/layers.py` | `74827adb7d2fb3b0d38039d6babfaf0843b04147` | `legacy/layers.py`: Swin, ConvNeXt, QueryPool, mask and checkpoint building blocks |
| `src/symm_world/velocity.py` | `d795bc2cc6978561e7cf9b5297ef3cd55d940c84` | `backbones.py`: full native U-Net layout and MONAI initialization |
| `src/symm_world/codec.py` | `5a928b6692746ab0d3a722ed8075853dfeb318c3` | `legacy/codec.py`: matching continuous three-phase codec and frozen nearest-code decode |
| `src/symm_world/training.py` | `ee9314c083ab2dbf8f0df3f878f1fc79b310f6ea` | Audit of original checkpoint format, stages and data boundaries; new trainer implemented separately |
| `src/symm_world/data.py` | `d292f793653b061866fdf247d200caf1eff4b23d` | Existing manifest/view contracts; new prospective landmark adapter |
| `pcr/src/tdn.py` | `b3ed51af2a40fb0bc612ba7aafdc1dcc435c7fa8` | Time-aware query readout and residual clinical-prior design reference; not a claim of checkpoint equivalence |

Original additions retain MIT; original Swin adaptation retains BSD-3-Clause;
VQ remains CC BY-NC 4.0 (MeWM-derived), including source attribution to
`cgznb/symm-fm` commit `92265d3b1749ae3b686f2089843c49da129fd4d2`,
`workflows/first_post_three_phase/mewm_ispy2/vqgan.py`.

## 2. MONAI — actual optional dependency and split-forward adaptation

https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/diffusion_model_unet.py

Blob `cb0c69d033c3c7fc2e17758ded3ce0e95568e877`.

The production MONAI backend instantiates the actual `DiffusionModelUNet` and
uses its actual down/middle/up modules. Its forward traversal is split explicitly
at the bottleneck to insert the semantic bridge. No monkeypatch or mutable
forward hook is used. Version is pinned to **1.5.1**; upgrading needs a new parity
audit. MONAI is Apache-2.0; license retained under `licenses/`.

A parity test compares the split forward against unmodified MONAI when installed.
The delivery environment had no MONAI, so this test is **skipped**, not passed.
No pretrained MONAI image weights are bundled.

## 3. DiT — semantic block architecture adaptation

https://github.com/facebookresearch/DiT/blob/main/models.py

Audited blob `c90eeba7b2eee18b40b2128045248795b4b38d91`, DiTBlock/FinalLayer.
Paper: https://arxiv.org/abs/2212.09748

`StateDiTBlock` adapts adaLN-Zero residual gating to semantic patient-state tokens,
adds conditioned cross-attention, and uses PyTorch MHA instead of importing a
natural-image timm backbone. Nine modulation vectors control self-attention,
cross-attention and FFN. This is not pretrained DiT and not a complete DiT
benchmark reproduction. The DiT-derived block retains CC BY-NC 4.0 attribution.

## 4. ConvNeXt and Video Swin — reused via audited V2 adaptations

https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

Audited blob `74c1e9407bb4096a3b27127515573e32865b40f7`.
Paper: https://arxiv.org/abs/2201.03545

Depthwise convolution, channels-last LayerNorm, 4× pointwise expansion and layer
scale are preserved in 3D. Source license MIT.

https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py

The V2 attribution pins blob `1a198142874224a6766f321d9e0dfc97a01ecb43`.
This package retains that existing adapted 3D window partition, cyclic mask,
relative bias and padding-mask logic; it does not claim to load torchvision
pretrained video weights. Source license BSD-3-Clause.

## 5. I-JEPA — training reference, not copied full model

https://github.com/facebookresearch/ijepa/blob/main/src/train.py

Audited blob `0f387d6b293a7708037235f2bc34a8cb12953dc4`, target/context forward.
Paper: https://arxiv.org/abs/2301.08243

EMA stop-gradient targets and masked representation learning informed the
representation stage. The new encoder is the user's 3D latent encoder, not the
I-JEPA ViT. No I-JEPA weights or full training source are redistributed.

## 6. REPA and iREPA — alignment reference

https://github.com/sihyun-yu/REPA/blob/main/loss.py

Audited blob `ae8f3f89217b074ca9909f41bf8c9829dd31e4fd`.
Paper: https://arxiv.org/abs/2410.06940

https://github.com/End2End-Diffusion/iREPA

Audited README blob `923bda64dddbc1ce6755679f727b4bcd1171b86d` in the preceding
architecture review. Paper: https://arxiv.org/abs/2512.10794

The implementation uses a 3D convolution projection and explicit spatial
centering/normalization with frozen target features. Its exact weighting and
3D medical setting are task-specific, not a verified reproduction of iREPA's
natural-image results. Global Pillar features cannot satisfy the dense alignment
contract. No REPA/iREPA pretrained weights or full source trees are bundled.

## 7. Flow Matching — path/solver mathematical reference

https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py

Audited blob `81cb7ed31f2434d03424ea9a5571a36bfc9f2681`.
Paper: https://arxiv.org/abs/2210.02747

The affine-path endpoint/derivative conventions were checked against the official
implementation. `flow.py` implements the user's joint symmetric path with
matching semantic coordinates and differentiable Euler/Heun locally. It does not
import or redistribute the full official CC-BY-NC flow_matching package.

## 8. Brain-WM — problem/architecture context only

https://github.com/thibault-wch/Brain-GBM-world-model

Paper: https://arxiv.org/abs/2603.07562

The joint imaging/clinical-task framing informed the earlier design discussion.
No Show-o2/Qwen/Wan model, Brain-WM treatment planner, segmentation aligner or
pretrained language-model weights were copied into this implementation. Calling
this package a Brain-WM reproduction or an established causal treatment planner
would be inaccurate.

## New implementation

The pure-function bidirectional bridge, source-only typed trajectory API,
probability-marginalized outcome objective, grounding/energy integration,
landmark data contract, new history/readout modules, training orchestration,
strict migration and tests are task-specific implementation work. These are
research components, not independently established methodological novelties.

Public source review does not substitute for actual clinical training, numerical
backend testing or a current novelty search before submission. All citations
identify what was consulted; no benchmark superiority is inferred from them.
