# DiT adaptation — CC BY-NC 4.0

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Source: https://github.com/facebookresearch/DiT/blob/main/models.py
Audited blob: c90eeba7b2eee18b40b2128045248795b4b38d91.

StateDiTBlock in src/responsewm/backbones.py adapts the adaLN-Zero block
architecture to semantic tokens, adds conditioned cross-attention and uses
PyTorch attention. It does not distribute DiT pretrained weights or the complete
original backbone. This DiT-derived adaptation retains CC BY-NC 4.0.
Legal text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
