# VQ codec adaptation — CC BY-NC 4.0

`src/responsewm/legacy/codec.py` is adapted from the user's
`cgznb/breast-world-model-v2-pcr/src/symm_world/codec.py`, itself adapted from
MeWM-derived `cgznb/symm-fm` commit
`92265d3b1749ae3b686f2089843c49da129fd4d2`,
`workflows/first_post_three_phase/mewm_ispy2/vqgan.py`.

This component retains the Creative Commons Attribution-NonCommercial 4.0
International license; it is not relicensed as MIT.
Legal text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
Original project: https://github.com/cgznb/breast-world-model-v2-pcr

Changes: standalone frozen inference codec; weights-only checkpoint loading;
explicit matching of original numeric/shape contracts; continuous three-phase
encoding and nearest-code decoding; straight-through approximate input gradient.
No original patient files or pretrained model weights are included.
