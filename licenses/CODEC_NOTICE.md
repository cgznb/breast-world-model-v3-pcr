# VQ codec exception: CC BY-NC 4.0

`src/symm_observation/codec.py` adapts the encoder, decoder, quantizer interface and checkpoint structure from:

- cgznb/symm-fm, commit `92265d3b1749ae3b686f2089843c49da129fd4d2`
- `workflows/first_post_three_phase/mewm_ispy2/vqgan.py`
- `workflows/first_post_three_phase/mewm_ispy2/first_post_world_data.py`
- The upstream workflow identifies this material as derived from MeWM: https://github.com/scott-yjyang/MeWM

The source workflow is under Creative Commons Attribution-NonCommercial 4.0 International, not the blanket MIT license of our original additions. Attribution to the original MeWM and cgznb/symm-fm contributors is retained. License: https://creativecommons.org/licenses/by-nc/4.0/legalcode

Changes in this adaptation: preserve parameter key names and numeric inference path; omit Lightning training/discriminator code; freeze codebook buffers; preserve a straight-through input derivative for decoded loss; add explicit three-phase wrapper and safe checkpoint-loading behavior.

No patient data or pretrained checkpoint is distributed. Their terms are separate. Commercial use of this derived codec requires rights beyond those granted by CC BY-NC 4.0. No original authors endorse the additions or any clinical use.

The exact upstream `three_phase_symmflow.py` is retained in `upstream_reference/` for review under the same upstream workflow license; its blob identity is recorded in docs/SOURCES.md.
