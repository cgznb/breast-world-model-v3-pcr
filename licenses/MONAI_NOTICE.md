MONAI Contributors. Copyright (c) MONAI Consortium.
Source: Project-MONAI/MONAI, version 1.5.1,
monai/networks/nets/diffusion_model_unet.py, Apache License 2.0.

The forward traversal is split into encode/decode in MonaiImageBackbone so a
semantic bridge can be placed at the bottleneck. Class conditioning and
ControlNet additional residual arguments are not exposed. Actual MONAI modules
are a separately installed pinned dependency; no pretrained weights included.
