# 来源审计与改编边界

本版是依据用户指定论文和已读取源码作出的研究适配，不宣称属于作者原版、已通过医学验证、已确立优先权或具有论文接收保证。

## CheXWorld

论文：Yue et al., “CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning”, 用户提供 arXiv:2504.13820v1。

- https://arxiv.org/abs/2504.13820
- https://github.com/LeapLabTHU/CheXWorld
- `models/jepa_add.py`，核验blob `32eb9b05e797cfb978d6f87a6a6c6e25ee97dc2c`：`IWM_Dual`的双context、双teacher、四监督复用。
- `models/jepa.py`，此前核验blob `b62ba254c84320e12195dc4ad56b63e4f1e4d664`：EMA/stop-gradient/target feature规范化。
- `models/jepa_vit_add.py`：conditional predictor的接口和空间/变换条件思路。
- `models/jepa_vit_3d.py`，此前核验blob `cdc2a9b2ede3dca4485532c73fe0c65fecb3b521`：只提供PatchEmbed3D，不是完整三维CheXWorld。
- `data_utils/world_model_transform.py`，此前核验blob `7462325dc0001bb0c34409a04636f701d5eb7d53`：记录augmentation parameters。

对应本版：`observation_encoder.py`、`observation_views.py`、`observation_heads.py`、`observation_ssl.py`。采用独立实现的3D/三相适配，没有复制整套未审计许可的上游CheXWorld源码，也没有装作加载了其X-ray预训练权重。transform参数和三维网络配置均与原文有所不同，见DESIGN。

## AlphaCell

用户提供预印本：“Towards building a World Model to simulate perturbation-induced cellular dynamics by AlphaCell”, doi:10.64898/2026.03.02.709176，2026年3月版本。

- https://doi.org/10.64898/2026.03.02.709176
- Figure 2与Methods §2.2–2.3：Base Model观测重建、随后重建+ArcFace/DANN，独立Flow Model。

仅借鉴观测重建、表征/动力学分工及可选semantic projector/域头思路。未核验其完整公开实现，不声称复制Mamba/MoE/OT-CFM/大decoder，也不转载PDF或原图。其单细胞结果不能当作MRI结果。

## 用户原始三相SymmFlow

- https://github.com/cgznb/symm-fm
- 核验commit `92265d3b1749ae3b686f2089843c49da129fd4d2`。
- `workflows/first_post_three_phase/mewm_ispy2/three_phase_symmflow.py`，blob `6ca00eb2a2bdfd7039dbda8688a88674fd970179`。完整原文件参考副本在 `upstream_reference/`，本地计算git blob一致。
- `configs/mewm_all_pairs_5090.yaml`，blob `1b9e9936c1c02f6c4d77415fbb7922629dbbbd08`。实际velocity为4尺度64/128/256/256，不沿用此前误写的3尺度配置作为“原网络”。
- `three_phase_all_pairs_training.py` / `three_phase_all_pairs_data.py`：准备身份检查、source-only接口、pair-specific views、train/val缓存区别。

`codec.py`是原MeWM-derived VQ的可加载inference子集，CC BY-NC 4.0；不分发权重。`flow.py`保留sigma_min=0的原双分支数学路径。`scripts/three_phase_observation_bridge.py`在原class上明确添加source-only observation features，不改原VQ坐标。

原SymmFlow上游项目：https://github.com/caetas/SymmetricFlow 。本版针对用户仓库的paired longitudinal适配，不声称复制该上游全部任务。

## MONAI与包内3D模块

- MONAI 1.5.1 `monai/networks/nets/diffusion_model_unet.py`：真实可选backend，https://github.com/Project-MONAI/MONAI/tree/1.5.1 。本包未安装MONAI的测试环境会skip，而不是偷换成native还声称MONAI通过。
- torchvision Video Swin：shifted-window和padding/relative-position设计来源，https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py 。保留BSD notice。
- ConvNeXt：depthwise/LayerNorm/MLP/layer-scale block适配来源，https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 。保留MIT notice；不声称使用预训练权重。

`native` velocity是自包含的多尺度3D U-Net研究实现，不等同MONAI。Stage A使用ViT而不是Swin；Swin模块在native velocity bottleneck使用。没有在本版把REPA/Consistency-FM/V-JEPA或未来JEPA模块混进A。
