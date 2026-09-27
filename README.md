# Breast World Model V3 + pCR

三相乳腺 MRI 世界模型 V3，以及完整的外部 PCR 预测代码。
本次快照整理于 2026-09-27，使用实际 ROI32 A/B 工作流。

## 代码组成

| 目录 | 内容 |
| --- | --- |
| `src/symm_observation/` | 单访视 JEPA 表征、StateAdapter、SymmFlow、VQ 接口、训练与推理 |
| `configs/registered_roi32_5090.yaml` | 实际 V3 ROI32 配置 |
| `pcr/` | 冻结 Pillar、TDN、临床先验、训练、预测与验证代码 |
| `vendor/mewm-ispy2/` | 原 ROI32 数据、预处理和 VQ 依赖源码与许可 |
| `tests/` | 表征、生成、数据合同和集成测试 |

A 阶段仅学习当前访视：三相 patch 编码、跨相位 Transformer 和 12 层 ViT，
配合同访视双裁剪 JEPA 与重建。B 阶段冻结 A 的 EMA 编码器，用 StateAdapter
聚合成 32 个 patient tokens，再结合临床条件驱动双分支 SymmFlow。
V3 的 A 阶段没有内部 PCR 头；PCR 由 `pcr/` 的独立分类流程完成。

## 安装和运行

推荐 Python 3.11；本次基础代码与 PCR 单元检查使用 Python 3.12。
PyTorch 请按设备安装；正式生成使用 MONAI 后端。

```bash
python -m pip install -e '.[test,monai,pcr]'
python -m pytest -q
python world.py smoke --output /tmp/world-v3-smoke
python world.py --help
python pcr/run.py --help
```

真实数据训练与生成命令见 [GENERATION_GUIDE.md](GENERATION_GUIDE.md)，
现有 ROI32 的适配见 [docs/REGISTERED_ROI32_LOCAL.md](docs/REGISTERED_ROI32_LOCAL.md)。
外部 PCR 的训练、特征提取和独立预测见 [pcr/README.md](pcr/README.md)。
网络设计见 [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)，本次验证见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 可复现范围

上传包含源码、配置、测试和说明。患者数据、特征缓存、真实划分及所有训练权重均需在本地提供。
生产入口要求匹配的 VQ 和生成器权重；合成 smoke 只能验证工程链路。
历史环境路径已替换为通用占位符；历史完整队列脚本需要设置对应的数据和实验输出路径。
`GENERATION_GUIDE.md` 的原交付时期说明属于历史背景，本次整理没有重新进行完整临床训练。

V2/V3 的生成器版本与 PCR 的 V1/V4 分类器版本是不同的命名。
PCR 使用完整生成轨迹的概率平均，再做同一种子下五折模型的概率平均。
生成器开发队列不是独立端到端测试集。

许可见 [LICENSE](LICENSE)、`licenses/`、`pcr/LICENSE` 和 `vendor/mewm-ispy2/LICENSE.md`。
新增入口的说明见 [docs/RELEASE.md](docs/RELEASE.md)。
