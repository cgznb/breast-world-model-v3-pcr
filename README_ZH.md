# 乳腺纵向世界模型 V3

**代码版本 3.0.0。** 主任务是在只有真实 **T0 MRI 与基线临床信息** 时，生成 T1→T2→T3 的可能轨迹，再对各轨迹的最终病理完全缓解（pCR）概率取平均。

[English](README.md) · [完整运行指南](docs/V3_RUN_ZH.md) · [V3 固定协议](docs/MULTISTAGE_V3_PROTOCOL_ZH.md) · [发布说明](docs/RELEASE_V3_20261002.md)

![V3 详细框架](docs/figures/breast-v3-detailed-framework.png)

[放大查看矢量 PDF](docs/figures/breast-v3-detailed-framework.pdf) · [可编辑 SVG](docs/figures/breast-v3-detailed-framework.svg)

## 当前网络

- 每次访视包含三个 MRI 增强相位；冻结 VQ codec 编码得到连续 latent `[24,8,32,32]`。相位与 T0/T1/T2/T3 临床访视是两个不同维度。
- ConvNeXt/Swin 编码器提取 dense、anatomy、disease tokens，与年龄、HR、HER2、MammaPrint 及其缺失掩码形成患者状态。六层事件因果 history Transformer 保存真实/生成事件。
- 原生 3D U-Net 与六块语义 flow Transformer 联合生成下一次 MRI latent 和 disease tokens，通过 **三处双向注意力桥** 耦合；同一生成器逐段推演，每段使用 20 步 Heun。
- 生成状态用于继续推演。T0 主任务的 pCR 读出保留 **真实 T0 memory**，读取完整生成轨迹，逐轨迹 sigmoid 后取平均；logit 为训练集拟合的临床 prior 加有界残差。
- 新到达的真实 MRI 通过独立观测更新分支修正状态；自由生成不接收未来 MRI 真值。当前没有经过核验的治疗计划输入，不能解释为治疗因果模型。

## A–D 训练与选模

| 阶段 | 更新内容与选模 | 物理 batch |
|---|---|---:|
| A 表征 | 重建、JEPA、辅助 pCR；学习率 `3e-5`；固定重建/JEPA/T0 observed NLL 组合选模 | 160 |
| B 生成 | 联合 flow、grounding、自由生成；按生成均值误差相对 T0 持久性误差选模 | 32 |
| C 读出 | 仅 pCR 最后一块 Transformer 和输出层；学习率 `1e-5`；入口模型可保留 | 192 |
| D 联合 | 保守更新生成器、状态及读出；入口可保留；增加生成误差保护条件 | 32 |

B 的 marginal pCR 权重为零，但真实观测更新分支仍有 pCR 监督，梯度可通过冻结的 pCR 头传回状态和生成模块。C/D 主选模指标固定为 **T0 full-future marginal NLL**；D 的生成相对误差不能超过入口的 1.10 倍。

多种子队列允许两个 A 同时运行，B/C/D 连同阶段评估独占训练资源。批内按随访可用结构分组，缺失随访仍按缺失处理；batch 是物理任务数，不保证批内患者互不重复。正式配置来自 RTX 5090 32 GB 的资源方案，迁移硬件后应重新预检。

## 快速开始

Linux、Python **3.11+**；先安装与显卡匹配的 PyTorch/CUDA。V3 主配置使用 native 后端，MONAI 为可选依赖。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python joint.py smoke-v3 \
  --config configs/multistage_smoke_v3.yaml \
  --output runs/v3_smoke
```

smoke 使用合成数据，只验证工程流程；输出目录应为新目录。本次发布实际验证以 [reports/release_validation.json](reports/release_validation.json) 为准。真实数据准备、完整 A→B→C→D、恢复、评价及多种子命令见 [运行指南](docs/V3_RUN_ZH.md)。

V3 继续使用 V2 患者数据 schema 和准备脚本，因此清单名仍为 `patient_trajectories_v2.json`；这不表示 V2 checkpoint CLI 能读取 V3 权重。V3 使用 `train-v3`、`smoke-v3`、`evaluate-v3`。

## 发布内容与证据范围

本次发布包含最终版 **代码、配置、测试、文档和框架图**，不包含患者数据、逐患者结果或训练权重，也不表示正在进行的训练已经完成。开发协议为 764 名训练患者、102 名验证患者，没有独立测试集；8 人影像检查是预定的开发验证子集。

发布整理前，本地 83 个 Python 文件已与正式部署源代码归档一一匹配；打包不改变模型与训练逻辑。来源摘要、发布调整和核验限制详见 [发布说明](docs/RELEASE_V3_20261002.md)。旧版说明已归档至 [历史中文首页](docs/historical/README_PRE_V3_ZH.md)；历史工程报告不能作为当前 V3 性能结果。

本项目保留混合许可：新原创部分为 MIT，codec/DiT 派生部分保留 CC BY-NC 4.0，其余第三方部分保留原许可。请同时阅读 [LICENSE](LICENSE)、[licenses/](licenses)、[来源说明](docs/SOURCES.md) 与 [多阶段来源说明](docs/MULTISTAGE_NETWORK_SOURCES.md)。
