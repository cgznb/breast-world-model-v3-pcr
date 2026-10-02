> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# 设计到实现的对应

| 设计 | 实际实现 | 测试/约束 |
|---|---|---|
| V2 三相共享表征 | `legacy/encoder.py`、`legacy/layers.py` | 保留 encoder 参数键，24通道合同，掩码先于卷积 |
| 真实/生成历史 H_t | `temporal.HistoryTransformer` | 仅传入截断前缀，缺失 token padding，真实/生成/计划角色分离 |
| 多区间治疗计划 | `temporal.AvailableConditioner` | 字段 missingness 与 known_at 验证；plan tokens 可见当时已知计划 |
| 全尺寸图像流 | `backbones.NativeImageBackbone` / `MonaiImageBackbone` | `reports/full_shape_native_forward.json`；MONAI parity 测试依赖可选库 |
| 随机语义流 | `StateDiTBlock` ×4 | 独立噪声、零初始化条件门控、无速度场 dropout |
| 双向空间/语义交互 | `BidirectionalBridge` ×2 | 交换前双方状态一致，纯函数，无 forward hook 可变状态 |
| 多步成对未来 | `model.sample_interval` / `forecast` | 每条轨迹独立历史、多段反向传播测试 |
| 同一样本对应 | `losses.rollout_outcome_loss` grounding | 冻结教师保留输入梯度测试；soft 约束而非绝对保证 |
| 轨迹 pCR | `TrajectoryPCRHead` | 3层完整序列 Transformer、时间差/状态差、临床先验和有界未来残差 |
| 概率积分 | `marginal_bernoulli_nll` | logsumexp 数值稳定、Jensen 对照、极端 logits、缺失标签零梯度 |
| 随机分布锚定 | `energy_score` | 同患者样本、缺失子轨迹、样本重排不变性、K>=2 |
| 真实轨迹训练 | `real_sequence` / `real_pcr` | 不作为 early-inference 输出；仅训练监督 |
| 原始 VQ 解码 | `legacy/codec.py` | 原结构 roundtrip、冻结 codebook、输入 STE 梯度测试 |
| 真实未来不能进入推理 | `contracts.py`、`inference.read_request` | 类型签名、监督污染不变性、文件级未来不读取测试 |
| 不误用相邻缺失 | `flow_loss` / `PatientSampler` | 无相邻真实配对则跳过该 FM 项，建议 direct-interval case |
| V2 checkpoint 初始化 | `checkpoints.migrate_v2` | 全键/维度严格核验；synthetic compatible state_dict 测试，不是实际私有权重实测 |
| 四阶段训练、恢复 | `training.py` | A/B/C/D 训练+部署测试，断点与不中断参数逐项相等 |
| 数据泄漏审计 | `data.ManifestStore` | 患者跨 split、字段 availability、未来目标不匹配、统计量仅训练集 |
| 概率与不确定性检查 | `metrics.py` / `training.evaluate` | 患者聚类与 paired bootstrap、AUROC/AUPRC/Brier/NLL/可靠性分箱 |

## 有意不做的内容

没有宣称把 MeanFlow、Brain-WM、REPA-E、VLA-JEPA、VRNN 等模型全部复现。代码没有加入一个没有训练依据的“单步 MeanFlow”；没有为 pCR 做 label-conditioned 生成；没有任意一套预训练 MRI foundation model 的通用键名转换；没有利用未知的未来治疗执行记录；没有将多样本方差自动叫作临床不确定性。

未包含原始患者 DICOM 处理、真实配准质量控制、Pillar 权重、原历史 TDN 全量代码、真实病理标签、五折划分或训练后检查点。这些不是用随机 mock 代替，而是要求由你已有工作流提供。

“所有逻辑”在此指已经落地的主联合架构、五类核心损失、四阶段优化、数据隔离与可执行接口。选做研究扩展如单步蒸馏、独立治疗因果识别、跨架构基础模型转换、原外部分类器复刻不在本版本中冒充完成。
