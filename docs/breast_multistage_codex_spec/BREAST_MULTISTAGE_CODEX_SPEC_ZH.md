> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# 乳腺多阶段连续患者状态模型：Codex 实施规范

**版本：1.0 · 日期：2026-09-29 · 性质：待实施的工程/算法设计，不是已经修改后的代码。**

目标仓库：`cgznb/medical-world-models`。改动范围：`breast_world_joint/`；不得顺手重构胃癌项目。

本次核验基线提交：`841458cb8e06e131af2a2248dc853f8476b2e0d0`（main）。实施时先记录本地 HEAD；若版本不同，逐文件对照本规范的类名和行为，不得强制 reset、覆盖用户新代码或沿用失效行号。[R0]

建议工作分支：`research/breast-causal-multistage-state-v2`。

## 交付目标，一句话说明

在保留 **VQ → 共享三相状态编码器 → 历史建模 → 图像/语义双流 SymmFlow → pCR** 主体的前提下，把实际 T0→T3 单跳实验升级成：

> 同一套共享转移网络，连续执行 T0→T1→T2→T3；真实 MRI 到达后更新患者状态；无新 MRI 时沿当前假设继续预测；同一个 pCR 读出头在每个合法真实观察前缀和每个明确标记的模拟状态上可查询。

完整交付必须包含网络、数据构建、训练、在线接口、评估、消融、检查点迁移、单元测试和中文说明。不能只增加 `for` 循环、把配置文件改名或为每个阶段单独训练分类器。

**本规范中的层数、权重、课程和阈值是建议的起始设计，尚未在该队列验证。公开论文提供结构参考，不保证这种组合有效，更不能保证会议录用。**

---

## 0. Codex 开始前必须遵守的规则

1. 先审阅当前工作树、基线配置、训练日志和实际数据合同，再写代码。不得伪造存在的患者文件、时间戳、治疗计划、预训练权重或 GPU 测试结果。
2. 保留现有 T0→T3 baseline 的复现入口、旧 checkpoint 读取及原始结果；新模型和新数据用 `v2` schema，不能悄悄改变旧接口语义。
3. 保留已实现的复杂主干。主要新增工作是状态生命周期、时间因果约束、多阶段监督和可测试的交互接口，不是再堆一个很大的基础模型。
4. 先完成本文 R1 必选范围。R2 真正临床连续时间扩展是条件性研究项，默认关闭，不能为了完成任务把阶段索引当成天数。
5. 可以复用允许使用的公开代码，但必须记录仓库、commit、原始文件、许可证、修改范围和来源测试。缺少许可证不能视作任意复制许可。
6. 不得自动启动完整真实患者长训；先交付测试、预检命令与配置。运行真实训练需要确实有数据/算力，并如实记录实际执行情况。
7. 每次新增 loss 必须回答：监督从哪里来、掩膜是什么、梯度到哪里、推理是否可得、有何消融。禁止“为先进而加 loss”。

## 1. 已核验事实，以及本规范对前述讨论的修正

### 1.1 当前实际情况

- `model.py::forecast()` 已经按 `future_days` 的 F 维循环生成，并把每次生成加入历史；它不是完全不具备多步能力。[R1]
- 正式实验入口使用 `direct_t0_t3.json`，一个 T0→T3 查询，F=1。`prepare_existing_ispy2.py` 同时可导出 `longitudinal.json`，但仓库记录其尚未正式多区间训练。[R2][R3]
- pCR 主要在完整 rollout 后调用一次 `read_trajectory()`，返回 `[B,K]`，没有逐阶段 prior/posterior 的在线状态协议。[R1]
- `flow_loss()` 每患者抽一条可用相邻边做 teacher-forced FM；B 阶段未把 source-only 多段 rollout 作为主训练路径。[R4]
- 原 `grounding` 已在 `[B,K,F,...]` 上按所有有效 F 计算；原 Energy Score 也已覆盖可观察子轨迹，不应删除后重新包装成完全新贡献。[R4]
- `memory()` 将所有请求位置及其计划 token 引入历史上下文，当前状态和“请求展示哪些未来时点”没有完全解耦。[R1]
- 当前公开数据协议为 764 名训练患者、102 名开发验证患者，临床四维，治疗零维；没有独立测试集。具体数字是仓库报告，不是本次重新读取患者数据后的统计。[R2]

### 1.2 必须纠正的设计误区

**抽一条边 ≠ 不科学。** 如果抽样与目标权重对应，单边随机训练可以是无偏估计。真正需要的是所有阶段有可量化覆盖，以及多步条件分布的训练。提供 `all_edges` 和 `balanced_edge_sample` 两种实现，不强制每个 batch 保存所有大图计算图。

**模拟 T1 ≠ 新观察到 T1。** 从 T0 噪声生成的 T1 不增加一项实际患者检查。模型内部 `forecasted_state_pcr@T1` 与真实 T1 到达后的 `observed_landmark_pcr@T1` 必须命名、监督和评估分开。

**增加时间 embedding ≠ 连续临床时间系统。** 本次主版本在 T0/T1/T2/T3 上连续更新状态，属于阶段性纵向模型。不能仅把 `stage_index` 换成 `calendar_days` 就声称学会任意天数的可靠演化。

**零门控 ≠ 一概安全。** 新观察更新不能因多重零初始化而完全不依赖 MRI。新患者初始化必须已有直接的观察路径；观察校正采用非零初始 gate 或单零初始化。图像侧新残差桥可以从零开始，但不能把门控和残差输出同时置零造成梯度死区。

**未来生成不增益的原因未知。** 仓库显示当前生成 T3 尚未改善 pCR，这支持重做对照和动态建模，不证明“缺少 T1/T2 就是唯一原因”。编码器过拟合、生成失真、分类捷径、尺度漂移等都要检查。[R2]

## 2. 版本边界：必须实现什么，哪些不能冒充完成

### R0：保留的原始对照

保留 `responsewm_v1` 的单跳 T0→T3 模型及输出。冻结其数据合同与评价设置；不得把旧结果修改成新模型结果。

### R1：本次必须完整实现

- 阶段索引下的共享多段转移、带真实观察更新的持久患者状态。
- `initialize / advance / observe / forecast / query_pcr` 五个接口。
- 当前状态、真实事件、预测分支、查询请求相互隔离。
- 同一个分类头支持真实观察前缀、模拟前缀和生成完整未来后的边缘化读出。
- 规范阶段网格 `[0,1,2,3]`，请求只影响返回哪些结果，不决定内部跳过哪些状态。
- 全阶段配对覆盖、source-only 多步训练、逐阶段与整轨迹分布监督、观察更新训练。
- 仍采用现有双流联合积分：图像未来与语义未来在同一次积分中共同生成。
- 当前真实数据 `action_dim=0` 可以完整运行；不能虚构治疗记录。

### R2：只在真实事件/日数可审计时启用

真实天数的无事件推进、MRI 之间临床事件更新、任意合法日数查询，需要额外的临床时间动力学、输入数据与验证。第 23 节给出设计边界。本次 R1 的验收不允许以“有个时间 MLP”替代 R2，也不允许因 R2 数据缺失而不完成 R1。

## 3. 统一术语与概率解释

### 3.1 三种时间，不能混用

| 名称 | 符号 | 含义 | 主版本处理 |
|---|---|---|---|
| 临床阶段 | j | T0/T1/T2/T3 | 必选，0/1/2/3 |
| 临床物理时间 | t | 从基线起经过的真实时间 | 仅审计通过后使用 |
| Flow 积分时间 | τ | 从噪声端点积分到目标端点的数值坐标 | 无量纲 `[0,1]` |

MRI 深度 D、同一次检查增强相位 P=3、患者纵向时间 J=4 是三条不同维度。不能将 D 当作纵向时间，也不能将三个增强相位作为 T0/T1/T2。

### 3.2 状态的三种坐标

- `Z_j`：空间连续 VQ latent，`[B,24,8,32,32]`。
- `O_j = E(Z_j)`：同次 MRI 的观察 tokens，32 dense + 4 anatomy + 8 disease，各 192 维。
- `S_j`：生成器的 disease 坐标，`[B,8,192]`，必须对应固定教师 `E_bar(Z).disease`。
- `M_j`：多事件形成的持久记忆，`[B,16,192]`，是历史信息的载体，不等同于 `S_j`。

**禁止把历史 Transformer 输出的任意 8 个 memory token 直接当作固定教师的 disease 目标。** 这两种空间需要明确映射和监督，不能因为维度相同就认为语义相同。

`M_j^-` 表示真实新观察到达前的预测记忆；`M_j^+` 表示吸收了合法新观察后的记忆。“posterior”仅为学习式观察校正的术语；本版并未建立精确 Bayesian posterior likelihood。

### 3.3 三种 pCR 输出

`pcr_observed`：根据当前真实可用历史预测最终 pCR。T0、T1、T2 都可用同一最终标签 Y 监督，但输入必须分别截断。

`pcr_forecasted_state[j,k]`：第 k 条模拟轨迹在第 j 阶段状态上的结局读出。它是模拟诊断，不能当作已经见到真实 Tj 的临床预测成绩。

`pcr_marginal`：在真实历史起点，对 K 条完整生成未来的结局概率求均值。这是“用生成未来辅助当前预测”的主结果。

同一 Y 可以监督多个真实 landmark；多次 loss 是一个患者的相关任务，必须归一化和按患者统计置信区间。不要将其当作多个独立患者扩充样本量。

## 4. 新结构总览

```text
真实 T0 MRI → 冻结 VQ → Z0 → ObservationEncoder → O0
                                                   │
                               非退化的观察初始化 + 历史编码
                                                   │
                          PatientState {Z0,S0,M0+,真实事件缓存}
                                    │                 │
                              shared pCR Q        advance 0→1
                                                      │
                                  interval condition + 独立噪声
                                                      │
                              图像 U-Net ⇄ 6层语义 Transformer
                                                      │
                                        (Z1_hat,S1_hat) 同时生成
                                                      │
                                 E_bar(Z1_hat) + 生成事件/状态更新
                                                      │
                                  {Z1_hat,S1_hat,M1-} → shared Q
                                                      │
               ┌──────────────────────────────────────┴─────────┐
               │没有真实 T1，继续预测                           │真实 T1 到达
               │                                               ↓
          advance 1→2                               O1_real = E(Z1_real)
               │                                               ↓
          (Z2_hat,S2_hat)                          ObservationAssimilator
               │                                               ↓
               ...                                  {Z1_real,S1_real,M1+}
                                                               │
                                                           shared Q
                                                               │
                                                          advance 1→2
```

`forecast()` 只在副本上展开；不会把假想 MRI 写入真实病历。收到真实 T1 后，旧预测分支中的 T1/T2/T3 不再是该后验状态的未来，必须失效或重新分叉。

图像和语义仍为同一个双流转移，**不要另加一个独立先验 Transformer 先决定 S1，再让生成器去追一个已经固定的 S1**，否则退回确定语义先验与随机图像不同步的问题。

## 5. 文件修改总表

| 路径（相对 `breast_world_joint/`） | 操作 | 关键职责 |
|---|---|---|
| `src/responsewm/contracts.py` | 扩展/分离 | v2 观察、区间、状态、轨迹、监督合同 |
| `src/responsewm/data.py` | 保留 v1 + 新加载器 | 患者轨迹、前缀截断、边索引、掩膜 |
| `src/responsewm/samplers.py` | 新增 | 患者/landmark/edge 三种归一化采样 |
| `scripts/prepare_existing_ispy2.py` | 向后兼容扩展 | 导出真实轨迹及统计，不覆盖原清单 |
| `src/responsewm/legacy/encoder.py` | 可选返回多尺度 | 保持已存在主干参数键与默认行为 |
| `src/responsewm/representation.py` | 新增 | 多尺度深层 JEPA、teacher、token pool |
| `src/responsewm/state.py` | 新增 | 持久状态、观察同化、生成状态提交 |
| `src/responsewm/temporal.py` | 新建 v2 类 | 因果事件历史与统一 pCR 查询 |
| `src/responsewm/conditioning.py` | 新增 | 条件类型分离、区间 action、resampler |
| `src/responsewm/backbones.py` | 扩展 v2 类 | 多尺度桥、六层语义场、可拆分 U-Net |
| `src/responsewm/flow.py` | 保留数学路径 | τ 积分与 `NoiseLedger` 接口 |
| `src/responsewm/rollout.py` | 新增 | 规范网格、只读分支、失效/重放策略 |
| `src/responsewm/model.py` | 新增 v2 封装 | 五个公共接口，不破坏原类 |
| `src/responsewm/losses.py` | 拆函数 | 配对 FM、source-only rollout、同化、landmark BCE |
| `src/responsewm/training.py` | 分阶段扩展 | A/B/C/D，参数组、来源记录、真实验证 |
| `src/responsewm/inference.py` / `cli.py` | v2入口 | 在线初始化/更新/查询；输出先验后验类型 |
| `src/responsewm/metrics.py` | 按协议扩展 | 明确阶段、观察阶段集合、预测深度、聚类CI |
| `src/responsewm/config.py` | v2 schema | 所有新字段严格注册/校验 |
| `src/responsewm/checkpoints.py` | 新迁移器 | 结构/语义/统计/数据暴露报告 |
| `tests/test_*_v2.py` | 新增 | 第 21 节全部验收 |
| `docs/MULTISTAGE_*.md` | 新增 | 实现、命令、源代码引用、限制与实验 |

---

## 6. 数据层：从单跳 case 到患者轨迹

### 6.1 新 schema：`responsewm_patient_trajectory_v2`

一名患者只存一份真实轨迹记录；训练时构建有界前缀和监督，不复制文件。不要求把真实缺失扫描补齐。

```json
{
  "schema": "responsewm_patient_trajectory_v2",
  "time_basis": "stage_index",
  "canonical_stages": [0, 1, 2, 3],
  "clinical_features": ["age_at_screening", "hr_positive", "her2_positive", "mammaprint_binary"],
  "action_features": [],
  "phase_order": ["pre_aqc0", "first_post_aqc1", "metadata_late"],
  "latent_shape": [24, 8, 32, 32],
  "vq_identity": "sha256:REPLACE_WITH_AUDITED_HASH",
  "patients": [{
    "patient_key": "LOCAL_PSEUDONYM",
    "split": "train",
    "baseline": {"values": [47, 1, 0, null], "known_at": [0, 0, 0, null]},
    "visits": [
      {"stage": 0, "latent": "T0.npy", "available_at": 0},
      {"stage": 1, "latent": "T1.npy", "available_at": 1},
      {"stage": 3, "latent": "T3.npy", "available_at": 3}
    ],
    "events": [],
    "interval_plans": [],
    "target": {"pcr": 1, "label_source": "LOCAL_AUDITED_DEFINITION"}
  }]
}
```

数字与路径仅是格式示例，不是这名患者真实记录。几何与可用性声明仍需从原合同和审计记录继承；相同 grid/affine 不等于肿瘤解剖配准完全正确。未核验的 anatomy invariance 默认关闭。

`patient_key` 只用于加载/分折/采样账本，不传神经网络。缺失数据以 absent visit/null/mask 表示；不能把空影像零张量解释为真实完全缓解。

### 6.2 四种 mask 分开

`stage_valid`：该阶段是否属于患者研究轨迹的结构性范围。

`observed_mask(as_of)`：截至当前时点真实已收到的 MRI。

`target_available_mask`：训练时可作为未来监督的 MRI，绝不传入预测器。

`output_query_mask`：本次希望返回哪些阶段。它不决定内部转移网格，也不进入当前生理状态编码器。

任意辅助临床值另有 `value_known_mask`。值为 0 与未知两者不同；没有治疗数据维度 0 与明确观察到“无治疗”也不同。

### 6.3 前缀定义

T0：只可见 T0 与 T0 之前已知临床信息。内部生成 T1/T2/T3。

T1：可见真实存在且可用时间 ≤T1 的扫描；生成 T2/T3。

T2：可见真实存在且可用时间 ≤T2 的扫描；生成 T3。

T3：可见术前真实 T3；仍预测最终病理 pCR。**最终病理结果和手术后信息不得输入术前 T3 头。** 术后病理 label 已知时不再作为“预测未知 pCR”的同一任务。

当 T1 缺失、T2 存在，T2 prefix 观察集合是 `{T0,T2}`，不能命名成完整 `{T0,T1,T2}`。输出中记录 `observed_stage_ids`，不要仅靠 `observed_visits=2` 分类实验。

### 6.4 规范转移网格与直接桥接

`forecast(T0, output_stages=[3])` 默认内部执行 **0→1→2→3**，只是返回 T3；`forecast(T0, output_stages=[1,2,3])` 采用相同转移，返回全部。相同 ledger 下两者的 T3 必须一致。

原始 0→3 单跳只能通过 `forecast_direct()` 或 `transition_mode=direct_ablation` 明确调用，不能作为新 sequential 默认入口的优化捷径。

真实配对可形成相邻边 01/12/23。若患者仅有 T0/T2，可把 02 作为 `bridge_auxiliary` FM 监督，明确是跳步辅助目标；不能假造 T1 真值，也不能把它计进 01/12 数据量。

所有相邻边类型没有数据时应给出清晰统计和限制，不能只依赖终点 BCE 宣称中间 MRI 已得到充分监督。

### 6.5 治疗与时间协议

当前主数据 `action_features=[]`。模型必须接受零维，并使用 `ACTION_UNOBSERVED` 非因果占位状态。它不表示 `NO_TREATMENT`。不要虚构化疗周期、放疗、手术或术后辅助治疗来充实架构。

将来添加区间治疗须记录：`known_at`、计划开始/结束、字段含义与单位、计划/已实施类别。预测 0→1 的动力学只使用截至源时点已知且作用于该区间的 controls。即使 2→3 的计划在 T0 已确定，它可影响“最终结局预测条件”，不能因为全局 attention 而无缘无故改变治疗尚未开始前的影像。

同一时间的事件以 `(available_at, clinical_order, event_id)` 排序；报告延迟使用 available_at 决定可见性。晚到报告不能 retroactively 改写之前导出的预测，而应形成新版本状态。

## 7. 观察编码器：保留主干，增加深层监督

### 7.1 不变的逐层尺寸

| 层 | 运算 | 输出（省略 B） |
|---|---|---|
| 原 latent | 24 通道 = 三相×8 | `24×8×32×32` |
| 相位重排 | 三相并入 batch | `8×8×32×32` |
| stem | Conv3D 8→48，k3/s2/p1 | `48×4×16×16` |
| stem细化 | ConvNeXt3D(48) ×2 | 同上 |
| phase mixer | 每体素三相 token；2层、d48、6头、FFN192 | 三相各48 |
| 相位融合 | concat144 → Conv1×1×1→96 | `96×4×16×16` |
| 相位差支路 | 连续VQ差；Conv24→96,k3/s2 + ConvNeXt96 | 同上；与融合相加 |
| Stage0 | Swin3D ×2，d96，3头 | `96×4×16×16` |
| Merge1 | PatchMerging3D 96→192 | `192×2×8×8` |
| Stage1 | Swin3D ×2，d192，6头 | 同上 |
| Merge2 | PatchMerging3D192→384 | `384×1×4×4` |
| Stage2 | Swin3D ×4，d384，12头 | 同上 |
| 多尺度投影 | 各尺度Conv1×1×1→192、pool至2×4×4 | 三份`32×192` |
| 融合 | concat576→Linear384→GELU→Linear192→LN | `32×192` |
| 空间位置 | 原3D位置编码 | 同上 |
| Query pools | 4 anatomy queries，8 disease queries | `4×192`,`8×192` |

以上是现有接口的目标保留形状；实际 forward 与 state_dict 以锁定 commit 为准。[R5]

Stage2 的深度维从1池化至2不产生新的空间分辨率。为了 checkpoint 兼容可以保留原融合路径；深层监督应注明这种对齐，不能当作新增体素细节。

### 7.2 新返回类型

`forward(z, return_pyramid=False)` 默认返回原 `PatientState`，原测试/权重路径不变。

开启时额外返回各尺度 pre-pool feature 与 canonical projected tokens，供训练和诊断使用。禁止用 `.detach()` 切断在线分支；teacher target 可在生成后 stop-gradient。

### 7.3 Deep JEPA，明确不是提前预测下一次MRI

借鉴 V-JEPA 2.1 的 intermediate/deep supervision 与 dense feature 训练，而非复制其视频预训练规模。[R7]

在已存在 4 层 `MaskedStatePredictor` 外新增三个 level embedding 与三个输出投影。可以共享 predictor 主干、按 level 分组前向；不能只比较原始可见特征而完全没有 mask/predictor。

每层：`LN192 → Linear192→384 → GELU → Linear384→192`。teacher 对应同一个真实 visit、相同几何坐标、未遮挡输入的 EMA 编码器。先在 latent 空间遮挡再卷积，避免 encoder 从遮挡后的隐藏 token 周围提前读到完整原数据。

默认：`masked_weight=1.0`，`visible_weight=0.1`；level raw weights `[0.2,0.3,1.0]` 归一化为和1。保留原 fused-level masked JEPA，但用 `deep_jepa_mix` 在原fused与三层loss间混合，避免无意重复翻倍权重。

Stage A **不做 T0→T1 JEPA 预测**。跨时间 dynamics 在 Stage B。variance/covariance 防塌陷继续保留但不要把空间 token 当独立患者报告统计。

---

## 8. 持久患者状态与真实观察同化

### 8.1 `PatientBelief` 数据结构

```python
@dataclass(frozen=True)
class PatientBelief:
    version: int
    stage: int
    as_of: float
    memory: Tensor             # [B,K,16,192]; 观测-only K可为1
    anchor_latent: Tensor      # [B,K,24,D,H,W]
    anchor_disease: Tensor     # [B,K,8,192]; 固定教师/生成S坐标
    anchor_origin: Tensor      # observed / predicted / no_image
    clinical_snapshot: ClinicalSnapshot
    evidence_log: EvidenceLog  # 真实事件，不能被forecast污染
    model_state_log: ModelStateLog # 预测/后验状态与版本，非额外真实证据
    cache: CausalCache | None
    branch_id: str
```

`K` 表示模型状态样本/分支，不声称是经过 likelihood 重加权的精确粒子滤波器。按批/样本打平做网络，返回时还原 `[B,K,...]`。任何 averaging 必须显式指定是否只用于读出。

`anchor_latent` 是**当前状态**的图像锚，不一定最后一个真实MRI；`last_observed_*` 信息由 evidence log 追溯。真实新MRI到达后 anchor 必须换成真实 Z，不能继续拿旧预测 Z 当源图像。

### 8.2 初始化不能屏蔽影像

新增 `ObservationSeedPool`：四个 global queries，对 O44 做一次 cross attention，接LN与FFN192→768→192，得到 `G4`。

`M_seed = concat(A4, S8, G4)`，共16 token。加上 memory-slot embedding 与基线临床条件的受控投影，然后经过同化模块/历史模块。即使后续新残差门控为零，初始化也必须依赖真实 MRI。

验收：两张不同 T0、相同临床，`M0` 不能恒等；MRI→M0→pCR 梯度非零。不能通过只比较随机初始化的最终概率不同来代替结构依赖测试。

### 8.3 `ObservationAssimilator`：4层、192维、6头

输入：预测先验 M^-16、真实观察 O44、真实临床更新tokens、质量mask、事件时间；可选先验图像重编码 tokens，仅用于 innovation，不当真值。

每层固定次序：

1. `q=LN(M)`，`kv=LN(O_real)`，CrossAttention（192维、6头，每头32）。
2. `M←M+g_obs⊙CA(q,kv,kv)`。
3. `LN(M)` → memory SelfAttention（同维度）→ gated residual。
4. `LN(M)` → Linear192→768 → GELU → Linear768→192 → gated residual。

gate 用 `sigmoid(Linear([summary(M), summary(O), gap, quality]))`，偏置例如−2，不能所有 gate 都严格0；无观察 mask 时整个校正必须严格 identity。gate 的输入不允许包含标签或未来观察。

可构建 innovation：真实 dense 与先验重编码 dense 的差，通过 `LN→Linear192→192` 加入 CA 的value。只在 grid可比较时启用；不默认将 voxel差解释为真实肿瘤变化。

**状态更新规则：**

`M_plus = Assimilator(M_minus, O_real)`；`Z_anchor=Z_real`；`S_anchor=E_bar(Z_real).disease`。后两项是源图像/生成坐标的校正，不是 `M_plus` 强行等于单扫描编码。历史中的患者背景/治疗信息必须保留在 M 中。

### 8.4 同化不是删除历史，也不是反复计入同一证据

真实 observation 按唯一 `event_id + payload_hash` 去重。重复提交相同内容返回同版本/同结果；同ID不同payload必须报冲突，而非默默追加。

如果真实 T1 到达：废弃从旧先验T1继续展开的未提交T2/T3分支；保留旧预测做审计，但不让它们进入新后验的真实 history attention。T1 prior 可以作为模型内部 M^- 使用，不能把真实T1与假想T1当成两次独立测量。

`observe(prior, obs)` 只接收现在到达的真实 observation；不允许 API 从患者全轨迹对象中按 `future_mask` 自动补一张真实扫描。

### 8.5 同化如何训练

对合法真实连续观察序列，先由上一个 posterior 调用 `advance()` 得到 prior，再喂当前真实 O，执行同化；最终标签/未来配对只能在此后构造 loss。对每个真实 posterior 执行 shared pCR BCE，并训练后续 transition，从而让 M_plus 服务实际任务。

可选 `L_assim_reconstruct`：将 M_plus 的8个disease槽，经 `LN→Linear192→384→GELU→Linear384→192` 重建 `stopgrad(E_bar(Z_real).disease)`；权重小，作用于head输出，不把整个记忆拉回单次MRI表征。

可选 `posterior_readout_consistency` 比较同一真实前缀在在线更新与规范replay下的输出；测试层要求相同算法/ledger可复现，训练层不强制“预测前”和“观察后”风险相等。

---

## 9. Causal Event History：职责与逐层实现

### 9.1 与同化模块的分工

同化模块做**局部新证据融合**，因果历史模块做**跨事件长期上下文聚合**。不要让两个模块各自维护一份互相冲突的患者时间。

每次真实观察或生成转移提交一个模型事件块；只读查询不提交任何块。事件块内容由当时可用信息生成，创建后不因未来事件到达而重写其原始输入。

建议事件块：

`[EVENT_TYPE, TIME, CLINICAL_SUMMARY_1, CLINICAL_SUMMARY_2, M_slot_1...M_slot_16]`，共20 token。MRI事件另可附加当次44个O token；使用 mask 与 `token_role` 明确。首版可保留 O44 提高空间上下文，之后用消融决定是否压缩。

事件块输入的16个M是当前局部同化/预测seed；它依赖此前已完成的状态，因此按事件顺序 unroll。训练并行处理多个真实 landmark时，必须构造各自合法prefix或正确block-causal图，不能从全患者双向特征中切片。

`HistoryContextBuilder` 返回当前事件的16个上下文化M，写入 PatientBelief，并只保留源模型事件的原始块供 replay。KV cache 必须逐层匹配，不将已经多层变换后的token当作重新计算时的原始输入。

### 9.2 6层事件因果Transformer

每层：Pre-LN192 → MHSA6头 → residual → Pre-LN192 → Linear192→768→GELU→Linear768→192 → residual。

因果 mask 在**事件块级别**定义：同一合法事件内 tokens 可以双向交互；事件b只能读事件≤b。真实新观察所在事件不能反向改变先前输出。参考 V-JEPA 2-AC 的 block-causal action/frame 结构，但将机器人action、图像帧网格替换成医疗事件和三维latent观察。[R6]

所有进入 dynamics 条件的历史模块 dropout=0；分类头可保留训练dropout=0.1。禁止在ODE两次Heun评估之间随机更换history feature。

`action_01` 影响从0到1的转移上下文；其值不能被分配到T2事件再假装对T1有效。未来已知计划不加入当前physiology M，只在政策条件读出或其有效区间的transition context中出现。

### 9.3 缓存与身份

首版先全前缀重算，得到正确性基线；只有与全前缀数值等价后才启用KV cache。4次MRI并不需要为加速提前引入不透明的缓存逻辑。

cache identity 包含参数版本、clinical/time schema、observed事件hash序列、geometry/VQ身份、dtype/backend。新观察或计划改变时只保留合法前缀缓存。

---

## 10. 条件编码：不再把不同语义全局平均

### 10.1 `ConditionBundle` 明确分组

- `state_tokens`：当前M16。
- `clinical_tokens`：源时点可用的字段token，沿用数值×feature-embedding + missing embedding。
- `interval_action_tokens`：当前转移区间A个特征/药物/计划token，保留各自身份。
- `source_stage_token`、`target_stage_token`：Embedding4→192。
- `interval_token`：阶段gap或合法日数特征 → Linear→SiLU→Linear→192。
- `mode_token`：sequential / direct_aux / optional retrodiction。

未来路径整体计划可以传结局读出专属 `PlanContext`；不能借此绕过interval causal mask影响过去图像。先划分可见性，再做attention，不是把所有事件输入后靠网络自己学会不看未来。

### 10.2 learned ConditionResampler（2层）

四个learned query，形状 `[1,4,192]`。每层：LN + CA(query,合法bundle tokens)，residual；LN + SA(query)，residual；LN + FFN192→768→192，residual。

输出 `cond_tokens [B,4,192]`。`cond_global = Linear192→512(LN(mean(cond_tokens)))` 提供native U-Net FiLM，语义branch通过另一个Linear192→192获得全局调制。这里平均的是**已attention聚合的query**，不是原始异类条件直接平均。原 `pooled_context` 应被复用或替换为这一次192→512映射，不能再额外重复投影一个已经512维的条件。

保留state/action分组的原始tokens，供语义block的专属cross-attention；没有action时使用单个 `ACTION_UNOBSERVED`，并在报告中注明。

### 10.3 不复制不适合队列的action定义

本次不直接搬Brain-WM的手术/放化疗policy token或V-JEPA的7维机器人动作。只有乳腺实际数据字典允许的特征才能输入。临床最终标签、术后病理、真实换药结果、后续扫描时间不得作为基线已知条件。[R2][R6][R8]

---

## 11. 图像/语义双流：精确的多尺度重构

### 11.1 保持数学和通道合同

Joint image输入仍 `[B,48,8,32,32]`，输出同形速度；Joint semantic仍 `[B,16,192]`，代表两个端点各8个disease tokens，不是16个memory token。

图像与语义共用τ、临床区间条件，各自噪声张量不同。同一轨迹样本的两个分支始终在同一次coupled field评估/积分中通信。

### 11.2 Native U-Net层表（不要改变已保存权重布局）

| 阶段 | 旧有模块/尺寸 | 新改动 |
|---|---|---|
| Input | Conv3D48→128,k3,p1；`[B,128,8,32,32]` | 保留 |
| Down0 | FiLM ResBlock128→128 ×2 | 保留 |
| Downsample0 | Conv128→128,k3,s2；`[B,128,4,16,16]` | 保留原布局，不写成新256卷积 |
| Down1 | ResBlock128→256，再256→256 | 后接Bridge@256 |
| Downsample1 | Conv256→256,k3,s2；`[B,256,2,8,8]` | 保留 |
| Down2 | ResBlock256→384，再384→384 | 后接Bridge@384-pre-mid |
| Mid | 原mid1/Swin×2/cross-attn/mid2 | 后接Bridge@384-mid |
| Up1 | resize+skip256，concat640→ResBlock→256，再256→256 | 保留 |
| Up0 | resize+skip128，concat384→ResBlock→128，再128→128 | 保留 |
| Output | GroupNorm→SiLU→Conv128→48,k3 | 保留 |

FiLM的每个block保持 `GN → SiLU → Conv3D → GN → scale/shift → SiLU → Conv3D + skip`。source/target/time/治疗信息通过路由条件进入所有FiLM层，不能只到最后分类头。

### 11.3 三处桥与六层语义block的执行次序

```text
image input→down0→down1 得 H256
semantic embedding→block0→block1 得 S2
(H256,S2) = Bridge256(H256,S2)
保存更新后的256 skip，并以更新后的H256继续downsample

image down2 得 H384
semantic block2→block3 得 S4
(H384,S4) = Bridge384_pre(H384,S4)

image原mid模块 得 Hmid
semantic block4→block5 得 S6
(Hmid,S6) = Bridge384_mid(Hmid,S6)

image decoder→V_Z；semantic LN+Linear192→192→V_S
```

**只在完整 `encode()` 后读取三个旧特征并加bridge是不够的。** 早期bridge更新必须真正流向后续downsample、skip和decoder；否则仍然不是多尺度联合动力学。

为此新增显式 `encode_level0/1/2/middle/decode` 或纯函数 `forward_coupled`。禁止用线程不安全forward hooks缓存激活。checkpoint recompute必须同输入同输出。

### 11.4 BidirectionalBridge逐层

图像C∈{256,384}：Conv1×1×1 C→192 → flatten空间 → LN192。语义S16：LN192。

`Δimage_tokens = MHA(Q=image, K=S, V=S)`；`ΔS=MHA(Q=S,K=image,V=image)`，均6头，每头32。输出image delta reshape→Conv1×1×1 192→C，分别乘可训练tanh gate回加。

双向增量均基于进入bridge时的原始pair计算，避免依赖更新次序。此处可以零gate初始化，但投影/attention不能同时全部零。每处独立gate，记录激活幅度和梯度。

256尺度1024空间token可接受为初版上限；如果更多，使用明确可逆坐标的pool供语义读取，并让image residual回到原网格。不能把全局向量repeat成有空间细节的feature。

### 11.5 SemanticTransitionBlock，6层

每层d=192，6头，FFN ratio4。顺序为：AdaLN-Zero self-attention → cross-attention到M16/clinical context → cross-attention到interval action tokens → FFN。

四个子操作各3个modulation向量（shift/scale/gate），`Linear192→12×192`；旧模型为9向量时需要显式迁移，不能把1536/2304等尺寸不匹配用strict=False跳过。

图像跨流interaction放在第2/4/6层后的Bridge中，不再每层额外加第三套同义image CA，以免重复不清。

modulation输入由τ_embedding + routed interval global构成，τ不使用临床日数替代。语义输入Linear192→192，加two-endpoint role、8-slot position编码；输出LN→Linear192→192。输出层可零初始化，初始化阶段下层梯度被延迟是可解释的，必须验证一个或多个optimizer step后能打开，不接受永久断梯度。

### 11.6 MONAI后端

当前正式结果使用native。[R2] 先在native完成v2语义，再给MONAI1.5.1提供相同边界的split traversal。保留原MONAI模块与许可证，关闭新桥时和未拆分网络做数值等价检查。

如果未安装MONAI，报告skipped，不能把native测试冒充MONAI通过。不得因追求最新版本擅自升级MONAI造成旧权重/数值行为变化；升级必须单独审计。

---

## 12. 统一pCR头：共享参数，不混淆可用信息

### 12.1 新 `SharedPCRReadout` 逐层设计

保留训练折拟合的clinical logistic prior，但替换“observed小MLP + future专属末层”的阶段依赖读出逻辑；新头以同一套query/Transformer/output适配所有前缀。

输入token：`[PCR_QUERY] + M16 + 当前合法clinical/plan summaries + 明确允许的trajectory tokens`。

trajectory每阶段S8保持8个槽，加stage/time embedding、observed/predicted provenance、相邻有效状态差编码。历史真实prefix和未来模拟suffix分别有mask，不能借 `observed_slots` 大小猜测哪张是真实观察。

三层Pre-LN Transformer，d192、6头、FFN192→768→192、分类dropout0.1；PCR_QUERY输出经LN→Linear192→1。加同一个clinical prior得到logit。不得为T0/T1/T2/T3建立四个末层，也不因阶段改变重拟合prior。

对于 `read_current()`：不传未发生真实扫描，也不需要为了query偷偷推进future。对于 `read_augmented()`：传生成完整suffix；仍复用完全相同的头。可给generated tokens加可训练小幅gate，但不是新增一个不同的分类器。

### 12.2 风险结果字段

```python
class PCRQueryResult:
    logit_per_sample: Tensor     # [B,K]
    probability: Tensor          # [B] = mean(sigmoid(logit))
    interpretation: str          # observed_landmark / forecasted_state / marginal_current
    as_of: float
    observed_stage_ids: tuple[int, ...]
    assumed_plan_version: str | None
```

`ForecastTrace`另返回：`stage_logits [B,K,J]`、`stage_probs [B,J]`、`stage_ids`、`stage_valid_mask`、`origin_stage`、`trajectory_id`、每个阶段的state版本。这里J只指规范预测阶段，不把K×J混成患者batch用于统计。

### 12.3 训练与解释边界

每个真实prefix的概率 `P(Y=1|H_observed,j)` 可以用最终Y监督。模型从T0想象T1后的 `Q(M1_hat)` 只是未来状态的模型读出；不能在主评价中标成“获得T1之后AUC”。

主生成损失保持 `BCE(mean_k p_k, Y)`，不是 `mean_k BCE(p_k,Y)`。逐生成阶段的边缘化loss可作为低权重选项，但不能要求所有模拟分支各自确定地匹配同一个Y，也不能以模拟阶段概率越来越极端作为好模型证明。

在模型自洽、未来信息分布和计划固定等条件下，未来条件概率的期望应与当前边际概率相容。不要强制每一条模拟风险等于当前风险；也不要强制每个阶段单调提高/降低。

---
## 13. 五个公共接口：严格行为合同

下面是目标 API 伪代码，不是声称这些函数现在已经存在。

```python
belief0 = model.initialize(observed_prefix, mode="observed")

trace01 = model.advance(
    belief0, interval=interval_01, noise_ledger=ledger, samples=K
)
prior1 = trace01.final_state

posterior1 = model.observe(
    prior1, observation=real_t1_event
)

risk1 = model.query_pcr(posterior1, mode="observed_landmark")

future = model.forecast(
    posterior1, output_stages=(2, 3),
    plan=known_plan, noise_ledger=ledger, samples=K
)
risk_augmented = model.query_pcr(
    posterior1, mode="marginal_current", future_trace=future
)
```

### 13.1 `initialize(observed_prefix)`

只接受 input-only 的真实前缀。只有T0时直接用ObservationSeedPool初始化。存在多次真实观察时，使用相同的顺序更新算法replay，不得一次性双向编码完整病历再挑某个前缀输出。

没有真实动态推演 prior 可用时，同化模块允许 `prior_is_model_prediction=false`，使用此前合法M作为历史背景，并明确gap；初始化从多个已观察visit产生的状态，不能谎称经历了已采样的预测轨迹。

训练/推理需选择并固定replay协议。要求online与replay一致的测试，必须用同一协议和相同noise ledger；不要比较一个跳过advance、一个执行advance的不同算法而期待逐bit相同。

### 13.2 `advance(belief, interval)`

R1中只前进一个规范区间，默认不接受0→3直接跳过。验证 `src_stage==belief.stage`，目标为下一规范阶段，known_at和区间control合法。

计算 `ConditionBundle(M_current, sourceClinical, activeIntervalActions,src,dst)`；冻结本次条件用于所有τ评估。采样并积分双流得到 `(Z_pred,S_pred)`。重编码生成Z供grounding和dense/anatomy使用。

以 `[A_pred4, S_pred8, G_pred4]` 构建预测记忆seed，结合先前history通过同一EventHistory更新得到M^-。生成的S不被直接替换成E(Z)来掩盖双流不一致；只有 `readout_source=reencode` 消融才改用后者。

建立`predicted`模型状态事件，保存parent version和sample_id，但不追加真实evidence。返回新状态，不原地修改调用者对象。初始化K=1可在这里fork到K，后续保持K条对应的parent，不能每阶段重新随机混洗粒子身份。已有K条状态时，默认只能按同样K继续；若需要每个posterior样本再采L条未来，必须显式保留`[B,K,L,...]`及嵌套平均，不能悄悄重复/丢弃粒子使权重改变。

### 13.3 `observe(prior, observation)`

仅同化到达当前或可合法推进到的观察时间。若输入已经是T3预测状态，后来想处理真实T1，必须从该trace保存的T1 prior或源前缀重放，不能让T3预测状态反向参与T1更新。

真实MRI一次编码后按K广播为观察条件，对每个prior memory分别同化；`Z_anchor`和`S_anchor`更新为真实图像/teacher坐标。真实观察减少不确定性是需要验证的现象，不强制让K个memory完全相同，也不虚构likelihood粒子重加权。

活动model-state log中同时间的预测占位被posterior版本替代/标记superseded；其历史先验可用于审计和创新融合，不在history attention里作为第二次独立MRI重复计证。真实evidence只增加一次。

### 13.4 `forecast()`

在state副本上连续advance，构成规范轨迹；按output_stages筛选返回结果。默认不接收target数组、标签、全患者对象、future_available mask或已知未来实际事件。

每步保存：生成Z/S、E_bar(Z).S、memory、shared pCR样本logit、来源与阶段。保持`internal_full_trace`与`requested_view`分离：用户只要T3也必须从完整规范轨迹计算同一个marginal pCR，不能先丢掉T1/T2再交给分类器。终点额外给出`pcr_marginal`，保留完整trajectory维度；不要把 `[B,K,J]` 先flatten再在不同患者之间做概率平均。

### 13.5 `query_pcr()`

`observed_landmark`只读当前真实前缀的posterior memory。`forecasted_state`只能在predicted状态上输出带明确标签的模拟读出。`marginal_current`接收由当前belief合法fork出的完整future_trace，复用shared头读出并概率平均。

查询本身不推进时钟、不追加事件、不消耗动力学RNG、不改变cache或anchor。训练用分类dropout与评估确定性分开；eval重复query结果应一致。

### 13.6 `NoiseLedger` 与请求顺序独立

每个噪声张量由稳定key派生：`(seed, source_physiology_evidence_hash, trajectory_id, transition_src, transition_dst, branch_prefix_hash, component)`。component区分image和semantic。source_physiology_evidence_hash只含当前真实影像/已发生生理相关事件；完整audit evidence log可以包含未来计划，但不能整个拿来给过去动力学重新seed。branch_prefix_hash只包含截至该转移的已执行/当前假设有效controls，不把未来未生效改动带进先前噪声。

**不要用 `output_stages`、request_id、一次全局generator被调用次数、batch内排序作为唯一seed依据。** 单独预测一个患者、调换batch、额外查询T1，都不应使T3样本无理由变化。

生成seed/患者key不是神经网络输入；seed仅控制采样。固定ledger用于回归测试，科研重复实验仍须多个独立seed，不能只挑一个有利seed。

### 13.7 运行时守恒条件

- 不改变VQ身份、标准化和几何坐标；decode前必须反标准化。
- 不把query事件当作新的真实检查。
- 不把flow τ积分步数当作经过的临床治疗次数。
- 真实T1到达后，未来从T1真实anchor继续，而不是从旧T1_hat继续。
- 同一病例不同时点共享转移网络和pCR网络参数；新增stage embedding不等于阶段私有network。

---

## 14. 配对训练：全阶段覆盖，不强制错误的大batch

### 14.1 目标分布先于采样实现

对边类型集合E={01,12,23}，真实配对集合D_e，N_e=|D_e|：

`L_edge = (1/|E_nonempty|) Σ_e (1/N_e) Σ_{patient in D_e} L_FM(patient,e)`。

这是stage-balanced目标。可以通过先均匀抽edge类型、再均匀抽该edge患者实现无偏估计。不能同时逆频率采样又不加说明地再乘1/√N_e，造成双重重加权。

`all_edges`：计算一个患者全部可用edge，按明确目标权重累加，必要时逐edge backward后一次optimizer.step。它用于正确性对照，不要求保存全部edge图。

`balanced_edge_sample`：生产默认；一个optimizer step在累积microbatch中轮询/随机覆盖多个edge类型，记录q和effective_weight。必须有empirical coverage与small-fixture期望等价测试。

对无某阶段MRI的患者，仍允许representation、label和source-only rollout任务；只跳过不存在的配对FM项。

### 14.2 Image与semantic FM

沿用原对称路径，earlier/later真实端点只在监督路径中出现：

`Zτ = concat((1−τ) εZ + τ Zlater, (1−τ) Zearlier + τ ηZ)`。

`Sτ = concat((1−τ) εS + τ Slater, (1−τ) Searlier + τ ηS)`。

目标速度分别为 `concat(Zlater−εZ, ηZ−Zearlier)`、`concat(Slater−εS, ηS−Searlier)`。

`L_FM = MSE(mean_elements(VZ−targetVZ)) + λS MSE(mean_elements(VS−targetVS))`。两个空间分别按维数平均。真实S由固定teacher生成，教师参数不被此目标更新。

条件必须是该transition源时点的合法prefix。不能用终点真实语义或全患者history去生成condition，然后称为source-only。含目标的带噪路径是FM训练本身允许的，不可拿它的endpoint估计直接算主早期pCR。

### 14.3 非相邻真实桥

02/13/03可作为稀疏数据的auxiliary direct transition训练，src/dst与动作区间必须真实、标签明示direct_aux。可用权重如0.25相对于adjacent目标，实际在配置定义并消融。

直接与多步模型应在分布上而不是逐样本噪声轨迹上比较。不要强制“同一个seed的0→3图像严格等于0→1→2→3图像”，两种生成图的随机耦合本来不同。请求一致性测试仅针对**同一规范内部轨迹**。

### 14.4 反向任务

R1主实验默认 `reverse_probability=0.0`。原SymmFlow双端路径可保留，不必为了正向临床任务重新写数学路径。反向retrodiction为独立消融，记录source可用性与方向，不解释为撤销治疗或恢复生理原状。

---

## 15. 真正的source-only多步训练与全部loss

### 15.1 三条训练路径同时存在，但信息边界不同

**真实配对路径：** 用真实earlier→later训练FM，稳定密度学习。

**自由预测路径：** 从随机合法真实origin开始，之后所有中间状态由模型生成，绝不夹带真实目标。即使目标MRI缺失仍推进，只在loss时mask掉该阶段真值。

**观察更新路径：** 在指定合法阶段让真实MRI到达，执行observe，然后继续预测。它用于训练滤波/后验，不得偷偷混入free-rollout路径以缩小误差。

### 15.2 保留现有grounding，并分阶段报告

`L_ground_j = mean_{i,k} ||S_gen(i,k,j) − E_bar(Z_gen(i,k,j)).disease||²`。

固定teacher的weights，生成Z前向不得包 `torch.no_grad()`，否则失去图像梯度。真实target编码可以no_grad。既有函数在F维已经有这类计算，扩展时复用并输出`grounding/T1,T2,T3`，不重复加一份同义loss。[R4]

### 15.3 每阶段Energy Score + 联合轨迹Energy Score

给定样本`X[i,k,j] = E_bar(Z_gen[i,k,j]).disease`，真实目标`Y[i,j] = E_bar(Z_real[i,j]).disease`。

每阶段：

`ES_j = mean_k d(X_kj,Y_j) − [1/(2K(K−1))] Σ_{k≠l} d(X_kj,X_lj)`。

距离先flatten各患者的`[8,192]`并除以sqrt(有效维数)。K>=2；缺失target不参与truth距离。

联合轨迹：保持同一k跨J的对应关系，flatten有效子轨迹再算ES。不能分别重排每阶段样本后仍称为joint trajectory score。**仅有每阶段ES不足以约束跨时间相关性，所以不要把原联合轨迹ES删掉。**

建议：`L_distribution = 0.5 * mean_valid_stages(ES_j) + 0.5 * ES_joint`。有限K U-statistic可略为负，不随意clamp为0。只有一个真实未来样本不妨碍跨患者估计分布评分，但不能单患者就断言校准。

不能对每一个随机sample施加强端点MSE，迫使所有未来都拟合唯一实际观察。低维图像特征ES与独立质量测量可补充；高维原像素ES先作为消融，防距离集中和未配准误差。

### 15.4 真实landmark pCR监督，患者归一化

对每个有label患者i，有合法真实prefix集合L_i：

`L_real_pcr = mean_i [ (1/|L_i|) Σ_{j in L_i} BCE(Q(H_real_i,j),Y_i) ]`。

同一患者多个prefix不加成多个独立样本。不要求T1一定有独立病理label。T3评估必须在pCR标签尚未知的时点，明确手术前。

只有T0信息时不能为了T1的loss在history中读取T1；应单独构造T1真实prefix任务。多prefix并行可用block-causal计算，但要有未来污染测试。

### 15.5 生成完整未来的边缘化pCR，保留原数学目标

每个真实origin i,j生成K条完整剩余轨迹，`logit[i,j,k] = Q(H_real_i,j, generated_suffix_k)`。

`logp_y = logsigmoid((2Y−1)*logit)`；
`L_marginal_i,j = −logsumexp_k(logp_y) + log(K)`。

先对origin在患者内平均，再对患者平均。禁止平均latent或logit后再BCE；禁止默认对每条假想分支分别赋同一个确定标签。

`generated_stage_marginal`：可对模拟prefix的概率均值做低权重辅助，默认关闭或权重0.1，单独命名、单独消融。它不是新的真实T1/T2标签，也不是额外的临床性能测量。

### 15.6 观察校正监督

真实posterior状态直接受到第15.4的label loss及后续可用transition监督。可再加第8.5的小权重重建目标，让同化吸收当前观察，同时保持历史记忆。

没有未来真实MRI时，不能产生虚构的后续配对loss；有label时仍可训练posterior pCR。没有label但有扫描，仍能训练生成与表征。

### 15.7 深层表征和局部对齐

`L_repr` 包含原masked/fused JEPA、多尺度deep JEPA、low-res重建、phase difference、variance/covariance。Pillar全局teacher、dense spatial teacher、segmentation、kinetics等只在真实sidecar存在时启用。

FM feature的REPA/iREPA式空间对齐仍可在真实target监督下使用；对应主干输出空间网格，不允许用global Pillar向量复制成局部真值。[R9]

### 15.8 总损失、初始配置及support

Stage对应启用项，不能一开始把所有loss全开：

`L = λrepr Lrepr + λedge Ledge + λdist Ldistribution + λground Lground + λreal Lreal_pcr + λmarg Lmarginal + λassim Lassim + λspatial Lspatial`。

建议初始归一化权重：imageFM1.0、stateFM1.0、spatial0.05、grounding0.1、distribution0.05、real-prefix pCR1.0、marginal pCR0.1逐步到0.5、assim reconstruction0.1。不是论文最优值；必须记录分项实际数量、幅度和梯度。

每项返回 `{value, support_patients, support_stages, support_edges, valid_mask_count}`。`support=0` 的value用与计算图兼容的zero，但不能报告成模型取得零误差。某一阶段长期无support需警报。

### 15.9 明确禁止的loss

不强制风险单调；不强制肿瘤总会缩小；不把MRI零病灶等同pCR；不拿pCR标签生成未来条件或mask；不强制先验和真实观察后验永远一致；不将模拟图像对自己训练的分类器好看当作医学真实性证明。

---

## 16. 四阶段优化与训练课程

### 16.1 Stage A：统一观察表征与真实prefix读出

数据：所有训练患者的真实T0/T1/T2/T3，按患者/visit归一化。训练ObservationEncoder、masked/deep predictors、ObservationSeedPool、Assimilator、EventHistory、SharedPCRReadout；目标教师只EMA更新。

同次MRI自监督，不做JEPA未来预测。真实prefix顺序更新可先不使用随机生成prior，利用此前M与合法gap上下文同化当前O。标记`prior_is_model_prediction=false`，后续B阶段再适应实际生成prior。

选择表征checkpoint不能只追求某次validation AUC峰值而完全忽视重建/塌陷指标。验证目标事先定义并记录；当前仓库早期checkpoint优于后期，不能机械复用10000步作为必须跑满的标准。[R2]

A结束后复制最佳在线encoder到fixed target。后续B–D统一固定此坐标，避免生成器追着不断变化的状态空间跑。

### 16.2 Stage B：先单边覆盖，再进入自由多步与观察更新

训练：双流、condition router、EventHistory、Assimilator/生成state update。冻结encoder/target/VQ，pCR头不优化或只监控。

B0：前20%最大预算，只用stage-balanced adjacent FM与可选bridge_aux，确认01/12/23均有实际更新。

B1：接下30%，每8个optimizer step加入一次2区间source-only rollout；有实际target的阶段做grounding和ES。观察更新任务开始使用模型prior而不是仅真实prefix。

B2：后50%，加入最长3区间rollout；逐步从两步过渡三步，不把所有真实中间扫描喂给free路径。grounding全预测阶段有效；ES仅可用truth有效。

比例和rollout频率是起始课程，可基于预声明验证规则调整。最终checkpoint必须经过目标长度训练，不能拿只见过一跳的checkpoint直接宣称多阶段完成。

### 16.3 Stage C：冻结世界模型，适配统一读出

冻结VQ、E、M更新、conditions、image/semantic flow；只更新SharedPCRReadout的Transformer/output和允许的读出gate。持续混合真实prefix、生成full-suffix边缘化、真实posterior样本。

可以缓存生成轨迹加速本阶段，但缓存必须包含generator/encoder/normalizer/plan/schema/seed哈希。旧缓存不能跨模型版本混用。真实数据的分类强基线仍单独保留。

### 16.4 Stage D：低学习率联合，不无约束追pCR

优化SharedPCR、condition/history/assimilation、semantic dynamics、新多尺度桥；图像默认仅middle+decoder。冻结VQ和teacher/encoder早期主干。

每个step保留真实配对FM项；source-only rollout参与pCR与分布目标，比例上限由算力/验证预声明。不要仅为改善classification抛弃真实生成监督。

参数组起始LR：shared pCR=1e-5，new history/assim/semantic=1e-5，image middle/decoder=3e-6，image encoder=0。A/B默认1e-4但使用weight decay、clip与warmup。不是效果保证。

### 16.5 单个联合iteration的精确顺序

```python
# 下面是目标流程，函数需由Codex实现并测试。
optimizer.zero_grad(set_to_none=True)

# ① 同一总目标的配对边分量；可逐microbatch反传，避免存全部图。
for edge_batch, normalized_weight in edge_microbatches:
    prefix, edge_truth = dataset.make_edge_task(edge_batch)
    edge_loss = losses.edge_fm(model, prefix, edge_truth)
    (normalized_weight * edge_loss).backward()

# ② 从合法真实origin开始，完全不使用future MRI/label做条件。
prefix, future_targets, pcr_target = dataset.make_rollout_task(patient_batch)
state = model.initialize(prefix)
trace = model.forecast(state, canonical_remaining_stages, ledger=ledger)
loss_gen = losses.grounding(trace)
loss_gen += losses.stage_and_joint_energy(trace, future_targets)
loss_gen += losses.marginal_pcr(state, trace, pcr_target)
loss_gen.backward()  # 不能detach每个生成中间状态来伪装端到端

# ③ 当前真实观察到达；这是独立posterior训练路径，不混入②。
posterior_loss = losses.observed_prefix_and_assimilation(model, patient_batch)
posterior_loss.backward()

clip_grad_norm_(trainable_parameters, max_norm)
optimizer.step()
```

上述三部分的patient/edge归一化权重必须定义清楚；真实实现若用loss分支不同抽样，不把每类sample数不一致当作隐形loss权重。不要对一个已经backward释放的graph再次使用不带retain/recompute的引用。

### 16.6 梯度保真与内存

K×J×Heun_steps条链式图很昂贵。顺序循环K并不会自动消除反向保存成本。生产首配 `batch_size=1, accumulation=4, K=2`；不能沿用单跳batch4并保证5090可放下。

20步Heun每区间40次field评估，3区间K2约240次，仅计field前向；还包含6层语义、三桥及重编码。新完整模型必须实测峰值显存/耗时。建议以microbatch和non-reentrant checkpoint为主，不悄悄detach。

`truncated_bptt`只作为显式消融配置；启用时日志/论文不能称为全轨迹端到端反传。2步smoke只验证工程；从20步预训练直接改2步不等价MeanFlow/蒸馏。

---

## 17. 配置规范（新schema；不得直接粘到旧配置加载器）

完整起始配置另附 `multistage_v2.proposed.yaml`。Codex先在dataclass注册所有字段、类型/范围和互斥检查，再生成可执行的正式配置。`proposed`代表设计输入，不代表当前仓库已经可以运行。

关键默认值：`time_basis=stage_index`；`canonical_stages=[0,1,2,3]`；`transition_mode=sequential`；history6、semantic6、assim4、PCR3、memory16、d192；桥接层256/384pre/384mid；query只读；action_dim从数据推断，当前必须为0。

必须校验：

- stage_index模式不得接受小数天数或未注册阶段。
- R2关闭时对任意日数查询给清晰unsupported提示，不自动映射到最近阶段。
- `output_stages`是规范网格子集，不能改变内部path。
- dimension能被heads整除，memory/semantic槽数与endpoints正确。
- `use_known_full_plan_in_dynamics=false`，interval actions按source可见性过滤。
- K>=2用于Energy Score；K=1只可做推理/单样本消融，不偷偷用假K补齐loss。
- 非商业依赖和预训练来源记录随配置入checkpoint，不因“公开”忽略许可。

## 18. 模型迁移与旧结果兼容

### 18.1 schema不可静默升级

旧 `ResponseWorldModel` 继续接受v1；新 `MultistageResponseWorldModel` 接受v2。旧checkpoint在v1 replay数值必须保持；旧eval字段不改成新stage结果。

旧数据升级只产生新文件，保留原manifest digest。新数据hash更改后不能直接`--resume`；可显式`--warm-start`，重新拟合/核验训练统计并记录来源暴露。

### 18.2 可迁移部分

VQ权重：完全冻结、同SHA与数字合同。

ObservationEncoder主干：新deep heads不改变原键时可完整迁移；若训练折改变，不从旧checkpoint偷带测试集统计。

Native图像U-Net：保留所有旧卷积/ResBlock参数；新bridge零gate初始时应尽量数值兼容旧image路径。condition router改变后模型整体不再数学等价，需要单独给`legacy_conditioner`的迁移测试与新router适配阶段。

semantic4→6：旧4层映射到新0..3，4..5新初始化；从9组调制扩为12组时，按self/history/FFN语义映射，把新action分支初始化为无扰动，写迁移表，不能简单按flatten位置截断。旧token_position与teacherS坐标可沿用。

history3→6：输入query数和事件合同改变，默认只迁移维度/功能同义的注意力/FFN；新memory/query重新训练，报告哪些权重未用。

pCR：旧observed/future双读出与新shared-head接口不同，默认不直接迁移末层；可作为teacher/baseline，不能只改input_dim就承诺等价。

### 18.3 迁移产物

`migration_report.json`至少包含source SHA、数据暴露声明、每个module的loaded/missing/new/shape-mismatch、normalization处理、bridge init、语义层映射、许可证、验证结果。所有不匹配要被明确处理或报错，禁止无报告的`strict=False`。

## 19. 验证与正式实验

### 19.1 两条不同的评价轨迹

**open-loop**：只用真实T0，预测T1/T2/T3，始终不读未来。报告每阶段图像/latent指标和T0起点的marginal pCR。

**online/prequential**：T0→T1 prior；记录prior，再真实T1观察更新；从T1 posterior预测T2；以此类推。每条预测只在数据到达前产生，结果保留生成时的as_of。

不能把online路径中的真实T1/T2补进open-loop后还声称T0-only性能。可同时画prior/posterior概率曲线，但模型模拟状态与真实证据更新必须不同图例。

### 19.2 指标与分组

pCR：AUROC、AUPRC、NLL、Brier、校准分箱和患者聚类bootstrap；主表分别按明确的observed_stage_ids/origin_stage报告。不把多个related landmark当独立case计算置信区间。

生成：T1/T2/T3阶段分项、生成深度、观察校正前后、joint trajectory distribution、语义/图像不一致、独立解码评估。ROI/affine不保证解剖对齐，因此局部像素指标需有质量mask与解释。

联合读出：真实prefix-only、同prefix+生成未来、同prefix+真实未来（仅有未来信息的参照，非公平早期baseline）。必须看生成是否真正贡献而非仅临床先验。

### 19.3 必须有的消融

A：现有T0→T3单跳复现。

B：仅把训练切成纵向并扩展逐阶段输出，不加新history/bridge/assim，隔离“数据覆盖”的贡献。

C：B + 持久状态与observe。

D：C + condition routing/block-causal history。

E：D + 多尺度双流桥/六层语义。

F：E + deep JEPA（若采用）及分布/同化目标完整版本。

另比较后编码`S=E(Z)`与独立语义双流；no_future_readout；无动作数据时不能伪造treatment-conditioned ablation。shared和stage-specific heads对比可作研究实验，但生产目标始终shared。

### 19.4 选模

A选representation稳定性和预先定义的真实prefix指标；B选各阶段平衡的真实生成/分布指标；C/D以预定义的真实landmark macro NLL/Brier为主，同时对独立生成质量退化设置预先记录的guard。

不要直接把不同量纲、不同F长度的raw FM MSE任意加到pCR NLL后宣称可跨所有实验公平选模。记录各项标度/权重；任何调整在查看test前确定。

当前102人是开发验证，反复选模不能改名成独立测试。重新划分已参与历史开发的患者也不能声称“从未接触”。报告上游VQ、ROI定位器、Pillar等预训练暴露情况。[R2]

### 19.5 训练日志最低字段

`loss/<term>/Tj`；`support/patients,label,edge01,edge12,edge23,bridge02,...`；每edge采样概率/权重；`rollout_depth`；`real_prefix_stage_set`；`K`；Heun步数；`future_mask`计数；`grad_norm/image,semantic,history,assim,pcr`；每bridge gate均值/梯度；prior/observed/marginal NLL；生成图像独立score；cache/codec/manifest hash；参数版本；精度/设备/显存峰值。

空support不报告成loss改善。模拟结果不能混入真实MRI质量标签。

---

## 20. 实施顺序与提交粒度

### PR1：先把纵向数据与正确评估跑通，不改主干

创建v2患者manifest与samplers，canonical网格，保留v1；统计edge覆盖、prefix分布和患者划分。基于现有循环建立multi-interval baseline，输出每阶段但区分risk语义。完成target隔离、missing mask、query-grid测试。

### PR2：状态生命周期与shared query

新增PatientBelief、initialize/advance/observe/fork/query，ObservationSeedPool/Assimilator以及统一读出。先以现有history骨干接通接口，完成真实观察校正与旧预测分支失效测试。

### PR3：因果历史与条件路由

加入6层block-causal history、分离计划与当前state、ConditionResampler。验证未来action/观察/查询不会改变过去physiology，支持action_dim0。先full-prefix，后cache parity。

### PR4：多尺度coupled velocity

重构native层边界并插三桥，扩六层semantic。增加zero-gate legacy parity、graph梯度、全尺寸forward/backward检查。MONAI作为同语义可选后端单独验证。

### PR5：loss、课程与deep supervision

多edge目标、阶段+joint ES、source-only rollout、真实prefix BCE、marginal BCE、同化监督；A→D独立参数冻结表；edge采样无偏性测试。deep JEPA需能通过ablation关闭。

### PR6：迁移、命令行、实验报告

迁移器、resume、独立预测、在线状态存档、安全请求、stage指标、源代码许可证清单。输出实际tests与preflight日志，不把未跑的GPU/真实患者训练列成完成。

每个PR保持原测试可执行，若因v2语义导致旧测试不适用，保留v1测试并另写v2，不删除测试来获得绿色结果。不可一次加入所有模块后只做synthetic scalar loss检查。

---

## 21. 验收测试清单（必须逐项有结果）

以下是需由Codex新增并运行的测试目标，不是本次已经通过的结果。

### 数据与可用性

T01 `test_patient_split_disjoint`：同患者跨split被拒绝。

T02 `test_stats_train_only`：任意修改validation/test像素不改变训练统计。

T03 `test_known_at_filter`：晚于as_of的临床/治疗字段不可见，终点标签不能进入输入。

T04 `test_four_masks_not_conflated`：真实未来缺失时仍生成，其target loss为无support而不是阴性。

T05 `test_nonadjacent_pair_tagging`：T0/T2配对只计02 aux，不冒充01/12。

T06 `test_stage_ids_not_observation_count`：{T0,T2}与{T0,T1}报告不同prefix。

T07 `test_action_dim_zero`：临床4/action0，A/B/C/D loss和gradient均finite。

T08 `test_actual_future_scan_date_rejected`：没有known-at证据的未来日数不能自动作为T0查询条件。

### 轨迹与请求

T09 `test_single_T3_query_still_three_transitions`：默认只请求T3也调用01/12/23。

T10 `test_output_grid_invariance`：相同ledger，请求[3]和[1,2,3]的T3逐样本及当前marginal pCR均相同。

T11 `test_query_is_read_only`：重复query不改state hash、clock、history、anchor、dynamics RNG。

T12 `test_batch_order_invariant_noise`：调换患者batch顺序/单例推理不改对应预测。

T13 `test_trajectory_identity_preserved`：每个k的T2使用自己T1，不跨患者/样本串接；更改已有K时必须显式嵌套或报错，不能悄悄改变样本权重。

T14 `test_forecast_never_loads_targets`：移走未来MRI文件后只用input request仍可预测。

T15 `test_poisoned_target_no_inference_change`：改label/target/aux不影响相同input和ledger预测。

T16 `test_no_skip_on_missing_MRI`：T1真值缺失不会跳过模拟T1，也不制造T1真实loss。

### 观察更新

T17 `test_initialize_depends_on_MRI`：非退化MRI→M0依赖及梯度。

T18 `test_observe_anchors_real_image`：T1到达后anchor等于真实Z1，后续源图不再是Z1_hat。

T19 `test_observe_preserves_history`：同样T1、不同合法历史时memory可以不同，而S_anchor保持真实teacher坐标。

T20 `test_observation_idempotence`：相同event重复提交不二次计证，同ID不同payload报冲突。

T21 `test_new_evidence_invalidates_future_branch`：旧T2/T3不能进入新T1后验路径。

T22 `test_online_replay_parity`：同一协议/ledger的在线与replay一致。

T23 `test_missing_observation_identity`：无有效新观察mask时同化不改state。

T24 `test_prior_not_equal_posterior_enforced`：真实不同MRI可改变pCR；无强制prior/posterior equality/monotonic约束。

### 因果与时间

T25 `test_future_real_observation_cannot_change_past_state`：拼入未来真实event后重算，之前输出不变。

T26 `test_future_plan_not_affect_earlier_physiology`：仅改2→3治疗，01生成latent不变；最终policy-conditioned pCR可合法变化。

T27 `test_event_block_mask`：同event内允许注意，future event不允许；padding不污染。

T28 `test_time_axes_separate`：临床阶段/日数变化不改变τ定义；τ变化不改临床时间字段。

T29 `test_stage_mode_rejects_arbitrary_days`：R2未开，14.5天等请求不得伪映射。

T30 `test_full_prefix_cache_parity`：cache与全重算，包含观察替换后的结果一致。

### 网络与梯度

T31 `test_backbone_shapes`：生产latent三尺度形状/48输出/16语义joint token正确。

T32 `test_zero_bridge_legacy_image_parity`：legacy condition下关闭新桥，重构前后U-Net输出一致。

T33 `test_early_bridge_affects_deeper_layers`：Bridge256确实改变down2/decoder，不是死支路。

T34 `test_image_and_semantic_receive_pcr_gradient`：打开/训练门控后，marginal BCE可达两路；记录不是只到readout。

T35 `test_multihop_gradient_to_first_transition`：终点loss可到T1生成/参数，无隐藏detach。

T36 `test_frozen_teacher_input_gradient`：teacher weights无grad，但E_bar(Z_gen)对Z_gen有grad。

T37 `test_checkpoint_recompute_parity`：checkpoint on/off前向及gradient一致。

T38 `test_ODE_velocity_deterministic`：同输入/condition，train mode的velocity不因dropout随机变化。

T39 `test_monai_split_parity_optional`：已安装锁定版本时验证；未安装明确skipped。

T40 `test_same_head_all_landmarks`：所有stage与prior/posterior使用同一head参数对象；不存在stage-private classifier。

### loss与抽样

T41 `test_marginal_BCE_probability_not_logit_mean`：数值符合mean(sigmoid)，极端logit有限。

T42 `test_missing_label_zero_gradient`：missing label不当0，pCR对应grad为0。

T43 `test_energy_patient_isolation`：ES不混患者；sample重排不改，跨时点不一致重排可能改变joint ES。

T44 `test_stage_energy_not_replacing_joint_energy`：配置两项有各自support和权重，分别计算。

T45 `test_edge_sampler_expected_objective`：小fixture穷举all-edges与采样估计在容差内一致。

T46 `test_no_double_stage_reweighting`：inverse频率采样/权重组合按目标公式验算。

T47 `test_free_rollout_teacher_target_separation`：运行轨迹与监督完全分开，可audit哪些tensor来自future。

T48 `test_assimilation_training_gradients`：有真实更新时assim层非零grad；不是只训练分类头。

T49 `test_no_forced_monotonicity`：loss代码无隐藏时序风险或体积单调惩罚。

T50 `test_deep_JEPA_same_visit`：teacher与context来自同一visit，future visit不会作为A阶段context。

### 工程与报告

T51 `test_v1_regression_unchanged`：旧模型和旧数据入口仍可复现兼容测试。

T52 `test_checkpoint_migration_report`：每个新/未加载/尺寸不符参数均有报告。

T53 `test_resume_step_boundary_exact`：可确定环境下中断/不间断的参数和RNG一致；CUDA非确定算子另说明容差。

T54 `test_independent_request_schema`：预测请求拒绝label/target/patient-data-dict等额外字段。

T55 `test_prefix_metrics_and_patient_bootstrap`：不同prefix单独统计，聚类抽样按患者。

T56 `test_codec_identity_and_denormalize`：错误codec SHA拒绝；decode使用原VQ坐标。

T57 `test_all_configs_validate`：所有正式/消融YAML有定义，不含未注册字段。

T58 `test_real_size_preflight`：在实际允许的环境执行B及D真实形状前向/反向，记录峰值；做不了时明确未运行。

T59 `test_three_step_training_smoke`：缩小空间但相同模块类，训练确实覆盖三段、每段loss与shared-head输出。

T60 `test_report_claims_match_executed_tests`：生成机器可读tested/skipped/not-run；不得把历史39/41通过数冒充新模型结果。

## 22. 交付清单与推荐命令

Codex最终应返回：全部修改文件、`CHANGELOG_MULTISTAGE_V2.md`、`IMPLEMENTATION_REPORT.md`、`source_manifest.json`、`migration_report.json`示例、数据审计脚本、正式与smoke配置、测试报告、分阶段命令和限制。

以下命令是**需要实现的目标CLI**，不是当前v1可直接执行的命令：

```bash
# 只创建新数据索引，不覆盖旧实验
python scripts/prepare_existing_ispy2.py \
  --manifest /path/to/source_v2_manifest.json \
  --latent-root /path/to/existing_latents \
  --codec /path/to/existing_vq.pt \
  --output /path/to/ispy2_multistage_v2 \
  --export-patient-trajectories

python joint.py audit-v2 \
  --manifest /path/to/ispy2_multistage_v2/patients.json \
  --output /path/to/audit_v2.json

python -m pytest -q tests --junitxml=reports/pytest_multistage_v2.xml
python joint.py smoke-v2 --config configs/multistage_smoke_v2.yaml \
  --output /path/to/new_smoke_v2

# 使用者核验数据、预算和算力后再运行
python joint.py train-v2 --stage all \
  --config configs/ispy2_multistage_native_v2.yaml \
  --manifest /path/to/ispy2_multistage_v2/patients.json \
  --output /path/to/new_multistage_run

python joint.py forecast-v2 --state /path/to/T0_state.pt \
  --output-stages 1 2 3 --samples 8 --steps 20 \
  --output /path/to/T0_openloop_trace.npz

python joint.py observe-v2 --state /path/to/T1_prior_state.pt \
  --observation /path/to/available_T1_observation.json \
  --output /path/to/T1_posterior_state.pt

python joint.py query-pcr-v2 --state /path/to/T1_posterior_state.pt \
  --mode observed_landmark --output /path/to/T1_pcr.json
```

患者状态存档使用weights-only可读的tensor/primitive结构，不能在未信任对象上用unsafe pickle fallback。不要把真实病历或完整权重提交GitHub；仅提交源码、匿名schema、聚合日志与用户授权的资料。

---

## 23. R2：真实临床时间与MRI之间事件更新（默认关闭）

本节防止把R1的阶段连续更新错误宣称为“任意时间都能可靠模拟”。当前数据没有核验的分段治疗时间，R2只能先实现接口/显式报错，不能在论文中列为已经验证的能力。[R2]

### 23.1 启用前提

有真实acquisition/报告available_at；实际与计划治疗事件区分；疾病/结局窗口与查询时点一致；覆盖一定日数间隔训练/验证。无法审计未来实际日数时继续stage_index。

只有几个MRI仍可尝试连续时间归纳模型，但inter-event时间轨迹不可完全由稀疏端点识别，必须报告假设和区间内泛化验证，不能靠漂亮连续动画证明。

### 23.2 新增独立于flow τ的state drift

可选 `ClinicalTimeDrift`：M16，d192，4层时间/active-control条件Transformer；每层SA6头、active-control CA、FFN768，输出`dM/dt`。真实临床t用单独输入和solver，绝不能把SymmFlow的τ中间点当临床t状态。

`t`依赖向量场与piecewise controls定义在绝对时间上。无事件时沿同一drift推进；事件到来做跳跃更新。采用固定或可控误差积分，并记录临床积分器和生成积分器两个独立配置。

这个drift是新增研究模块，与现有双流的S坐标并不自动一致。需以真实随访teacher状态、后续FM和真实prefix pCR监督，定义M→condition的接口，并验证直达与细分推进的数值/分布兼容；不能无监督漂移后直接输出可信MRI。

### 23.3 no-image事件不能伪造MRI

新增 `observe_event()`：化验或已发生治疗事件通过其专属tokens更新M和active control，不更新last-real-image timestamp，不假装新MRI已获取。要在该时刻生成影像，需要明确调用image query分支，生成图仍标predicted。

`query_pcr(t)`从authoritative状态fork，在副本上推进到t后读出；不能修改authoritative history。增加中间只读query不应改变晚些时候的结果。

### 23.4 额外验收

Δt=0无事件严格identity；固定control下直达/细分state drift数值误差受solver tolerance控制；查询顺序不改状态；未来事件不能影响过去；延迟报告仅在available_at后进入；不输出超过训练时间域的无提示高置信预测。

若使用SDE，则必须有一致Brownian噪声树/耦合；每次查询重新抽独立噪声不满足同一患者连续path语义。R1不必实现SDE。

---

## 24. 公开代码参考：应当复用什么，不能声称什么

### 24.1 优先阅读顺序

先当前仓库，再Dreamer的observe/imagine分离，再V-JEPA-AC block-causal结构，再V-JEPA2.1 deep supervision，然后DiT/REPA/Brain-WM与同场景TDN。不要只看摘要或下载权重而不读接口。

| 来源 | 本任务应参考的内容 | 不照搬/不宣称 |
|---|---|---|
| 用户当前仓库 [R0–R5] | 三相encoder、VQ、对称路径、完整U-Net、现有loss/训练 | 不把已有多阶段grounding说成新发明 |
| DreamerV3，Nature 2025 [R10] | `initial/observe/imagine`的先验与真实观察边界、序列训练 | 不搬强化学习actor/reward；本版不声称RSSM精确变分后验 |
| V-JEPA2-AC官方 [R6] | action/state/visual按时间块交错、block-causal attention | 不用7维机器人动作，不照搬24×1024规模 |
| V-JEPA2.1，2026预印本 [R7] | 深层/密集表征监督、可见/遮挡token区分 | 3D乳腺适配不是它的官方医学复现 |
| DiT [R11] | AdaLN-Zero、独立调制与残差初始化 | 结构借鉴不等于加载其自然图像权重 |
| REPA/iREPA [R9] | 局部特征对齐、卷积投影与空间归一化 | 不把全局Pillar repeat成局部teacher |
| Brain-WM，2026预印本 [R8] | 生成与临床任务受控共享、任务专属路径 | 不整套搬Show-o/Qwen，不宣称治疗因果识别 |
| Longitudinal Temporal Pillar，2026预印本 [R12] | 同场景serial MRI/pCR基线、query头与clinical prior | 原1152→低维权重不直接等价新192维空间 |
| MONAI [R13] | 实际完整diffusion U-Net和数值/shape测试 | 缺依赖不能说已验证；维持锁定版本 |
| MeWM，ICCV2025 [R14] | 医学治疗条件生成与结局评估背景 | “生成后预测结局”本身不是本项目独有创新 |

### 24.2 许可证与来源清单

每个复制/适配文件记录：`upstream_repository, upstream_commit, upstream_path, upstream_license, local_path, copied_or_adapted, changes, verification`。新增源标头保留原copyright。版本字段写commit，不把Git blob SHA冒充commit。

来源带非商业许可时按其范围保留；新MIT代码不能替整个混合包重新授权。需要商业或其他特殊用途时由项目负责人核对，代码agent不能自动假定许可兼容。

参考Dreamer数学接口但没有复制源文件时，记录`conceptual_reference`；使用官方MONAI作为dependency则记`runtime_dependency`；从论文启发的自写3D适配则记`task_specific_adaptation`。这样不会把“读过”包装成“完整复现”。

### 24.3 文献与固定源地址

以下链接便于Codex取源；本规范不复制论文的大段文字。涉及2026工作均按本次可核验的预印本/官方代码状态表述，不擅自标成已被顶会录用。

**[R0] 当前目标仓库与本次基线。**
https://github.com/cgznb/medical-world-models/tree/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint

**[R1] `model.py`，forecast/memory/状态行为。**
https://github.com/cgznb/medical-world-models/blob/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint/src/responsewm/model.py

**[R2] 当前I-SPY2适配、正式实验与限制记录。**
https://github.com/cgznb/medical-world-models/blob/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint/docs/ISPY2_EXISTING_DATA_ZH.md

**[R3] 当前数据导出脚本。**
https://github.com/cgznb/medical-world-models/blob/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint/scripts/prepare_existing_ispy2.py

**[R4] 当前loss实现。**
https://github.com/cgznb/medical-world-models/blob/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint/src/responsewm/losses.py

**[R5] 当前encoder与backbones。**
https://github.com/cgznb/medical-world-models/blob/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint/src/responsewm/legacy/encoder.py
https://github.com/cgznb/medical-world-models/blob/841458cb8e06e131af2a2248dc853f8476b2e0d0/breast_world_joint/src/responsewm/backbones.py

**[R6] V-JEPA2-AC官方action-conditioned predictor。**
https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/ac_predictor.py
https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/utils/modules.py
应读 `VisionTransformerPredictorAC`、`build_action_block_causal_attention_mask`、`ACBlock`。

**[R7] V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning。**
https://arxiv.org/abs/2603.14482
https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py
https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/predictor.py

**[R8] Brain-WM: Brain Glioblastoma World Model。**
https://arxiv.org/abs/2603.07562
https://github.com/thibault-wch/Brain-GBM-world-model
重点读 `models/modeling_showo2_qwen2_5.py`、任务分支和条件处理。main需在实施时锁定commit；不要把本文的Task-specific多尺度bridge声称为Brain-WM原实现。

**[R9] REPA / iREPA。**
https://github.com/sihyun-yu/REPA/blob/main/loss.py
https://arxiv.org/abs/2512.10794
https://github.com/End2End-Diffusion/iREPA
实施时锁commit，借鉴空间对齐而非复制其自然图像数据协议。

**[R10] DreamerV3 / Mastering diverse control tasks through world models。**
https://www.nature.com/articles/s41586-025-08744-2
https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py
重点读 `initial`、`observe/_observe`、`imagine`、`loss`。该代码为JAX/ninjax，本项目PyTorch适配只借用合理的状态接口与监督分离，不假装直接移植全部训练系统。

**[R11] DiT官方。**
https://github.com/facebookresearch/DiT/blob/main/models.py
https://arxiv.org/abs/2212.09748
实施时锁commit及许可，关注DiTBlock/FinalLayer与初始化。

**[R12] Longitudinal 3D Foundation Modeling for Neoadjuvant Breast Cancer Response Prediction from Serial DCE-MRI。**
https://arxiv.org/abs/2608.09991
https://github.com/omarftt/longitudinal_temporal_pillar/blob/main/src/tdn.py
本规范仅用作同场景强基线和query/prior参考，需锁定其实际代码版本再运行。

**[R13] MONAI1.5.1 diffusion U-Net。**
https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/diffusion_model_unet.py

**[R14] Medical World Model，ICCV2025。**
https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Medical_World_Model_ICCV_2025_paper.html
该来源用于研究定位；本任务不要求重新搬整个MeWM工作流。

---

## 25. 最终给Codex的完成标准

实现后的系统必须能够用一段真实可执行的测试脚本展示以下过程，而非只有算法示意：

1. 输入真实T0及当时可用信息，得到持久状态M0与同一head的当前pCR。
2. 在无真实未来输入的条件下，按规范阶段生成T1−、T2−、T3−，每步有Z/S/M和明确解释的pCR读出。
3. 将真实T1作为新到达观察同化为T1+，旧T2/T3分支失效，从真实Z1重新预测后续。
4. 同一head对T0、T1+、T2+、T3术前真实prefix预测同一个最终pCR；模拟prefix结果另列。
5. 缺少T1/T2真值时仍可推演；不对缺失数据做虚构监督；完整轨迹评价不混入真实中间影像。
6. 请求只返回T3或返回T1/T2/T3，在相同ledger下不改变对应T3轨迹。
7. 真实图像与语义的联合生成被约束；分类梯度能到多段生成链，但不牺牲监督隔离和teacher坐标。
8. 实际测试、未执行项、数据限制、许可证和迁移范围全部有可检查产物。

**论文可以主张的应是经过实验支持的连续状态更新、影像约束的随机轨迹和动态结局读出；不能把层数、文件数、引用论文数量或一个能够运行的demo本身当作有效性证明。**

— 规范结束 —
