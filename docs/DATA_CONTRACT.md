# 输入、监督与时间边界

## 文件组织

训练使用 `responsewm_manifest_v1` JSON，包含 schema、phase_order、clinical_features、action_features、latent_shape、vq_identity、time_basis、shared_grid_verified、cases。`synthetic=true` 只供测试且必须显式允许。每个 case 是一个患者在一个 landmark 的任务；同一患者的所有 case 只能属于同一个 train/val/test split。

每个连续 latent 文件为 `.npy` 的 `[24,D,H,W]` 数组，或 `.npz` 中键名 `latent` 的同形数组。必须是**原始连续 VQ latent**，不是离散 codebook ID、不是已标准化两次的张量，也不是直接 MRI 像素。三相顺序固定为 `pre_aqc0, first_post_aqc1, metadata_late`。当前生产尺寸为 `[24,8,32,32]`，测试使用缩小空间。

`vq_identity` 建议严格写为 `sha256:<VQ权重文件SHA256>`。编码与解码命令均使用这个身份核对匹配 codec。`shared_grid_verified` 与每个 input 的 `source_only_geometry` 是必须真实核验的声明，不是软件推断出的事实。

## 每个 case

```json
{
  "id": "local_case_T0",
  "patient_id": "local_pseudonymous_patient",
  "split": "train",
  "input": {
    "landmark_day": 0,
    "observed": [
      {"latent": "/path/to/T0.npy", "day": 0, "available_at": 0}
    ],
    "clinical": [47, 1, null],
    "clinical_known_at": [0, 0, null],
    "queries": [
      {"day": 1, "known_at": 0, "actions": [1, 0], "actions_known_at": [0, 0]},
      {"day": 2, "known_at": 0, "actions": [0, 1], "actions_known_at": [0, 0]},
      {"day": 3, "known_at": 0, "actions": [0, 1], "actions_known_at": [0, 0]}
    ],
    "source_only_geometry": true
  },
  "target": {
    "pcr": 1,
    "observed_auxiliary": [null],
    "future": [
      {"latent": "/path/to/T1.npy", "day": 1, "anatomy_comparable": false},
      null,
      {"latent": "/path/to/T3.npy", "day": 3, "anatomy_comparable": false}
    ]
  }
}
```

这个例子是 **stage_index**，1/2/3 不是天数。示例数字不对应用户队列，不是治疗编码建议。对于 `calendar_days`，每个字段应改用实际合法日数。默认 landmark 恰好为最后一项已观察 MRI 的 day；本版不支持没有新 MRI 的任意中间日期观察更新。

临床/治疗向量维度与顺序由 manifest 字段名决定，不要求固定 17/8 维。复用旧分类流程时应使用你实际的 `TABULAR_FEATURE_NAMES` 顺序及相同含义；示例字段不能直接冒充原 17 项。类别字段先由训练折确定合法 one-hot 或数值编码，再写成数值向量。本包不把任意整数类别编码假定为连续病理程度。

`actions[j]` 定义为当前计划的“前一个查询/最后观察到第 j 个查询”区间治疗特征。可以编码当前可获知的药物多热、剂量或其他明确含义变量。字段名、单位、是否预定/已执行、缺失规则由研究者写入数据制作记录。本包没有从自由文本自动识别药物、没有调用 LLM、没有自动补全未知周期。

可用值的 known_at 必须不晚于 landmark；未知值为 null，其 mask=false，不是数值零。未来计划改变但在 T0 不知情，不能把最终执行信息填入 T0。查询时点已指定并不意味着该次扫描一定实际完成；若缺失，target 槽为 null，模型仍可预测。

## 缺失与配对

模型张量层允许已观察历史中有空位，使用最大有效索引而非 `count-1` 找最后观察；manifest 使用实际存在的有序列表，由 batch padding 产生空位。未来请求必须为连续前缀，而未来监督 mask 可以有洞，两者不能混用。

例如仅 T0/T3 真实存在而请求 T1/T2/T3：可以监督最终 pCR、最终分布和 T3 状态；但不存在 T1→T2、T2→T3 的相邻真实 FM 配对。应另导出只查询 T3 的 direct-interval case，以便用 T0→T3 做配对生成训练。B 阶段患者采样器只选择具备合法配对的 case；D 阶段没有配对的患者仍可贡献可用的结局监督。不要创建假的中间图像。

不对真实随访日数与查询日数做隐式最近邻匹配。要预测按治疗阶段的图像但不知未来实际日数，使用 `stage_index`。这种模式不支持把输出解释为任意真实天数的精确预测。

## 训练/预测隔离

训练数据加载返回 `(ForecastInput, Supervision)`，后者独立包含真实未来、标签与辅助信息。`model.forecast()` 的签名只接受前者，没有 target、标签、patient_id、teacher forcing 参数。

部署 `responsewm_request_v1` 只包含 schema、特征名、相位、latent_shape、VQ身份、time_basis 和 input。由 `scripts/export_request.py` 导出，它不包含 target 或 patient_id。预测器不会为推理拟合 normalization，也不会去打开未来 latent 文件。

标签默认二元最终 pCR；缺失用 null。软件不决定 pCR 的病理定义。论文中应明确 breast-only/ypT0/is ypN0 等实际结局口径与标签来源，不能混用。

## 辅助 sidecar

每个真实 visit 可有一个 `.npz`，由 `target.observed_auxiliary` 或 `target.future[j].auxiliary` 引用。键名限制为以下各项；没有数组时该项 loss 不执行。

| 键 | 形状（生产） | 含义 |
|---|---|---|
| pillar | `[1152]` | 已冻结、经过原规范预处理的真实 MRI 的全局 Pillar 特征 |
| dense_teacher | `[32,192]` | 与当前 `(2,4,4)` token 网格对应的真实空间特征 |
| segmentation / segmentation_mask | `[1,2,4,4]` / 可广播有效 mask | 已核验对齐的软分割监督；全零病灶合法 |
| kinetics / kinetics_mask | `[3,2,4,4]` / 有效 mask | 三项明确测量的相位关系/指标，必须记录预处理含义 |
| biomarkers / biomarkers_mask | `[4]` / `[4]` | 自己定义并核验的四个可用测量；不是自动生成的生物标志物 |

全局 Pillar 向量不能复制 32 次作为 dense_teacher。监督 mask 的 true 表示有测量/有效覆盖，而不是病灶阳性。不要使用 pCR 标签生成“预测图像应该多小”的分割伪标签。

外部教师编码、空间注册和低分辨率目标制作依赖你的原数据预处理；本包不会静默自动匹配任意教师权重或空间单位。已经导出的原 Pillar 特征可直接写成 sidecar，代码将核对维度和 finite 值。

## 归一化与分折

latent 均值/方差仅用训练折的去重真实 visit 计算，包括训练患者的后续影像；这是合法训练监督，不是推理读取测试未来。clinical/action 数值统计只用训练折已知值。prior logistic regression 只用训练折标签，按患者调整重复 landmark 权重。

manifest 内容和所有 latent/sidecar 的 SHA256 写入 run。test 不参与优化和 checkpoint 选择。不同路径的重复患者、外部预训练接触、源代码中的私有标签等无法仅靠文件路径一致性检测；需要独立审计。

为保证声明与统计一致，当前 evaluate 只接收训练时同一个 manifest 的指定 split。外部独立队列可以通过严格 input-only request 逐例预测后用指标函数评价；本版没有宽松自动合并外部队列的命令。
