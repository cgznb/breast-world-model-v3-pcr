# 输入契约和从原仓库迁移

## 坐标和 dtype

所有入库 latent 是原 `encode_continuous` 输出 `[24,D,H,W]`：依次为 pre、first-post、metadata-late，各 8 channels。NPY/NPZ 可用 float16/float32，读取转 float32；不得先套旧 FM z-score。NPZ latent 键为 `latent`。训练集 channel mean/std 在 A 内部拟合、随 encoder checkpoint 保存。B 直接复用 A 的坐标统计量。

原缓存标准大小是 `[24,24,64,64]`。A crop 可小于完整 grid；B 的 source/target 形状必须匹配，不做隐式 resample。A 的 patch 必须整除完整输入尺寸和 crop。shape、相位顺序正确不代表完成了三相或纵向配准。

## A visits.json 示例

```json
{
  "schema": "three_phase_visits_v3",
  "phase_order": ["pre_aqc0", "first_post_aqc1", "metadata_late"],
  "latent_channels": 24,
  "codec_id": "sha256:实际VQ文件SHA256",
  "visits": [
    {
      "id": "patient001_T0_canonical",
      "patient_id": "patient001",
      "visit_id": "patient001_T0",
      "stage": "T0",
      "split": "train",
      "latent_path": "/absolute/path/T0.npy",
      "valid_path": "/absolute/path/T0_valid.npy",
      "geometry": {"latent_to_lps": [[0,0,2,0],[0,2,0,0],[8,0,0,0],[0,0,0,1]]},
      "phase_alignment_verified": true,
      "source_available_grid": true,
      "support_assumed": false,
      "phenotype_label": -1,
      "domain_label": -1
    }
  ]
}
```

示例 affine 只是格式演示，不得照抄为真实数据坐标。`latent_to_lps` 将数组索引 `(d,h,w,1)` 映射为 LPS 坐标。也可传原 workflow `geometry` 的 `shape_zyx / spacing_xyz_mm / origin_lps_mm / direction_lps`；实现按 4× VQ 下采样和 cell center 构造代表性 affine，不宣称其等于卷积完整 receptive field 的中心。真实复杂重采样时应显式提供经过验证的 affine。

所有路径可绝对或相对当前 manifest。记录的白名单是 `data.py:VISIT_KEYS`；不会自动把任意字段拿来训练。一个 patient/visit 只能保留一个 A 样本，多个 crop 在训练时从该数组采样。不能把一个 visit 在 all-pairs 里出现的多份 view 当作多个独立患者样本。

`valid_path` 为同 latent lattice 的 `[1,D,H,W]` 或 `[3,D,H,W]`，数值在0–1，1表示有可信采集覆盖。不是肿瘤 mask。无 valid 时显式假设全覆盖，audit 会报告，不称作已核验。A token validity 依据三相交集；默认 patch 平均覆盖≥0.95 才保留。某 crop 需要至少2个有效 patch，以同时产生 visible context 和 masked target。

`stage` 是标识和 B 配对元数据，**不喂给 A encoder/predictor**。pCR、下一访视 latent、未来治疗、病理/生存字段在 A 记录中不允许。

## 当前访视辅助监督

`kinetic_path`：已在 latent lattice 的 `[2,D,H,W]`，键 `kinetics`。必须指定 `kinetic_provenance:"measured_same_visit_shared_normalization"`，来源是同访视的 early-pre / late-early 图像信号差。禁止将 VQ feature differences 或 decoder proxy 填在这个实测字段。

`segmentation_path`：同 lattice `[1,D,H,W]`，键 `segmentation`，值0–1。全零有效；缺失使用无该字段，而不是生成零mask。

`phenotype_label`：当前访视可信分组的整数索引；`-1`代表缺失。配置中声明 `phenotype_definition`、`phenotype_classes`。本包不自动定义 HR/HER2 分组的临床含义，不接收以患者ID或未来pCR冒充当前phenotype。

`domain_label`：可信 scanner/site 类别索引；`-1`缺失。配置必须明确 `domain_definition: scanner` 或 `site`；不要用phase/treatment/visit作技术域。DANN只在确有至少两类train监督时启用。

`image_path`：已按匹配 VQ 预处理的三相 `[3,D_img,H_img,W_img]`，NPZ键 `images`，image dims通常是latent的4倍。还须有：

```json
"image_normalization": {
  "stored": "normalized",
  "shared_across_phases": true,
  "mean": 真实共享均值,
  "std": 真实共享标准差
}
```

这是格式示意，不是合法可复制的JSON数值。不要用分别phase z-score后的图像声称其仍是共同物理信号尺度。普通 cache 训练不读取 image_path；image-backed准备工具才会读取并校验它。

## B pairs.json

```json
{
  "schema": "three_phase_pairs_v3",
  "phase_order": ["pre_aqc0", "first_post_aqc1", "metadata_late"],
  "latent_channels": 24,
  "codec_id": "sha256:实际VQ文件SHA256",
  "pairs": [
    {
      "id": "patient001_T0_T1",
      "patient_id": "patient001", "split": "train",
      "source": {"完整的source visit记录": "与A相同的记录格式"},
      "target": {"完整的target visit记录": "与A相同的记录格式"},
      "conditions": {
        "stage_i": "T0", "stage_j": "T1",
        "interval_verified": false,
        "treatment_arm": "verified_arm", "action_segments": []
      }
    }
  ]
}
```

上面 source/target 需替换成完整 visit object，不是实际可运行的patient数据。转换器会自动生成正确结构。没有配对的 observation 不需要出现在 B。一个 patient 的所有 records 必须同一个split。A训练患者不能进入B验证/测试，A验证患者不能作为B独立测试患者。

允许条件：treatment_arm, hr_status, her2_status, mammaprint, menopausal_status, stage_i, stage_j, age, delta_days, interval_verified, action_segments。每个 action segment 严格包含 drug,start,end,dose,known_at_source；只接收在预测源时点已知的计划。未核验 interval 的 delta_days 被删除，不用0或访视编号假装实际日期。类别词表、数值统计只在 B train 拟合。

## legacy converter 注意事项

转换器读取 `admitted_inventory.json` 的 visits/views/pairs。A按(patient,visit)去重，优先选source_visit_id==visit_id的原 view；B保留对应pair的原source/target view，因此可能和A使用不同crop但使用同一个VQ空间。验证集reference若含images/support，会从support构造保守的4× latent有效覆盖。训练集没有reference不会自动产生“实测kinetics”。

可提供 `--qc /path/to/qc.json`，格式：

```json
{"views": {
  "原始view_id": {
    "phase_alignment_verified": true,
    "source_available_grid": true,
    "valid_path": "相对qc文件的valid.npy",
    "image_path": "相对qc文件的images.npy",
    "image_normalization": {"stored":"normalized","shared_across_phases":true,"mean":0.0,"std":1.0}
  }
}}
```

不要为了通过检查随意把布尔项改成true。若配准/定位不可验证，保留false并将任务界定为ROI表观预测；不得称作真实空间肿瘤生长。source_only生成不读取future文件，但无法逆转原预处理中已经使用未来中心定位的问题。

原inventory不存在的孤立访视不能由转换器恢复。你可从原访视清单补充到独立A manifest；本版Dataset对仅一个visit的患者没有限制。

## augmentation bank

`prepare-sidecars`生成的每条augmentation包含latent_path、4维action、transform_space=image、相同visit_id、base_latent_sha256和匹配codec_id。runtime会校验这些字段和base内容身份。action描述gain-1、offset、noise_std、blur_sigma；没有使用治疗/临床时间。注意本实现是可追溯的影像增强，不是扫描仪仿真模型。
