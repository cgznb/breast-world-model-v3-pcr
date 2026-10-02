# ResponseWM：未来 MRI 与 pCR 的联合随机世界模型

**基于 `cgznb/breast-world-model-v2-pcr` 提交 `11220d38dd076439951d08928335d5fe4fa880b7` 的独立升级工作流。**

这是可执行的研究代码，包含共享三相状态编码器、历史 Transformer、双向交互的图像/语义 SymmFlow、可微多步轨迹、轨迹边缘化 pCR 监督、四阶段训练、独立预测、V2 权重迁移、VQ 接口、数据审计、消融配置与测试。

**交付范围是完整的新联合工作流，不是原仓库约九万行历史代码的镜像。** 原仓库的 Pillar/TDN 分类基线、原始 MRI 预处理和历史脚本不被覆盖。`scripts/assemble_with_v2.py` 可在本地把原仓库已跟踪源码与本升级包合并成 ZIP。

本包不含患者数据、真实划分、VQ/Pillar 权重或已训练的联合模型。没有进行真实患者训练、CUDA/BF16 训练或临床有效性验证。代码完整性不等于 CVPR 录用水平；研究贡献、性能和泛化必须通过严格实验确定。实际测试证据见 `reports/` 与 `docs/VALIDATION.md`。

## 1. 先运行工程验证

已有 GPU PyTorch 环境时不要为了这个包随意更换 CUDA wheel。

```bash
python -m pip install -e '.[test]'
python scripts/validate_environment.py
python -m pytest -q
python joint.py smoke --output /absolute/path/to/new_synthetic_smoke
```

`smoke` 使用真正的缩小版 3D Swin/ConvNeXt/U-Net/Transformer，不把模型替换成单层 mock。它执行 representation → flow → readout → joint，随后进行输入独立的预测和合成数据评估。合成数据上的 AUROC 等数字不是医学实验结果。

MONAI 后端另装：

```bash
python -m pip install 'monai==1.5.1'
python -m pytest -q tests/test_model.py -k monai
```

`joint_monai.yaml` 调用实际 MONAI 模块。`joint_native.yaml` 使用显式的完整原生 3D U-Net。缺少 MONAI 不会静默切换后端。本次交付环境只能执行 CPU 原生后端，MONAI 对照测试为跳过项。

## 2. 模型对应关系

```text
已观察三相 MRI → 冻结 VQ → 已观察空间 latent 序列
                                   ↓
                 共享 3D Swin/ConvNeXt 状态编码器
                                   ↓
              历史 Transformer + 当时已知临床/治疗计划
                                   ↓
                                  H_t
                   ┌───────────────┴────────────────┐
              48 通道图像 SymmFlow             16 个语义 joint tokens
              完整 3D 条件 U-Net              4 层 adaLN-Zero Transformer
                   └────── 瓶颈双向交互 ×2 ──────────┘
                                   ↓
                   每次采样得到 (未来 Z, 未来 S)
                        ↓                    ↓
                  冻结 VQ 解码器       下一步历史更新 / pCR 时序头
                        ↓                    ↓
                     未来 MRI          每条完整轨迹一个概率
                                             ↓
                                       对 K 条轨迹概率取均值
```

生产配置：VQ latent `[B,24,8,32,32]`；dense tokens `[B,32,192]`；anatomy `[B,4,192]`；disease `[B,8,192]`；图像通道 `[128,256,384]`；历史 3 层、语义 4 层、pCR 3 层。编码器有多尺度空间主干，图像骨干有真实残差/跳跃连接，交互发生在速度场内部，不只是末端拼接。

**同一次检查的三相增强影像与纵向 T0–T3 不是同一个维度。** 24 通道仍然是三组 8 通道连续 VQ 坐标，不重新命名为“解剖/疾病通道”。

## 3. 接入现有缓存

推荐复用你已有的 V2 manifest。新工作流需要一份按预测时点核验的信息表，不能从最终病历自动推断“当时已知”的治疗和时间。

```bash
python joint.py convert-v2 \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --landmarks /absolute/path/to/landmark_spec.json \
  --output /absolute/path/to/joint_manifest.json

python joint.py audit \
  --manifest /absolute/path/to/joint_manifest.json \
  --scan-arrays --output /absolute/path/to/joint_audit.json
```

格式见 `docs/DATA_CONTRACT.md`、`configs/landmark_spec.stage_index.example.json`。示例中的路径、VQ SHA、特征与核验布尔值必须替换为真实记录，不能把未核验项改为 true 以消除报错。

**日期容易泄漏，因此提供两种互斥模式。**

`calendar_days` 用真实日数，但未来查询日期必须在当前时点已指定；与配对目标应一致。本版不进行时间容差匹配或插值。没有预先确定的未来实际扫描日期时，不应从随访影像读取该日期作为 T0 输入。

`stage_index` 用 T0=0、T1=1、T2=2、T3=3，采用独立的阶段坐标编码，不冒充实际时间间隔。它适合先实现按治疗阶段预测。临床时间与无量纲流时间 τ 始终分离。

同一患者可导出多个合法 landmark，共用最终 pCR 标签；不同阶段是在预测**同一最终结局**，不是假定每次 MRI 都有即时 pCR 病理标签。缺失未来 MRI 使用 null，缺失 pCR 使用 null，均不会作为阴性监督。

## 4. 训练

```bash
# 有前瞻性阶段计划、没有合法未来实际日数时，使用阶段坐标配置。
python joint.py train \
  --config configs/joint_stage_index_monai.yaml \
  --manifest /absolute/path/to/joint_manifest.json \
  --output /absolute/path/to/new_joint_run --stage all
```

四阶段逻辑：

| 阶段 | 更新参数 | 监督 |
|---|---|---|
| representation | 状态编码器、EMA 目标、历史/条件编码、pCR、辅助头 | 掩码表征、低分辨率重建、相位差、可用教师特征/测量、真实轨迹与观察历史 pCR |
| flow | 双流速度场、历史/条件编码 | 配对图像和语义 SymmFlow、局部空间对齐；状态坐标固定 |
| readout | pCR 时序头 | 冻结生成器产生的 source-only 轨迹 + 真实轨迹；概率边缘化监督 |
| joint | 双流、历史/条件编码、pCR；默认只开放图像中后段 | 配对生成目标 + 可微 rollout 边缘化 pCR + 图像/状态一致性 + Energy Score |

单独恢复：

```bash
python joint.py train --config configs/joint_stage_index_monai.yaml \
  --manifest /absolute/path/to/joint_manifest.json \
  --output /absolute/path/to/new_joint_run --stage joint --resume
```

保存 optimizer、学习率步数、PyTorch/Python/NumPy/CUDA RNG、患者采样器状态。配置、manifest 或缓存内容改变时拒绝沿用同一个 run。验证集选择 best；test 不用于选 checkpoint。

### 从你现有 V2 初始化

```bash
python joint.py train --config configs/joint_monai.yaml \
  --manifest /absolute/path/to/joint_manifest.json \
  --output /absolute/path/to/new_joint_run --stage representation \
  --init-v2 /absolute/path/to/v2_flow_or_representation.pt
```

编码器与图像骨干必须维度/后端一致。迁移器严格核对所有键与形状，记录未迁移部分，不使用悄悄忽略错误的 partial load。新临床条件、历史状态、随机语义和 pCR 头需要训练；旧外部 TDN 的 1152 维权重不能直接当作新 192 维读出。

当前折训练统计会保留，随后重新校准表征；不是声称换折后迁移出的生成分布立刻等价。V2 检查点如果曾接触现在的测试患者，即使本次 manifest 患者不重叠，也不能宣称端到端独立测试。

### 显存与采样

默认训练 K=2、20 步 Heun；评估默认 K=8、20 步。小样本调试使用 2 步不代表生产模型可直接少步采样。20 步 Heun 每区间需要 40 次速度场评估，三段、K=2 约为 240 次；可微图的成本不会因为顺序循环而消失。

建议先导出 T0→T3 的单区间 case，检查收敛与显存，再使用三段轨迹。不要把未来 mask 直接改成零来假装做了单区间训练；查询、目标和条件必须一致。全尺寸 CUDA/BF16 显存预算尚未实测，没有保证某张显卡一定能运行默认配置。MeanFlow/少步蒸馏没有在本包中冒名实现。

## 5. 独立预测与 MRI 解码

```bash
python scripts/export_request.py \
  --manifest /absolute/path/to/joint_manifest.json --case-id an_audited_case \
  --output /absolute/path/to/request.json

python joint.py predict \
  --checkpoint /absolute/path/to/new_joint_run/joint/best.pt \
  --request /absolute/path/to/request.json \
  --device cuda --samples 8 --steps 20 \
  --codec /absolute/path/to/matching_vq.pt \
  --output /absolute/path/to/prediction.npz
```

不传 `--codec` 时输出 latent、状态和概率。传入时校验 codec SHA，并先反标准化再解码，拒绝不匹配的 VQ。推理 request 不接受标签、患者标识、真实未来路径、完整训练 manifest 或 teacher-forcing 开关。

输出：`latent [1,K,F,24,D,H,W]`（原始连续 VQ 坐标）、`state/image_state [1,K,F,8,192]`、`trajectory_probabilities [1,K]`、`pcr_probability [1]`；可选 `images [1,K,F,3,4D,4H,4W]`。同一个 k 跨 F 维对应同一完整轨迹。

已有处理后的单次三相 MRI 也可以通过 `joint.py encode` 建立连续 latent。它要求匹配 VQ 的强度预处理与 source-only 几何记录；不提供未经核验的原始 DICOM 配准/肿瘤裁剪捷径。

## 6. 评估与消融

```bash
python joint.py evaluate \
  --checkpoint /absolute/path/to/new_joint_run/joint/best.pt \
  --manifest /absolute/path/to/joint_manifest.json --split test \
  --device cuda --samples 16 --steps 20 --bootstrap 1000 \
  --output /absolute/path/to/test_report.json
```

输出 AUROC、AUPRC、NLL、Brier、分箱校准和患者聚类 bootstrap。按已观察次数分组的指标用于 landmark 对比，混合不同起点的 aggregate 不能替代阶段性主结果。若一个患者某阶段缺扫描，“两个观察”可能是 T0+T2 而非 T0+T1，必须按你的研究协议进一步区分。

`configs/ablations/` 包含去掉双向交互、去掉未来读出、重编码状态、逐轨迹 BCE、无 grounding、无 Energy Score、无空间对齐、冻结图像等配置。使用相同患者划分和公开预算。去掉未来读出的配置仍训练生成分支以保持训练信息对照，并非省算力实现；它也不能替代全部独立的强判别基线。

`export_for_legacy_pcr.py` 将已经解码的预测按未来时间点输出为原 `pcr/run.py encode` 接口要求的 `[K,3,D,H,W]`，用于独立 Pillar/TDN 检查。不会凭空生成原分类器权重、真实 T0 特征或物理间距。完整外部分类基线仍在你的原始 `pcr/` 目录。

## 7. 关键科学约束

主 pCR 路径只从历史 + 可用条件 + 噪声开始；真实未来只进入训练目标。图像/语义固定教师保持输入梯度，区别于把前向包进 `no_grad()`。pCR 对完整轨迹概率均值计算稳定 Bernoulli NLL，不先平均图像、特征或 logit。Energy Score 按患者计算；不要求每条随机未来都等于唯一真实未来。生成标签不得作为条件。

这些限制降低常见工程泄漏风险，**不证明真实病历的可用性声明、ROI 几何、校准或治疗因果效应已经成立**。生成器与分类器仍可能在固定特征空间中投机，需要独立影像模型和真实测量验证。pCR 不被硬编码成 MRI 病灶体积为零。

更多内容：`docs/ALGORITHM.md`、`docs/IMPLEMENTATION_MAP.md`、`docs/SOURCES.md`、`docs/DATA_CONTRACT.md`、`docs/EXPERIMENTS.md`。许可证是混合范围，VQ 与 DiT 适配保留非商业限制，见 `LICENSE`、`licenses/`。
