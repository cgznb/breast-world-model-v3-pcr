# V3 安装、数据准备与训练运行指南

以下命令在仓库根目录执行，面向 Linux、Python 3.11+。`/path/to/...` 表示使用者本地资产路径。代码包不提供真实患者数据、临床标签、codec 或训练权重。实验规则见 [V3 固定协议](MULTISTAGE_V3_PROTOCOL_ZH.md)。

## 1. 安装与合成流程检查

先安装支持所用 GPU 的 PyTorch/CUDA，再安装项目。V3 正式配置使用 native 后端，无须安装可选 MONAI。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python joint.py smoke-v3 \
  --config configs/multistage_smoke_v3.yaml \
  --output runs/v3_smoke
```

smoke 使用新输出目录，在 CPU 上用缩小网络和合成数据执行四阶段，不代表临床效果。发布验证结果见 [release_validation.json](../reports/release_validation.json)。

## 2. 准备患者轨迹与影像诊断

V3 复用 V2 患者数据 schema。若已有合法的 `patient_trajectories_v2.json`、训练统计和匹配数组，可直接使用。若持有旧 `responsewm_manifest_v1` 纵向清单，可转换为：

```bash
python scripts/prepare_multistage_v2.py \
  --manifest /path/to/longitudinal.json \
  --codec /path/to/codec.pt \
  --output data/trajectories
```

输出 `patient_trajectories_v2.json`、`train_statistics_v2.json` 和审计记录。脚本合并真实访视，保留划分和缺失随访，不复制数组；原数据路径必须可访问，已有同名输出不会被覆盖。它不承担原始 DICOM 的完整预处理。真实三相 latent 应为 `[24,8,32,32]`，具有经过审计的 T0 来源可用网格与匹配 codec。更早的 `symm_world_manifest_v2` 缓存需先通过 `scripts/prepare_existing_ispy2.py` 适配，参数见该脚本 `--help`。

准备 8 名预定完整验证患者的影像子集：

```bash
python scripts/prepare_imaging_eval_v3.py \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --data-root /path/to/audited_cache \
  --output data/imaging_eval \
  --cases 8 --split val
```

`--data-root` 必须包含 `image_manifest.json`、`mask_manifest.json`、原 `manifest.json`、`codec.pt`，以及清单引用的 MRI/latent/ROI/support/伪掩膜数组。脚本核验患者、空间网格和 codec 身份。

主配置 `configs/ispy2_multistage_native_v3.yaml` 的 `protocol.imaging_manifest` 为 `data/imaging_eval/manifest.json`，相对当前工作目录；`protocol.codec_path: codec.pt` 相对该 manifest 所在目录，实际为 `data/imaging_eval/codec.pt`。

没有影像诊断资产时，可在**另一个实验配置**中设 `imaging_manifest: null`、`codec_path: null`、`image_eval_cases: 0`。这样会省略解码影像检查，不能视为完整原协议复现。

## 3. CUDA 预检与 batch

默认 A/B/C/D 物理 batch 为 **160/32/192/32**，验证 batch=32，训练 K=2、验证 K=4，每段 Heun 20 步。新硬件先预检；需要降 batch 时修改正式训练前的配置，之后重新预检。

```bash
python scripts/preflight_multistage_v3.py \
  --config configs/ispy2_multistage_native_v3.yaml \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --statistics data/trajectories/train_statistics_v2.json \
  --output runs/v3_preflight.json
```

此脚本在临时模型上完成四阶段 loss、跨时间反传与 AdamW 更新，记录梯度和显存，不保存训练权重；要求完整真实训练轨迹数至少达到所设最大 batch。

可另检查固定患者评价与真实 MRI 解码路径：

```bash
python scripts/preflight_evaluation_v3.py \
  --config configs/ispy2_multistage_native_v3.yaml \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --statistics data/trajectories/train_statistics_v2.json \
  --output runs/v3_evaluation_preflight
```

这也是未训练模型的工程检查，分数不表示训练结果。

## 4. A→B→C→D 与中断恢复

```bash
python joint.py train-v3 \
  --config configs/ispy2_multistage_native_v3.yaml \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --output runs/v3_primary --stage all
```

恢复使用相同配置、数据、源码与输出目录：

```bash
python joint.py train-v3 \
  --config configs/ispy2_multistage_native_v3.yaml \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --output runs/v3_primary --stage all --resume
```

单阶段名称为 `representation`、`flow`、`readout`、`joint`；后续阶段依赖同一 run 中前一阶段选定的模型。A/B/C/D 最大步数为 200/3000/150/400，允许按协议早停。`last.pt` 用于恢复，`best.pt` 用于阶段交接与评价。

- A：重建/JEPA/状态重建与辅助 pCR，学习率 `3e-5`、pCR 权重 0.10。
- B：marginal pCR 权重为零，真实观测更新分支仍有 pCR 梯度，通过冻结读出进入状态和生成模块。
- C：只更新 pCR 最后一块 Transformer 与输出层，按 T0 full-future marginal NLL 选模；入口 step 0 可保留。
- D：继承 C best，保守联合更新；按同一主指标选模，且生成相对误差不超过 D 入口的 1.10 倍；入口同样可保留。

B/C/D 完成后自动评价所选 `best.pt`，保存 `evaluation_selected.json`；启用影像诊断时还保存 `imaging_selected/`。这些运行结果可能含患者级信息，应保留在私有实验目录。

可给完整 `--stage all` 命令加 `--run-ablations`，执行预定的 A 学习率及 C 更新范围/学习率/残差惩罚诊断；它们不会自动替换主实验。

## 5. 多种子队列

必须提供**同一配置与同一 manifest** 的完整四阶段预检；预检配置只允许种子不同，改变 batch 或其它参数后应重新预检。

```bash
python scripts/launch_multiseed_v3.py \
  --config configs/ispy2_multistage_native_v3.yaml \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --preflight runs/v3_preflight.json \
  --output runs/v3_multiseed \
  --seeds 20261011 20261012 20261013 \
  --workers 2
```

launcher 创建独立种子配置和不可覆盖计划，启动后台 worker。状态见 `queue_status.json`、`queue.log` 与各 `seed_*/run/controller_status.json`。阶段锁允许两个 A 并行，B/C/D 连同评价独占；第一个种子自动执行预设消融。不要同时启动绕过该队列的额外 GPU 训练。

批内按随访可用结构分组，保留缺失处理，允许重复采样患者；物理 batch 不等于独立患者数。队列中断且原进程退出后，重跑同一命令恢复原计划。活跃进程或配置/数据摘要变化会触发拒绝启动。

## 6. 评价 V3 checkpoint

```bash
python joint.py evaluate-v3 \
  --checkpoint runs/v3_primary/joint/best.pt \
  --manifest data/trajectories/patient_trajectories_v2.json \
  --device cuda --split val --bootstrap 1000 \
  --output runs/v3_primary/final_evaluation.json
```

评价从 checkpoint 恢复配置和训练统计，要求 manifest 摘要完全一致。启用影像诊断时，checkpoint 内的资产路径须可访问；相对路径时保持仓库根目录为工作目录。当前 CLI 没有单独覆盖这些影像资产路径的参数。

主任务使用所有 102 名 T0 验证患者；真实历史对照固定在协议中的完整随访子集。临床 prior、真实 MRI 历史、联合生成未来、同轨迹重新编码未来使用预定比较方式。缺失未来真值仅使对应生成评分缺项，不补为真实观测。影像检查包含持久性基线、生成均值、解码 MRI、固定 T0 ROI 和有效伪病灶 ROI；伪掩膜不是专家分割。

V3 CLI 为 `train-v3`、`smoke-v3`、`evaluate-v3`。保留的 `initialize-v2`、`forecast-v2`、`observe-v2`、`query-pcr-v2`、`evaluate-v2` 属于旧 checkpoint 工作流，不应用于加载 V3 权重。数据 schema 复用不代表 checkpoint 兼容。

本次是代码发布，没有附带患者数据或模型权重，也未宣称训练完成。历史报告和未训练预检不能证明 V3 优于 V2 或临床 baseline；源代码核验与发布范围见 [发布说明](RELEASE_V3_20261002.md)。
