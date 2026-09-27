# 三相 MRI Observation-v3：A 只学当前表征，B 才预测未来

本机缩小版三相适配及启动说明见 [REGISTERED_ROI32_LOCAL.md](docs/REGISTERED_ROI32_LOCAL.md)。实际训练使用 `data/registered_roi32/` 和独立的 `runs/registered_roi32_20260920/`；该适配保留原模型宽度和深度，新增按显存实测选择 batch 的启动流程。

这是 **新版三相 A/B 工作流的完整可独立运行源码包**，并提供对 `cgznb/symm-fm` 原三相训练入口的增量接入。不是原仓库所有历史任务的全量镜像，不包含 MRI、患者缓存、VQ 权重或训练好的新模型。本次环境无法下载原仓库的二进制 archive；因此没有把独立工作流冒充“已克隆全部上游仓库”。`scripts/assemble_full_repository.py` 可在你本地原仓库上生成完整“上游源码＋本版”的合并 ZIP。

本版替换之前 world-v2 的职责划分：**A 中没有 FuturePredictor、未来 disease tokens、治疗/临床编码器、pCR 监督、纵向 anatomy consistency 或 C 阶段。** A 的 JEPA predictor 只做同一访视内空间/观测变换任务，B 推理不会加载这个 predictor。

## 先运行 CPU 自检

建议单独建立 Python 3.11/3.12 环境；本次实际测试环境是 Python 3.13.5、PyTorch 2.10.0+cpu。旧仓库历史依赖可能另有限制，两套环境不需要混装。

```bash
cd symm-fm-threephase-observation-v3
python -m pip install -e '.[test]'
python world.py smoke --output /tmp/symm_observation_v3_smoke
python -m pytest -q
```

`smoke` 使用真正的缩小版 Transformer 和多尺度 3D U-Net，执行 A、B、同访视特征提取、未来生成。它会在生成前删除测试病例的真实未来 latent。合成数据和临时权重仅生成在你指定的目录，源码 ZIP 不带这些权重。

当前环境的实际报告见 `reports/VALIDATION.json`、`reports/pytest.xml` 与 `reports/smoke.json`。CPU 路径通过不代表已经验证真实患者效果、完整分辨率 GPU 显存或 MONAI 后端；MONAI 未安装时对应测试显式跳过。

## 直接接入已有三相缓存

`--root` 是已有运行目录，其中应有 `admitted_inventory.json` 和 `latents/`。该目录由原三相 prepare 流程产生，不是仓库根目录。

```bash
python world.py convert-legacy \
  --root /path/to/three_phase_run \
  --codec-checkpoint /path/to/matching/shared/codec.pt \
  --output /path/to/new_observation_manifests

python world.py audit \
  --visits /path/to/new_observation_manifests/visits.json \
  --pairs /path/to/new_observation_manifests/pairs.json

# 保持原仓库同类型的 MONAI velocity 后端；安装与显卡匹配的 PyTorch 后执行。
python -m pip install 'monai==1.5.1'

python world.py train \
  --stage all \
  --config configs/cache_only.yaml \
  --visits /path/to/new_observation_manifests/visits.json \
  --pairs /path/to/new_observation_manifests/pairs.json \
  --output /path/to/new_observation_run
```

也可以分开执行：

```bash
python world.py train --stage A --config configs/cache_only.yaml \
  --visits /path/to/visits.json --output /path/to/new_run

python world.py train --stage B --config configs/cache_only.yaml \
  --pairs /path/to/pairs.json --a-checkpoint /path/to/new_run/A/best.pt \
  --output /path/to/new_run
```

恢复使用相同命令加 `--resume`。`--stop-after N` 按 optimizer updates 控制临时停止；改变配置、数据文件内容、A checkpoint 或资产身份时不能继续旧 optimizer 状态。缓存版不需要原始 MRI 即可运行 A/B；真实 DCE 信号约束和 image-domain modeling 默认关闭，不用 latent 变换伪装 MRI 采集变化。

`configs/cache_native.yaml` 是完整包内 3D U-Net 后端，不依赖 MONAI，但与原 MONAI U-Net **不是同一个网络**。`configs/cache_only.yaml` 的 MONAI 宽度、注意力层级沿用已核对的原配置 `[64,128,256,256] / [False,False,True,True]`。它不是加载原生成器权重；新增 A/B 网络均需训练。

## 提取当前特征或生成未来 latent

输入 NPY 必须是**原始 continuous VQ latent** `[24,D,H,W]`，不是按旧 FM statistics 标准化后的值。转换器读取的原缓存满足这个坐标约定。裸 NPY 需显式给出匹配 `codec_id`；推荐用 `encode-source` 生成带身份的 NPZ。

```bash
python world.py extract --checkpoint /path/to/run/A/best.pt \
  --source /path/to/current_latent.npy --codec-id 'sha256:ACTUAL_VQ_FILE_DIGEST' \
  --output /path/to/current_features.npz --device cuda

python world.py sample --checkpoint /path/to/run/B/best.pt \
  --source /path/to/current_latent.npy --codec-id 'sha256:ACTUAL_VQ_FILE_DIGEST' \
  --conditions /path/to/source_conditions.json \
  --output /path/to/predicted_future.npz --device cuda --samples 4
```

`codec_id` 从生成的 `visits.json` 原样复制，不填写字面量 `ACTUAL_VQ_FILE_DIGEST`。再加 `--codec-checkpoint /path/to/codec.pt` 可同时输出解码后的三相 MRI 数组。没有合法源几何时不自动制造 NIfTI origin/direction。生成文件的 `latent` 含样本轴 `[K,24,D,H,W]`；再次单样本推演前需显式选择样本，不能把 K 维误当输入 channel。

`source_conditions.json` 例如：

```json
{
  "stage_i": "T0", "stage_j": "T1",
  "treatment_arm": "your_verified_arm",
  "hr_status": "1", "her2_status": "0", "age": 50,
  "interval_verified": true, "delta_days": 21,
  "action_segments": []
}
```

未知类别有独立 embedding；没有核验的间隔设 `interval_verified:false`，不会把脱敏日期当成生物学时间。禁止把 pCR、术后病理、RCB、实际未来观测塞入条件。计划类事件必须声明 `known_at_source:true`，这只是输入契约，不会自动证明你的原始临床数据已经满足可用性。

## 有原始三相影像时的进阶模式

为每个 visit 提供 `image_path`、真实三相覆盖 mask、空间对齐核验和 `image_normalization` 后：

```bash
python world.py prepare-sidecars --visits /path/to/image_backed_visits.json \
  --codec-checkpoint /path/to/codec.pt --output /path/to/domain_bank --device cuda

python world.py train --stage A --config configs/image_backed.yaml \
  --visits /path/to/domain_bank/visits.json --output /path/to/image_backed_run
```

该命令对同访视 MRI 做已知 gain/offset/noise/blur，再通过匹配的冻结 VQ 编码；先检查原图经过该 VQ 是否能重现原缓存。它生成同访视 augmentation bank 和 measured signal-difference targets，不读取未来访视。默认不做 gamma 变换，不把这些增强称作经过验证的 scanner simulator。

数据格式、坐标、标签和 sidecar 说明见 `docs/DATA_CONTRACT.md`。

## 不替换原训练器，只给原三相 B 增加当前状态条件

```bash
python scripts/install_into_repo.py --repo /path/to/symm-fm --enable-original-b

python scripts/write_original_b_config.py \
  --base-config /path/to/symm-fm/workflows/first_post_three_phase/configs/three_phase_symmflow_all_pairs_t0_v2.yaml \
  --a-checkpoint /path/to/new_run/A/best.pt \
  --output-root /path/to/brand_new_original_b_run \
  --output-config /path/to/three_phase_with_observation.yaml
```

然后用原仓库入口和新配置执行。安装器保留原 `ThreePhaseSymmFlow` 和原配置行为，仅在新配置有 `observation_checkpoint` 时选择新的模型。它不改你的患者文件、旧 checkpoint 和已运行实验。桥接保留原 MONAI 网络、原临床 schema、原速度 objective、原 controller/evaluation，适合做严格的原 baseline 对照。细节见 `docs/UPSTREAM_INTEGRATION.md`。

需要整份原仓库＋修改版源码：

```bash
python scripts/assemble_full_repository.py \
  --local-upstream /path/to/symm-fm \
  --output /path/to/symm-fm-observation-v3-full.zip
```

## 模块与边界

A：共享 phase patch embed → 同位置跨相位 Transformer → 12 层 3D-position ViT；6 层 predictor；4 层 observation readout。两 crop 来自同一次访视，local 在 masked tokens 上监督，global 按三维相对位置监督另一 crop 全部有效 tokens。两 crop 不具有临床时间含义。原 VQ 仍冻结。

B：冻结的 A encoder → 当前 patient tokens，与 B 的临床条件拼接 → 原 VQ 空间中的双分支 SymmFlow。仅 B 输入纵向 pair 和治疗/时间条件；A 的 SSL predictor 与 readout 不出现在 B 中。默认只用速度 MSE；可选 frozen semantic endpoint loss 为独立消融。

A 可用训练患者的每个访视作为独立样本，也可只用 baseline；不跨访视做自监督配对。现有 all-pairs inventory 不包含的 singleton 需要从你的访视级清单补入，转换器不能凭空恢复。

“原 VQ+语义 encoder”不等于重新训练 VQ encoder；VQ 已丢失的信息不能保证由下游语义模块恢复。原 ROI 可能含未来定位信息；本版不会把 source-only 文件读取或相同 shape 冒充预处理无泄漏或纵向配准。缓存默认只是 ROI 表观预测，医学/因果有效性均未建立。

完整结构与来源见 `docs/DESIGN_ZH.md`、`docs/SOURCES.md`。许可按文件分开，新代码的 MIT 不覆盖 MeWM-derived codec 的 CC BY-NC 4.0 限制。未打包第三方预训练权重，未声称复现两篇论文的医学结果。
