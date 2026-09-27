# 原仓库接入与保持 baseline

核对对象：`cgznb/symm-fm` 初始源码提交 `92265d3b1749ae3b686f2089843c49da129fd4d2`。本包包括原 `three_phase_symmflow.py` 的逐字参考副本，其 Git blob SHA 为 `6ca00eb2a2bdfd7039dbda8688a88674fd970179`。它用于审计，不作为独立程序运行。

## 两种使用路径

**独立路径**：本包的 `world.py` 完成manifest转换、A、B、特征提取和推理，不需要导入上游那些历史工作流。原数据预处理、注册、分割和旧 VQ 训练仍由你已有程序完成。MONAI后端真实调用官方模型，但独立B临床编码器和优化配置不逐项复制原实验；不能称作严格等预算baseline复现。

**原入口路径**：先用本包训练A，再安装bridge到原repo。原controller、Dataset、clinical schema、MONAI U-Net、velocity objective、Euler sampler和评价链保持，只增加当前访视的patient tokens。这个路径更适合验证“只加一个预训练表征encoder”带来的贡献。

```bash
python scripts/install_into_repo.py --repo /path/to/symm-fm --enable-original-b
```

新增：

- `workflows/observation_v3/`：独立新工作流全部模块。
- 根目录 `run_observation_v3.py`：A及独立B入口。
- `workflows/first_post_three_phase/mewm_ispy2/three_phase_observation_bridge.py`。

唯一原文件补丁是 `three_phase_all_pairs_training.py` 的 model construction。未设置 `observation_checkpoint` 时仍构造原 `ThreePhaseSymmFlow`；设置时才构造新subclass。补丁匹配原确切语句，发现上游代码变化会拒绝猜测。原文件另保存 `.py.observation_v3_backup`；不会改动或删除旧实验的配置、MRI、cache或checkpoint。

创建新配置：

```bash
python scripts/write_original_b_config.py \
  --base-config /path/to/symm-fm/workflows/first_post_three_phase/configs/three_phase_symmflow_all_pairs_t0_v2.yaml \
  --a-checkpoint /path/to/a_run/A/best.pt \
  --output-root /path/to/new_original_b_run \
  --output-config /path/to/three_phase_observation.yaml
```

继续在原仓库运行：

```bash
cd /path/to/symm-fm
python run.py ispy2 scripts/run_three_phase_symmflow.py \
  --config /path/to/three_phase_observation.yaml --stage all
```

新输出目录会执行原准备/核验逻辑，不自动信任旧cache。保留或调整原 `reuse_preparation_from` 时，必须满足原的几何/资产身份检查；不能直接把旧训练checkpoint拿来resume。

## normalization 和资产身份

bridge接收原Dataset的标准化latent，通过原统计量反标准化，再交给A内部标准化；否则两套mean/std不同会造成隐性分布错误。生成器仍预测原Dataset的VQ坐标，原decode链不变。

A checkpoint 的codec_id要求是实际 `cfg.codec_checkpoint` 的SHA256。任何不匹配直接报错。原repo的 `public()` 会过滤hash命名的字段和长hex字符串，因此bridge将A内容身份以整数byte数组存入 `description.observation_asset.content_identity_bytes`，同时保存size/mtime，使原resume契约仍能发现A资产变化。

bridge检查A曝光患者与原B holdout。原loader没有把valid mask交给模型时，bridge按原latent全覆盖处理，并在model description中记录；它不会凭空认定support或配准经过QA。独立工作流可使用显式valid mask。

## 生成全量合并源码ZIP

```bash
python scripts/assemble_full_repository.py \
  --local-upstream /path/to/symm-fm \
  --output /path/to/symm-fm-observation-v3-full.zip
```

脚本从本地 Git tracked 文件组装，保留其未提交的tracked源码编辑，但不包含Git历史；会排除MRI、weights、CSV等敏感/二进制数据、.env、local paths和密钥。它在临时拷贝中安装bridge，不写入原repo。原仓库若已经安装v3，应使用未安装的干净副本或检查安装状态，脚本不会覆盖已有v3。

本次交付已经实际测试独立A/B完整小型路径、安装模板匹配、backup、拒绝覆盖、启动器、合并ZIP安全过滤，以及bridge代码的编译与原core接口的静态核对。没有实际执行用户私有GPU controller/prepare管线；真实权重和数据测试需在用户环境进行。
