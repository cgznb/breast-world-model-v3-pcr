> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# 本次交付实际验证记录

日期：2026-09-28。证据对应本 ZIP 内源码，不代表原仓库全部历史测试已重新运行。

## 环境

Python 3.13.5；PyTorch 2.10.0+cpu；CUDA 不可用；MONAI 未安装。完整记录见 `reports/environment.json`。README 中的 Python>=3.10/PyTorch>=2.5 是软件依赖声明，不是对这些所有组合逐一验证过的承诺。

## 自动化测试

执行：

```bash
python -m pytest -q --junitxml=reports/pytest.xml
```

本次结果：**39 passed, 1 skipped**。原始输出见 `reports/pytest_stdout.txt`，逐项结果见 `reports/pytest.xml`。

覆盖概率均值而非 logit 均值、Jensen 对照、极端 logit 数值稳定、缺失标签零梯度、按患者的 Energy Score、空间教师维度、正反向 ODE、监督污染不变性、未来文件不读取、padding 不影响预测、语义/图像梯度、多区间反传、固定教师输入梯度、checkpointing 一致性、所有四阶段损失、四阶段真实训练入口与独立预测、VQ 编解码及冻结 codebook、严格 V2 权重迁移、患者划分/可用性、train-only 统计、患者采样 RNG、断点续训参数逐项一致、输入资产变更拒绝、阶段坐标/配对合同、终末观察无未来、无未来读出噪声不变性和全部 YAML 校验。

**跳过项：MONAI 1.5.1 split-forward 与原 MONAI forward 的数值一致性。** 测试文件已提供；本环境没有该依赖，不能声称执行通过。

部分梯度单元测试主动把零初始化残差门控打开，以检验已经激活后的计算图通路。这不等于模型在初始化时已经学会耦合；另有四阶段 optimizer 测试检验从零初始化出发的训练路径。

## 四阶段命令行 smoke

执行：

```bash
python joint.py smoke --output /mnt/data/responsewm_delivery_smoke
```

representation、flow、readout、joint 各执行 2 个 optimizer steps，损失及梯度有限，产生检查点并通过独立 input-only predict。使用8个合成患者、多个 landmark 和缺失未来槽位。

输出 latent `[1,2,3,24,4,8,8]`，状态 `[1,2,3,4,24]`，两条完整轨迹概率 `[1,2]` 和一个边际概率 `[1]`。这是同类模型的缩小配置，不是假单层网络，也不是真实 MRI。日志见 `reports/smoke_stdout.txt` 和 `reports/smoke_report.json`。

## 完整宽度与实际 latent 尺寸

执行：

```bash
python scripts/check_full_shape.py --output reports/full_shape_native_forward.json
```

原生后端，真实生产宽度、192维状态、8个disease tokens，输入 `[1,1,24,8,32,32]`。单条单区间轨迹产生 `[1,1,1,24,8,32,32]` latent 及 `[1,1,1,8,192]` 语义状态，输出有限。模型总参数98,457,388，含固定教师，非全可训练参数数。

该测试使用随机权重、1步Heun以验证全尺寸前向结构；**不是1步采样质量实验，不是全尺寸反向、20步三段轨迹或GPU显存验证**。精确配置与结果见 JSON。

## 未完成/未声称的验证

没有真实患者训练或效能结果；没有真实 VQ/Pillar 私有权重读取；没有实际 CUDA/BF16 或 MONAI 执行；没有独立医院测试、真实配准质量核验、生成 MRI 专家评估、概率校准验证、临床因果效应识别、顶会 benchmark 复现或原仓库全部测试重跑。

V2 迁移测试使用具有相同参数布局的合成 state_dict；严格迁移器将在你提供真实检查点时再次校验，并不会假装所有旧检查点都能读取。完整混合许可和公开源码来源已记录，不授予患者数据/外部模型权重的使用权限。
