# 修改说明：Observation-only Stage A + Longitudinal SymmFlow Stage B

版本 0.3.0。下述是代码中实际实现的职责和操作，不是已经证实有效的医学结论。

## 1. 本次删除了什么

从上一版方案删除 A 的 future-state predictor、future disease tokens、longitudinal future loss、跨访视 anatomy 稳定性约束、固定 anatomy/disease 分解、A 的 pCR head。取消 C 阶段。本版不支持用旧 world-v2 checkpoint “恢复训练”；网络、目标和数据契约均不同。

**A 唯一的数据实体是 visit；B 才有 source-target visit pair。** A 的 manifest、Dataset、训练入口、checkpoint 和日志都独立。A 的 JSON 记录禁止未知字段，不接受治疗、下一访视路径、pCR、预后等作为特征输入。一个只有 T0 的患者也可训练 A。

A 中的 predictor 是对“同次检查内未观察区域”的辅助特征预测器，不是下一临床阶段的预测器。它不在 B 的 state_dict 或生成路径中。B 的源条件完全由当前真实 source 和声明为源时点可用的临床条件计算。

## 2. 文献支持与本项目适配的区别

CheXWorld（用户提供 arXiv:2504.13820v1）在 §4.1–4.4、Figure 1(d) 中由同一 radiograph 采样两个 crop，做 local anatomical structure、global anatomical layout、domain variation 联合学习。§B.1 的 predictor 为 6 层、384 维，teacher 为 EMA。其 global task 根据 crop 相对位置预测全部 target patches，而 local task 只在被遮挡位置计损失。

本版保留同访视两 crop、四条监督和已知相对位置/外观参数的逻辑，但从二维 X-ray 扩展到三相 MRI 的三维 VQ latent。共享非重叠 patch embedding、相位融合、crop 配置、384 维 encoder、teacher 最近 K 层平均、四参数 MRI augmentation 都是本项目适配，**不是原文三维实现或直接可用的 X-ray 预训练权重**。缓存 VQ 的邻近 latent 感受野已可能重叠，不保证 raw MRI 层面的严格遮挡独立性。

AlphaCell（用户提供 2026.03.02.709176v1）Base Model 的重建驱动训练、semantic projector、ArcFace 和 DANN 见 Methods §2.2–2.3；其 Flow Model 单独学习干预动力学。本版只借鉴“保留观测信息、分开表征与动力学职责”的设计原则。没有复制其 RNA tokenization、Bi-Mamba/MoE、32×128 cell state、十亿参数 decoder 或 OT 数据配对。它是未同行评审的预印本；其细胞实验不能直接证明本 MRI 适配有效。没有声称核验了作者完整公开代码。

## 3. 保留两个坐标空间

原 continuous VQ latent 为 `z_raw ∈ R^[24,D,H,W]`，三相每相 8 channels。A 使用自己的 train-only channel mean/std，得到 `z`。新的 `E_obs(z)` 输出 `[N,384]` 当前访视 tokens。它不是新的 24-channel VQ codebook，也不能直接被旧 decoder 解码。

B 的生成变量仍是标准化的原 VQ latent；输出恢复为 `z_raw` 后通过**匹配的原 frozen VQ quantizer+decoder**得到三相 MRI。旧 VQ 不参与 A/B 优化，因此不使旧连续 latent 缓存失效。

原仓库桥接还要处理另一组 normalization：原 Dataset 的 standardized source 先用**原统计量反标准化**，再由 A 内部使用 A 统计量。不能把“旧标准化 latent”再次当原始 latent 标准化。

## 4. A 编码器和 auxiliary readout

生产配置：

1. 输入 `[B,24,D,H,W]` 按 phase 拆分为 `[B*3,8,D,H,W]`。
2. 共享 `Conv3d(8,128,kernel_size=stride=(2,4,4))`，得到各相 patch features。kernel=stride，无新的跨 patch 卷积感受野。
3. 按 visible indices 同时删除某个空间位置的**三个相位**。在任何新 spatial attention 之前生效，不计算未遵守 mask 的三相差分旁路。
4. 每位置三相加 phase embedding，通过 2 层 pre-norm cross-phase Transformer，4 heads，4×FFN。
5. 拼接三相 feature，LayerNorm+Linear 到 384 维。加真实值可输入的 3D sinusoidal position encoding。
6. 12 层 spatial Transformer，6 heads，4×FFN，残差和 LayerNorm。空间可变 token 数，padding mask 明确，不把 padded slots 当影像。
7. 可返回有效 token 序列、mask、坐标、原 patch indices、grid 和 pooled feature。teacher 默认平均最近 4 个 blocks 的输出再归一化，是可改配置的适配。

训练 crop 默认 latent `[16,32,32]`，patch grid `[8,8,8]`，共 512 个三相联合 tokens。标准原始缓存 `[24,24,64,64]` 全量推理的 grid 为 `[12,16,16]`，共 3072 tokens；不默认为 512，也不把其全部喂入 U-Net 的跨注意力。B 用 32 个 learned queries 做两层 attentive pooling，再投影为 256 维 patient tokens。

SSL predictor：context feature 投影到 384，加 context 坐标；target mask queries 加其在 context 坐标系中的位置，再通过 3 层 policy MLP 注入 4 个已知观测变换参数；拼接 context/query，6 层 Transformer，最后输出 target 特征。统一实现会创建全部有效 query 位置，local 仅对 masked queries 算 loss。这与原文只构造 masked queries 的实现有计算形式差异，但没有把 target 特征提供给 context。

Observation readout：4 层 Transformer+patch unprojection，将**clean 当前访视 encoder features**解码到 27-channel voxel grid：前 24 个为原 standardized latent 重建，2 个为同访视 signal differences，1 个为 segmentation logit。没有原输入到输出的 skip，所以 reconstruction 必须经过新 encoder。辅助输出不用于生成 MRI。

## 5. 同访视四路目标

固定某个 visit t，采样两个 crop `Y_a, Y_b`。视图的 action 是强度/噪声/模糊参数，不是治疗、时间间隔或临床阶段。

`X_a = visible_patches(T_action_a(Y_a))`

`X_b = visible_patches(T_action_b(Y_b))`

两次 context encoder 前向、两次 no-grad teacher 前向，得到 `Z_a,Z_b,H_a,H_b`。计算：

- `a→a`、`b→b`：预测当前 crop 内 masked clean teacher tokens。
- `a→b`、`b→a`：给定三维相对位置，预测另一 crop 的全部有效 clean teacher tokens。

`L_local = 0.5(MSE_masked(P_aa,H_a)+MSE_masked(P_bb,H_b))`

`L_global = 0.5(MSE_valid(P_ab,H_b)+MSE_valid(P_ba,H_a))`

`L_A = L_local + L_global + 0.25 L_current_reconstruction + optional_current_visit_losses`

一次默认更新的每个 microbatch 还需一次 clean student encoder 前向和一次 readout 前向。即通常是 3 次 student encoder、2 次 teacher encoder、4 次 predictor、1 次 readout，**不是总共一次网络前向**。gradient accumulation 会进一步乘以 microbatch 次数。成本需要实际记录，不能用网络深度推断某 GPU 可运行。

同访视 geometry 使用 latent-to-LPS affine。patch affine 同时纳入 crop origin、patch center 和 patch stride；`A_context^{-1} A_target` 将 target query 映射到 context 的三维坐标。相同访视 crop 的外部旋转/平移并不会改变相对坐标。没有几何时可在同一个 latent array 的 index 坐标内操作，不能宣称这种情况建立了解剖坐标。

## 6. domain modeling 和同访视医学约束

`cache_only.yaml` 默认关闭 image-domain modeling、kinetics、分割、ArcFace 和 DANN。不把对 VQ latent 加噪声/改亮度称作 MRI 扫描域增强。

`prepare-sidecars` 需要真实同访视图像、明确共享的归一化单位和匹配 VQ。先核验 `VQ(images)` 与原 cache 形状和数值大体吻合；再在 image space 做 shared gain/offset、noise/blur，用 VQ 重新编码。缓存记录 source image/latent/VQ 的内容身份、visit ID、action 和 `transform_space=image`。这模拟拟议的观测扰动，不是经过临床验证的 scanner physics。

kinetics 是两条独立信号差 `early−pre`、`late−early`，同访视且在 shared normalized image 单位中定义；不称 SER 或 FTV。下采样到原 latent lattice，损失受同访视三相有效覆盖限制。没有实测图像，默认不计算；codec reconstruction proxy 不可标为 measured。

segmentation 支持同访视可靠 mask 的 soft BCE+Dice；全零肿瘤是有效标签，缺失不置为零。ArcFace 使用单独 MLP semantic projector 和当前 phenotype 索引，不使用患者 ID 或 pCR。DANN 使用单独投影分支、token-wise GRL 和 scanner/site classifier，不对 DCE phase、治疗或时间标签做域消除。两类头在配置的 `supervised_start_step` 后才启动，重建继续保留。

这些 projection 不能保证梯度不会影响 backbone，零相关也不证明生物学解耦。本版没有硬分 anatomy/disease。DANN 的类别混杂风险、与 domain-equivariance 的目标冲突应单独消融，不默认一起启用。

## 7. B 仍采用原 SymmFlow 路径

设 earlier/later standardized latent 为 E/L：

`x_tau = (1-tau)*eps_x + tau*L`

`y_tau = (1-tau)*E + tau*eps_y`

`target_v = concat(L-eps_x, eps_y-E)`

U-Net 一次输出两分支速度。默认 loss 仅为两个分支的 MSE，和原路径一致。`tau` 是生成时间，不是 `delta_days`。A 不编码治疗；B 的 ClinicalEncoder 才编码白名单临床字段、核验后的时间及已知 treatment plan。

B 条件是 `Clinical(source_available) + Adapter(E_obs(current_source))`，没有 predicted future tokens。A 的 encoder 处于 eval、requires_grad=False；B 训练 clinical encoder、state adapter、velocity。另维护 B EMA 用于采样与选择。

MONAI 后端真实实例化 MONAI 1.5.1 DiffusionModelUNet；宽度/注意力层级沿原三相配置。native 是完整 3D U-Net，包含 FiLM ResBlocks、多尺度 skip、bottleneck window self-attention 和 condition cross-attention，不是假网络，也不宣称与 MONAI 相同。没有随包提供任何新预训练权重。

可选 endpoint loss：由同一次预测速度计算 `L_hat=x_tau+(1-tau)*v_x` 或 `E_hat=y_tau-tau*v_y`，经过冻结 observation encoder 后与真实端点特征比较。**生成端点的 encoder 路径保留输入梯度**；真目标 no-grad。默认关闭；它是单次 denoised endpoint estimate，不是完整生成 sample，也不是原文 CheXWorld loss。

原桥接入口只实现“原网络+当前表征条件+原速度 loss”的受控改动，不默默改变原优化器/条件 schema/采样器。独立工作流的可选 endpoint loss 和额外双向训练由独立配置控制。

## 8. 双向与接口边界

独立 workflow 默认 `reverse_probability=0`，只训练有 clean earlier condition 的 forward 任务，与原三相 forecast 入口一致。需要反向推断时，显式设置概率，例如 0.5，让一部分 batch 的 clean condition 来自 later，并用 direction token 区分。checkpoint 记录实际出现的 forward/reverse updates；未训练 reverse 时拒绝 reverse sample。

“反向历史重建”不是治疗逆转，也没有强 cycle loss。原仓库 bridge 只负责原 forward 入口。

sample 只接受 source、conditions、checkpoint、noise seed 及可选匹配 VQ。真实 target 在评价中只在生成完成后读取。所有 raw NPY 坐标和 codec_id 必须核验；同样的 shape 不代表同样的 codebook 或 normalization。

## 9. 验证与复现

测试覆盖单访视数据契约、mask 泄漏、ragged padding、三维 affine、4 路 SSL 梯度、EMA、未知未来条件拒绝、source-only B、frozen input gradient、真实小型 VQ、真实 image-space augmentation bank、完整 A/B 更新、精确恢复和集成脚本。原 core 源文件参考副本的 git blob SHA 与 API 返回一致。

测试结果是代码级证据，不是 I-SPY2 性能。没有真实权重/患者图像，没有 full-size GPU profiling，当前 MONAI 缺失时对应测试 skip。桥接的模板应用与接口代码已检查，但没有在用户私有完整原 workflow、真实缓存与 GPU 上执行正式训练。

公平实验应固定原患者划分、相近预算，比较：原 baseline；只用重建的 encoder；加入同访视 local；加入 global；有实测图像后加入 domain/kinetics；最后单独加 B endpoint loss。A 预训练不能接触独立测试患者，不能只用自己的 learned feature loss 判断生成质量。原 ROI 的未来中心定位、配准和医学终点验证仍需独立检查。
