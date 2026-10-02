> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# 算法与梯度边界

## 1. 概率任务

在当前合法观察集合 H_t 下预测最终二元 pCR 标签 Y。模型近似

`p(Y=1 | H_t) = E_{future trajectory ~ p_theta(.|H_t)}[q_psi(Y=1 | H_t, trajectory)]`。

已观察临床信息和已确定的治疗计划属于 H_t。未知未来治疗不能读取实际执行记录填充；缺失通过显式 mask 表示。当前实现对已给定/缺失的条件进行条件预测，没有实现潜在治疗决策过程的联合因果模型。

每个患者、每条轨迹、每段未来拥有独立新噪声。同一轨迹的后续条件历史包含该轨迹已生成影像和语义，不能跨轨迹错配，也不回填真实中间 MRI。

## 2. 联合对称路径

设源 latent 为 Z_a、未来为 Z_b，源状态 S_a=E_bar(Z_a)、未来状态 S_b=E_bar(Z_b)。独立高斯噪声为 ε、η、ξ、ζ。

`Z_tau = concat((1-tau)*ε + tau*Z_b, (1-tau)*Z_a + tau*η)`

`S_tau = concat((1-tau)*ξ + tau*S_b, (1-tau)*S_a + tau*ζ)`

对应速度目标分别为 `concat(Z_b-ε, η-Z_a)` 和 `concat(S_b-ξ, ζ-S_a)`。两项分别按自身元素数平均，避免图像维度数压倒语义维度数。

正向采样初值 `(ε,Z_a)`、`(ξ,S_a)`，从 τ=0 积分至 1 并取第一半。反向采样以已观察的较晚端点和噪声组成末端，从 1 积分至 0，取第二半。反向表示 retrodiction，不是撤销治疗的生理过程。

速度场无 dropout；相同输入产生相同速度，随机性来自初始噪声。Heun 每步两次场评估；Euler 一次。ODE 累积使用 float32，CUDA 场评估可以在显式选择下使用 BF16。没有用随机 dropout 伪装未来分布。

## 3. 同一未来的图像与状态

`CoupledVelocity` 的 image branch 使用原生或 MONAI U-Net。4 个 semantic blocks 每两个后与 U-Net 瓶颈交换特征。残差门控从零初始化，降低修改预训练图像骨干时的突变；刚初始化的交互不代表立即学会跨流协同。

`L_ground = mean_{patient,k,valid future} ||S_generated - E_bar(Z_generated).disease||²`。

固定 E_bar 的参数，但保留对输入 Z_generated 的导数。它约束“这个样本的语义对应这个样本的影像”，不是令所有随机样本对应唯一真实影像。解剖 tokens 是受约束的表征，不是已证明可识别的解剖/疾病因果解耦。

语义目标是固定的 LayerNorm 后状态坐标，图像仍是原连续 VQ 坐标。B–D 阶段不同时改变语义教师，避免 moving-target 配准。A 阶段 EMA 教师用于掩码目标；B 开始时复制最终在线编码器并冻结。

## 4. 结局读出

历史 H_t 形成 observed residual，已知临床信息形成训练折拟合的固定 logistic prior。多层 pCR Transformer 查询完整状态序列、状态差、时间和真实/生成角色，产生有界 future residual。

`logit_k = clinical_prior + observed_residual(H_t) + bounded_future_residual(H_t, trajectory_k)`

`p_bar = mean_k sigmoid(logit_k)`。

生成轨迹标签损失在 log 空间实现：

`-logsumexp_k(logsigmoid((2Y-1)*logit_k)) + log(K)`。

它不同于 `mean_k BCE(logit_k,Y)`，后者作为显式消融保留。有限 K 的 log Monte Carlo 估计存在偏差；代码并没有宣称无偏似然或“概率均值必然校准”。多次抽样只是模型中的未来变异，不能直接命名为经验证的患者生物学不确定性。

真实轨迹 pCR 分支训练分类器理解实际观察的变化，输入未来属于训练期监督信息。这条分支不能计为早期模型的测试预测。F=0 时只能使用已观察路径，未来 residual 被置零。

## 5. 分布评分

使用固定教师提取的生成图像状态，而不仅是自由语义向量，计算 Energy Score：

`ES = 1/K * sum_k ||s_k-y|| - 1/(2K(K-1)) * sum_{k!=l} ||s_k-s_l||`。

以有效未来坐标数平方根归一化，每个患者分别计算。缺失未来坐标投影掉，全缺失不产生梯度；至少需要 K=2。有限样本 U-statistic 可能略为负值，不能因此裁剪成零。该分数只在选定表征/已观察子轨迹空间评价分布，不证明完整像素分布已校准。

联合损失为 FM_image + FM_state + spatial_alignment + real_pcr + marginal_pcr + observed_pcr + grounding + Energy Score + 小权重 readout 一致性与 residual L2。具体权重见 YAML，不是已调优的“论文最优参数”。

## 6. 空间/教师监督

已有 V2 的多尺度编码器、phase difference、掩码先于卷积、QueryPool 保留。三相 latent 差不是未经测量的 MRI 生理增强率。全局 Pillar 特征只进入全局蒸馏；dense teacher 必须有实际空间 token，不能重复全局特征。空间对齐使用卷积投影及按空间维中心化/方差标准化后的 cosine loss，借鉴 iREPA，不声称等价复现其自然图像实验。

分割/kinetics/biomarkers 需要真实或明确有来源的 sidecars 与有效性 mask。没有任何 sidecar 时对应损失为零，并记录 support 计数。当前这些辅助目标主要用于 A 阶段表征，不是宣称每项都在联合解码影像上直接监督。Pillar 权重不随包分发。

VQ decode 前先反标准化、进行匹配 codebook 的最近邻映射。最近邻 forward 是精确选择，backward 使用 straight-through 近似，不是假装离散量化具有真实的恒等导数。默认主联合训练在 latent 空间，独立解码后检查是必要的补充实验。

## 7. 不泄漏不等于已有效

FM 插值确实包含真实未来，但它只用于密度训练；代码没有拿此端点估计代替 source-only 结局路径。`ForecastInput` 类型与 `Supervision` 分开，文件级部署接口也不接受标签和未来路径。

可用时间/几何声明由你提供。软件只能检验逻辑一致性，不能证明临床记录在那个时刻确实可获得，也不能证明人工画的 ROI 没看过未来。测试覆盖的是接口隔离，不是对所有可能数据泄漏的形式化安全证明。

固定参数下生成未来来自历史与独立噪声，不是新获得的患者检查。优势只能来自纵向学习的归纳偏置/表征和概率结构，必须与使用相同纵向训练信息的直接判别模型比较。

pCR 涉及最终病理结局，不能硬编码为“影像体积消失”。本代码没有治疗选择因果识别、随机试验替代、个体最佳方案推荐或临床自动决策功能。
