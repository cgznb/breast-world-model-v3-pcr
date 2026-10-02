> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# 多阶段训练与在线使用

原工程：`/home/cxx/test/breast_world_joint`。正式任务已由独立进程启动，通常只需检查日志，
不要在现有控制器仍运行时再次启动同一任务。

```bash
tail -f /data1/cxx/breast_multistage_v2_20260929/run/console.log
cat /data1/cxx/breast_multistage_v2_20260929/run/controller_status.json
```

## 训练和恢复

```bash
cd /home/cxx/test/breast_world_joint
BREAST_PY=/home/cxx/gastric_world_model_codex/.venv/bin/python

$BREAST_PY joint.py audit-v2 \
  --manifest /data1/cxx/breast_multistage_v2_20260929/data/patient_trajectories_v2.json \
  --scan-arrays --output /data1/cxx/breast_multistage_v2_20260929/audit_recheck.json

# 仅当控制器已退出时使用；自动跳过已完成阶段并恢复中断阶段。
$BREAST_PY joint.py train-v2 --stage all --resume \
  --config configs/ispy2_multistage_native_v2.yaml \
  --manifest /data1/cxx/breast_multistage_v2_20260929/data/patient_trajectories_v2.json \
  --output /data1/cxx/breast_multistage_v2_20260929/run

# 独立工程 smoke，必须指定尚不存在训练结果的新目录。
$BREAST_PY joint.py smoke-v2 --config configs/multistage_smoke_v2.yaml \
  --output /data1/cxx/breast_multistage_v2_20260929/smoke_new
```

## 五个接口

`initialize(prefix)` 建立真实观察状态；`advance(state, IntervalSpec(0,1))` 返回单段 ForecastTrace，
下一状态为 `.final_state`；`forecast` 在副本展开；`observe` 接收新真实 MRI；`query_pcr` 只读。
模型类为 `responsewm.model_v2.MultistageResponseWorldModel`。
可执行的五接口演示同时位于 `tests/test_inference_v2.py::test_five_state_apis_and_cli_without_future_assets`。

```bash
# request.json 是 input-only responsewm_request_v2，不能传训练 manifest。
$BREAST_PY joint.py initialize-v2 --checkpoint /path/to/joint/best.pt \
  --request /path/to/request.json --output /path/to/T0_state.pt --device cuda

# 同时保存每个模拟阶段的 prior，供新真实 MRI 到达时更新。
$BREAST_PY joint.py forecast-v2 --state /path/to/T0_state.pt \
  --output-stages 1 2 3 --samples 8 --steps 20 --seed 20260929 --device cuda \
  --output /path/to/T0_openloop.npz --state-output-dir /path/to/priors

$BREAST_PY joint.py observe-v2 --state /path/to/priors/T1_prior.pt \
  --observation /path/to/available_T1.json --output /path/to/T1_posterior.pt --device cuda

$BREAST_PY joint.py query-pcr-v2 --state /path/to/T1_posterior.pt \
  --mode observed_landmark --output /path/to/T1_pcr.json --device cuda

$BREAST_PY joint.py forecast-v2 --state /path/to/T1_posterior.pt \
  --output-stages 2 3 --samples 8 --steps 20 --device cuda --output /path/to/T1_future.npz
```

请求沿用原 input 字段 `landmark_day/observed/clinical/clinical_known_at/queries/source_only_geometry`，
但 schema 为 `responsewm_request_v2`、time_basis 为 stage_index。请求顶层还须提供与检查点完全匹配的
clinical_features、action_features、phase_order、latent_shape、vq_identity。不允许 label、target、患者对象等额外字段。
示例构造函数见测试中的 `input_request`，其使用匿名 synthetic 数据。

观察 JSON 的严格字段为：

```json
{
  "schema": "responsewm_observation_v2",
  "stage": 1,
  "available_at": 1,
  "latent": "/path/to/raw_T1_latent.npz",
  "event_id": "new_real_T1",
  "vq_identity": "sha256:0957a168dc407e602fcadb789f54775838dd568a8f6f2c110f860c222865daae"
}
```

传入原始连续 VQ latent，加载器按训练统计标准化；导出的 `.npz` 已反标准化。
可选 `forecast-v2 --codec /data1/cxx/compare/local/data/codec.pt` 解码，先核对 SHA。
真实影像到达后使用新 posterior 重建未来，旧 prior 文件只留作历史审计。

## 评估

```bash
$BREAST_PY joint.py evaluate-v2 --checkpoint /path/to/joint/best.pt \
  --manifest /data1/cxx/breast_multistage_v2_20260929/data/patient_trajectories_v2.json \
  --split val --device cuda --bootstrap 1000 --output /path/to/evaluation_v2.json
```

输出真实 prefix、同 prefix 生成辅助结果、模拟阶段诊断和 prequential 结果。
该数据没有独立 test，使用 `--split test` 会报错。消融配置只准备未运行，位于 `configs/ablations_v2/`。
