> 历史版本参考文档；当前 V3 使用仓库首页及 `docs/V3_RUN_ZH.md` 的命令与协议。

# 文件说明

这是给 Codex 的**修改规范包**，不是已修改代码包。

- `BREAST_MULTISTAGE_CODEX_SPEC_ZH.md`：主规范，26节，逐层设计、数据合同、状态接口、训练、迁移、60条验收及公开源码索引。
- `CODEX_START_HERE.txt`：可直接粘贴到 Codex 的启动指令。先提供主规范，再粘贴此文件。
- `multistage_v2.proposed.yaml`：配置设计草案；需要 Codex 先实现并注册 v2 schema，不能直接用于当前 v1。
- `DOCUMENT_VALIDATION.json`：文档结构/代码片段语法/YAML及维度一致性检查，不是模型测试结果。
- `PACKAGE_MANIFEST.json`：文件SHA256。

规范基线：`cgznb/medical-world-models@841458cb8e06e131af2a2248dc853f8476b2e0d0`，2026-09-29核验。

本次不修改仓库，不启动训练，不包含患者数据、权重或未经执行的性能结果。
