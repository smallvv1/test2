# GPC S1-Pair 可辨识性审计归档

本归档基于远端参考提交 `de4eee375fbfc216948495200ee06e4b7d1f2eef`。

内容：

- `source_snapshot/`：实际执行的 S1-Pair、A/A、A/B、预算审计和 Phase7I 源码快照；
- `results/`：S1-Pair 完整 epoch 离线 JSON/CSV、A/A/A/B 原始小型 JSON、预算计划 JSON；
- `reports/`：本轮中文报告；
- `logs/`：本轮离线审计日志。

明确排除：checkpoint、数据集、Caption 特征、图片、PPU cache、Token、密钥和私钥。

模型 forward/backward 未在本轮执行；S1-Pair 审计为 train-only CPU 离线关系计划审计，`cpu_model_fallback=false`。

本归档不把 S1-Exposure 解释为纯正样本对选择：严格 Pair 可替换率为 55/301（18.27%）。
