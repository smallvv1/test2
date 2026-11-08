# NDP-SHN reliable-risk full60 seed=1234 archive

本目录保存本轮 `NDP_SHN_RELIABLE_RISK` 的小体积复现实验材料。正式模型约 1.2 GB，未复制到仓库；模型路径和 SHA256 记录在 `summary.json` 与正式日志目录的 `RUN_COMPLETE.json` 中。

- `metrics_per_epoch.csv`：60 epoch 训练统计。
- `train.log`：epoch 级训练日志。
- `online_neighbor_summary.csv`：train-only bank 刷新与关系覆盖摘要。
- `LTCC_eval_final.json`、`PRCC_eval_final.json`：官方统一评估结果。
- `LTCC_RUN_COMPLETE.json`、`PRCC_RUN_COMPLETE.json`：训练完成、配置和 checkpoint 完整性元数据。
- 完整日志保留在 `/home/DIFFER/log_NDP/ndp_shn_reliable_risk_full60_seed1234_20261108/`，历史目录未覆盖。
