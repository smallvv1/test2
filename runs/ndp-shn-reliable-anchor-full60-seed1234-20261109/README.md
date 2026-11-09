# NDP_SHN_RELIABLE_ANCHOR 正式日志归档

- seed: `1234`
- datasets: LTCC, PRCC
- epochs: `60`
- device: 2 × PPU-ZW810E
- arm: `NDP_SHN_RELIABLE_ANCHOR`
- lambda_gpc: `0.1`（GPC 未修改）
- lambda_ndp: `0.001`
- training references: train split only
- official test query/gallery used during training: `false`
- per-rank checkpoint: `false`

本目录仅存放小体积日志、CSV、JSON 和评估输出，不包含约 1.2GB 的模型文件。完整模型和原始日志保存在 `/home/DIFFER/log_NDP/ndp_shn_reliable_anchor_full60_seed1234_20261109_restart/`。

结论：Anchor-view 风险定义和稀疏边界候选选择器已完成正式验证，但 LTCC 明显退化、PRCC 仅 mAP 小幅提升且 Rank-1 下降，不能作为稳定 NDP 增益。
