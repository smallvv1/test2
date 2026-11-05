# NDP_SHN_MP Full60 / seed 1234

本目录只保存小体积实验摘要；完整训练日志和模型位于：

`/home/DIFFER/log_NDP/ndp_shn_mp_full60_seed1234_20261105/`

- LTCC、PRCC 均完成 60/60 epoch。
- GPC 固定为原实现，权重 `0.1`；NDP_SHN_MP 权重 `0.001`。
- 训练只使用 train split，官方 query/gallery 仅用于最终评估。
- 只保留 `resume_latest.pth`（训练中）和 `model_final.pth`（最终）；没有逐 Rank checkpoint。
- 完整指标、变化量、训练状态和模型哈希见 [`NDP_SHN_MP_FULL60_RESULTS_20261106.md`](../../reports/NDP_SHN_MP_FULL60_RESULTS_20261106.md)。
