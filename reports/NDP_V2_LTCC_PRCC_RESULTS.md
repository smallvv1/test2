# NDP_V2 LTCC / PRCC 短程结果

代码实验提交：`9451d9979db3860eca9cfa7936137b112816708b`。

本轮 seed=1234，LTCC/PRCC 各完成 5 raw epoch；两者均通过 20-step PPU Smoke。NDP_V2 只在 epoch 3–5 以 `0.0005 -> 0.001` 逐步启用，GPC-v1、权重 0.1、EVA02、CE、Triplet、Bio、sampler 和评估协议未修改。参考图仅使用 train split。

| 数据集/协议 | B0 R1 | B0 mAP | NDP_V2 R1 | NDP_V2 mAP | ΔR1 (pp) | ΔmAP (pp) |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 86.004 | 53.780 | 85.396 | 53.770 | -0.609 | -0.010 |
| LTCC Clothes-Changing | 58.163 | 32.665 | 55.867 | 32.984 | -2.296 | +0.319 |
| PRCC Same-Clothes | 100.000 | 99.361 | 100.000 | 99.360 | +0.000 | -0.001 |
| PRCC C→A | 63.731 | 63.433 | 63.872 | 63.477 | +0.141 | +0.044 |

PRCC 初次正式启动因逐候选正样本相似度同步过慢而停止；随后改为分块张量计算，重新运行完整 5 epoch，未覆盖失败目录。LTCC/PRCC 的最终模型、metrics、评估 JSON 和 `SHORT_RUN_COMPLETE.json` 保存在服务器的 `log_NDP/ndp_v2_short_20261030_173957` 与 `ndp_v2_short_20261030_180855`。mINP 为评估器不支持（Unavailable）。

训练末轮：LTCC NDP raw loss `0.010102`，selected/active `3620/2090`，AMP skip `0`；PRCC raw loss `0.004313`，selected/active `834/364`，AMP skip `0`。本轮新目录未生成 `rank*state.pt` 或 `rank*state.pth`；只使用单文件恢复机制和最终模型权重。

结论：LTCC Clothes-Changing Rank-1 明显下降，PRCC C→A 仅有 `+0.141 pp` 的探索性小幅变化，尚无跨数据集稳定增益证据，**不扩展 NDP_V2 正式 60 epoch**。
