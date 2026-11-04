# NDP-SHN Full60 Archive

本目录只归档小体积实验摘要，不包含模型权重或逐 Rank checkpoint。完整训练/评估日志位于服务器：

`/home/DIFFER/log_NDP/ndp_shn_full60_seed1234_20261104/`

| Dataset / protocol | Bio+GPC Rank-1 / mAP | NDP-SHN Rank-1 / mAP | Δ Rank-1 / mAP (pp) |
|---|---:|---:|---:|
| LTCC General | 84.3813 / 54.1862 | 84.1785 / 54.1941 | -0.2028 / +0.0079 |
| LTCC CC | 55.3571 / 33.9367 | 56.6327 / 33.4036 | +1.2755 / -0.5331 |
| PRCC Same-Clothes | 100.0000 / 99.4787 | 100.0000 / 99.4692 | +0.0000 / -0.0095 |
| PRCC C→A | 64.0982 / 63.4143 | 64.1547 / 63.6552 | +0.0565 / +0.2409 |

训练协议：seed=1234、60 epoch、两张 PPU、GPC 权重 0.1、NDP-SHN 权重 0.001、epoch 3 起激活；仅使用 train split 做身份中心与负身份选择。正式目录的 `RUN_COMPLETE.json` 标记没有生成逐 Rank checkpoint，仅保留最终 `model_final.pth`。

详细结论见 [`reports/NDP_SHN_FULL60_RESULTS_20261104.md`](../../reports/NDP_SHN_FULL60_RESULTS_20261104.md)。
