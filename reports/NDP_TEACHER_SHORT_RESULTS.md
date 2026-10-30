# NDP-Teacher（NTEACH）LTCC / PRCC 短程结果

本轮 seed=1234，LTCC/PRCC 的 B0 与 NTEACH 均完成 5 raw epoch 和固定官方评估。NTEACH 使用初始化时冻结的 Bio+GPC 教师特征图，只对教师 margin ≥0.02 且当前 margin 进入 `(-0.05, 0)` 的真实混淆关系施加有限 hinge；lambda=0.001，从第2轮启用。GPC-v1、权重0.1、EVA02、CE、Triplet、Bio、sampler和评估协议未修改。

| 数据集/协议 | B0 R1 | B0 mAP | NTEACH R1 | NTEACH mAP | ΔR1 (pp) | ΔmAP (pp) |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.976 | 52.531 | 83.773 | 52.813 | -0.203 | +0.283 |
| LTCC Clothes-Changing | 47.959 | 28.714 | 51.531 | 29.454 | +3.571 | +0.740 |
| PRCC Same-Clothes | 100.000 | 99.167 | 100.000 | 99.196 | +0.000 | +0.030 |
| PRCC C→A | 64.437 | 63.104 | 64.268 | 62.840 | -0.169 | -0.265 |

LTCC 显示正向线索，但 PRCC C→A 同时下降；当前没有跨数据集稳定增益证据，因此不扩展正式 60 epoch。四组新实验均没有生成 `rank*state.pt/pth`，仅保留单文件恢复与最终模型机制。mINP 为评估器不支持（Unavailable）。

代码实现提交：本分支的 NTEACH 代码；训练日志和完整表格保存在服务器 `/home/DIFFER/log_NDP/ndp_nteach_short_20261030_184754/`。
