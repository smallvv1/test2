# NDP_SELECT LTCC/PRCC 短程结果

本轮 seed=1234，LTCC/PRCC 各完成 B0 与 NDP_SELECT 的 5 raw continuation epoch；两臂从同一 model-only warm-start 起点开始，GPC-v1、权重 0.1、EVA02、主损失、sampler 与评估协议固定。

| 数据集/协议 | B0 R1 | B0 mAP | NDP_SELECT R1 | NDP_SELECT mAP |
|---|---:|---:|---:|---:|
| LTCC General | 86.004 | 53.780 | 86.004 | 53.970 |
| LTCC Clothes-Changing | 58.163 | 32.665 | 56.378 | 33.127 |
| PRCC Same-Clothes | 100.000 | 99.361 | 100.000 | 99.416 |
| PRCC C→A | 63.731 | 63.433 | 64.380 | 63.594 |

结论：PRCC C→A 有小幅正向变化，但 LTCC Clothes-Changing Rank-1 下降 1.786 pp；没有跨数据集稳定增益证据，因此不扩展正式 60 epoch。NDP 梯度统计已改为有限值记录，B0 不计算 NDP。评估器未提供 mINP。完整日志在服务器 \`/home/DIFFER/log_NDP/ndp_select_short_20261030_151334/\`。
