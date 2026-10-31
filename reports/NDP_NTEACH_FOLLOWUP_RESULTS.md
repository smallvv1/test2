# NTEACH 后续配对实验结果（2026-10-31）

代码 commit：`acb1d9d`。新增 `NTEACH_SAFE`：在原冻结教师门控上增加 detached 的 `safe_positive_reliable` 条件，要求正样本由至少两个不同衣物 ID 支持；GPC-v1 与 `0.1` 权重不变。

| 数据集/协议 | 五轮 B0 R1 / mAP | 五轮 NTEACH R1 / mAP | 五轮 NTEACH_SAFE R1 / mAP |
|---|---:|---:|---:|
| LTCC General | 84.381 / 54.431 | 85.598 / 54.101 | 83.367 / 54.193 |
| LTCC CC | 54.337 / 33.411 | 55.357 / 33.015 | 55.612 / 33.133 |
| PRCC Same-Clothes | 100.000 / 99.426 | 100.000 / 99.347 | 100.000 / 99.447 |
| PRCC C→A | 64.014 / 62.829 | 63.731 / 62.891 | 63.506 / 62.876 |

结果显示 NTEACH_SAFE 未同时改善 LTCC/PRCC 的 Rank-1 和 mAP；不启动正式 NTEACH_SAFE 60 epoch。PRCC Bio+GPC 60 epoch 已完成官方评估：Same-Clothes 100.000/99.479，C→A 64.098/63.414。新实验仅使用 train split 建图，未生成逐 Rank checkpoint；仅保留单文件恢复点和最终模型。
