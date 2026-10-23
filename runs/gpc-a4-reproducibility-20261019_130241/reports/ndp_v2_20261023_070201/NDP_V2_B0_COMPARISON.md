# NDP-V2 B0/N2 对照（1 epoch 与 3 epoch）

## 固定 LTCC update596：短程结果

| Seed | 模型 | 训练 Epoch | Rank-1 | mAP | mINP |
|---|---|---|---:|---:|---:|
| 1234 | B0 | 1 epoch / 596 updates | 0.533163 | 0.318329 | 0.086222 |
| 1234 | N2 | 1 epoch / 596 updates | 0.543367 | 0.325284 | 0.089718 |
| 4321 | B0 | 1 epoch / 596 updates | 0.512755 | 0.310752 | 0.081075 |
| 4321 | N2 | 1 epoch / 596 updates | 0.517857 | 0.314960 | 0.083284 |

| Seed | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---:|---:|---:|
| 1234 | +0.010204 | +0.006955 | +0.003495 |
| 4321 | +0.005102 | +0.004208 | +0.002209 |
| 均值 | +0.007653 | +0.005582 | +0.002852 |

## 固定 LTCC update1788：3 epoch结果

| Seed | 模型 | 训练 Epoch | Rank-1 | mAP | mINP |
|---|---|---|---:|---:|---:|
| 1234 | B0 | 3 epochs / 1788 updates | 0.505102 | 0.303841 | 0.083263 |
| 1234 | N2 | 3 epochs / 1788 updates | 0.500000 | 0.300502 | 0.079472 |
| 4321 | B0 | 3 epochs / 1788 updates | 0.507653 | 0.307395 | 0.085509 |
| 4321 | N2 | 3 epochs / 1788 updates | 0.515306 | 0.310650 | 0.087107 |

| Seed | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---:|---:|---:|
| 1234 | -0.005102 | -0.003339 | -0.003791 |
| 4321 | +0.007653 | +0.003255 | +0.001598 |
| 均值 | +0.001276 | -0.000042 | -0.001097 |

1 epoch 的两个 seed 方向均为正；3 epoch 的 seed1234 变为负，seed4321 仍为正，均值 Rank-1 仅约 `+0.001276`、mAP `-0.000042`、mINP `-0.001097`，不支持稳定的长程增益。3 epoch 结果完整来自 `/home/DIFFER/log_NDP/ndp_v2_20261023_070201/ltcc_seed*_update1788.json`，没有用中间节点反向选参。

## train-only局部指标（update596）

局部指标来自固定 2048 anchor 和 9576-image train-only bank；逐关系 CSV：[seed1234](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/local_metrics/seed1234/local_relation_metrics_seed1234.csv)、[seed4321](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/local_metrics/seed4321/local_relation_metrics_seed4321.csv)。

| Seed | Arm | correct relation rate | mean margin | 新增错误 PID | 当前错误 PID | Purity@10 |
|---|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 0.82080078125 | 0.06199808842211496 | 97 | 899 | 0.972021484375 |
| 1234 | N2 | 0.81494140625 | 0.06151092174695805 | 129 | 932 | 0.970703125 |
| 4321 | B0 | 0.81884765625 | 0.054427792521892115 | 52 | 903 | 0.968212890625 |
| 4321 | N2 | 0.81982421875 | 0.0546735509269638 | 56 | 887 | 0.9676269531249999 |

seed1234 的 N2 局部 relation rate/Purity@10 比 B0 差，seed4321 current error PID 减少但 Purity@10 仍略降；局部安全性没有稳定证据。3 epoch 未重新导出全量局部特征，避免重复昂贵前向；因此不能把 update596 局部表误标为 update1788。

## 训练与评估完整性

- B0/N2 同 E60 起点、同 LR、同 sampler、同有效更新预算；N2 仅额外计算独立 NDP。
- 官方 query/gallery 只在预先固定的 update596/update1788节点报告，不参与参数选择。
- PPU EVA02 A/A 既有约0.4%差异保留；本轮不声称已修复。
- 正面：NDP 确实训练、V2 激活率显著高于 N1、1 epoch LTCC 两 seed 同向提升。
- 负面：3 epoch 增益不一致，seed1234 长程退化；局部指标存在附带损害；N2长跑仍有 AMP skip。
