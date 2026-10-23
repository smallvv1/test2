# NDP-V2 B0 对照（两 seed，update596）

数据目录：`/home/DIFFER/log_NDP/ndp_v2_20261023_061358`。B0 与 N2 都从同一 seed 的 E60 `rank0/1_state_epoch060.pt` 启动，LR 同为 `1e-6`，同 sampler 顺序，完成 596 个有效 optimizer update。官方 LTCC 结果来自固定 query/gallery clothes-changing 过滤、归一化 `feat[0]`、无 rerank；不是用来选超参数。

## LTCC 结果

| Seed | 模型 | 训练 Epoch | Rank-1 | mAP | mINP |
|---|---|---|---:|---:|---:|
| 1234 | B0 | 1 continuation epoch / 596 updates | 0.533163 | 0.318329 | 0.086222 |
| 1234 | N2 | 1 continuation epoch / 596 updates | 0.543367 | 0.325284 | 0.089718 |
| 4321 | B0 | 1 continuation epoch / 596 updates | 0.512755 | 0.310752 | 0.081075 |
| 4321 | N2 | 1 continuation epoch / 596 updates | 0.517857 | 0.314960 | 0.083284 |

差值（N2-B0）：

| Seed | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---:|---:|---:|
| 1234 | +0.010204 | +0.006955 | +0.003495 |
| 4321 | +0.005102 | +0.004208 | +0.002209 |
| 两 seed均值 | +0.007653 | +0.005582 | +0.002852 |

两 seed 差值的样本标准差分别为 Rank-1=`0.003608`、mAP=`0.001942`、mINP=`0.000910`。样本数只有2，不能作显著性或泛化保证；但方向在两个 seed 上一致。

## 训练侧与激活覆盖

| Seed | Arm | 有效更新 | 激活 batch/率 | selected | active relation | mean NDP | AMP skip |
|---|---|---:|---:|---:|---:|---:|---:|
| 1234 | B0 | 596 | 0/596 (0.000%) | 0 | 0 | 0.000000 | 0 |
| 1234 | N2 | 596 | 274/597 (45.896%) | 1042 | 423 | 0.009743 | 1 |
| 4321 | B0 | 596 | 0/596 (0.000%) | 0 | 0 | 0.000000 | 0 |
| 4321 | N2 | 596 | 253/596 (42.450%) | 995 | 386 | 0.009163 | 0 |

V2 相对历史 N1：N1 seed1234 为约 `54/400=13.5%` active batch、135 active relation；seed4321 为约 `42/400=10.5%`、98 active relation。V2 第一 epoch 为 seed1234 `274/597=45.9%`、423 active relation；seed4321 `253/596=42.4%`、386 active relation。因此 V2 明显更容易激活，但覆盖增大并不自动等于局部安全。

## train-only 局部身份指标

这些指标在固定的 2048-anchor、9576-image train-only bank 上计算；逐关系 CSV 为：[seed1234 CSV](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/local_metrics/seed1234/local_relation_metrics_seed1234.csv)、[seed4321 CSV](/home/DIFFER/log_NDP/ndp_v2_20261023_061358/local_metrics/seed4321/local_relation_metrics_seed4321.csv)。

| Seed | Arm | correct relation rate | mean margin | 新增错误 PID（相对 update0） | 当前错误 PID | Purity@10 |
|---|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 0.82080078125 | 0.06199808842211496 | 97 | 899 | 0.972021484375 |
| 1234 | N2 | 0.81494140625 | 0.06151092174695805 | 129 | 932 | 0.970703125 |
| 4321 | B0 | 0.81884765625 | 0.054427792521892115 | 52 | 903 | 0.968212890625 |
| 4321 | N2 | 0.81982421875 | 0.0546735509269638 | 56 | 887 | 0.9676269531249999 |

seed1234：N2 的 relation rate `0.814941` 低于 B0 `0.820801`，新增错误 PID `129` 高于 `97`，Purity@10 `0.970703` 低于 `0.972021`。seed4321：N2 的 relation rate `0.819824` 略高于 B0 `0.818848`，当前错误 PID `887` 低于 `903`，但 Purity@10 仍略低（`0.967627` vs `0.968213`）。因此“局部身份 margin/Purity 一致改善”没有成立；官方检索提升与局部关系安全性存在待解释的分离。

## 正面与负面证据

- 正面：NDP 的反应式项确实有训练梯度；V2 将 active batch 从 N1 的约10%–14%提高到约42%–46%；两个 seed 的 LTCC Rank-1/mAP/mINP 同方向提升。
- 负面：seed1234 的 train-only 局部关系退化更明显；seed4321 虽 current error PID 减少，Purity@10 仍下降。N2 seed1234 还有一次 AMP skip。
- 因此不能把 LTCC 两 seed 的提升直接解释为“目标身份边界被安全保护”；需要继续做局部附带影响和梯度预算审计。

## 未完成事项

没有跑 3–5 epoch、10 epoch 或 60 epoch；没有将官方指标用于调参；没有修改 GPC、EVA02、Triplet、Bio、采样或评估协议；没有推送 GitHub main。
