# NDP_RANK LTCC / PRCC 正式 60 Epoch 结果

## 结论摘要

本轮 `NDP_RANK` 在 seed=1234 下完成了 LTCC 与 PRCC 各 60 个完整训练 epoch，并通过官方评估。两套实验均未生成 `rank*state.pt`、`rank*state.pth` 或其他逐 Rank 状态文件。

NDP_RANK 没有形成稳定的跨数据集增益：LTCC Clothes-Changing 的 Rank-1 和 mAP 均下降；PRCC C→A 有小幅上升，PRCC Same-Clothes 基本饱和且仅 mAP 有极小变化。因此不将本轮结果表述为已验证的 NDP 提升。

## 运行与版本

- 代码提交：`3a1c2ffe394cf346d24ac2842c15919d0222b4b7`（`Run frozen-teacher NDP ranking arm`）。
- NDP runner SHA256：`4804c776a937a382aaa4de7dee5766757b884acf7677b58293ddccce1edcfd31`。
- 正式 60 epoch 入口 SHA256：`45705bdc6c9123af401a44d7d8b34ee84293c7b29b303a2579d9a73a131ed31d`。
- seed：`1234`。
- 输出目录：`/home/DIFFER/log_NDP/ndp_rank_full60_seed1234_20261031_183800/`。
- GPC context：`phase7i_relation_specificity`；GPC 权重固定为 `0.1`。
- 训练仅使用 train split 建图；`query/gallery_used_during_training=false`。
- 设备：PPU/GPU 双卡 torchrun；未使用 CPU 训练回退。

实际启动命令（LTCC；PRCC 仅替换 `--dataset`、初始化模型、教师模型和输出目录）：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/torchrun --standalone --nproc_per_node=2 \
  /home/DIFFER/scripts/train_ndp_listwise_full60.py \
  --mode train --dataset ltcc --seed 1234 --arm NDP_RANK --epochs 60 \
  --init-model /data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth \
  --teacher-model /home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth \
  --output-dir /home/DIFFER/log_NDP/ndp_rank_full60_seed1234_20261031_183800/LTCC_NDP_RANK_full
```

## 方法定义

基础损失保持为：

`L_base = CE + Triplet + clipBio + 0.1 * GPC-v1`

NDP_RANK 使用已完成的 Bio+GPC 模型作为冻结教师，仅在 train-only 关系图上提供局部排序参考。教师关系要求 `0.02 <= M_teacher <= 0.10`；学生当前 margin 要求小于 `0.05`；目标为：

`target = clamp(0.5 * M_teacher, 0.01, 0.05)`

`L_total = L_base + 0.005 * ReLU(target - M_student)`

正负端点特征停止梯度，当前 anchor 保留梯度。没有修改 EVA02、CE、原始 Batch-Hard Triplet、clipBio、GPC-v1、GPC 正样本选择、采样器或评估器。

教师模型：

- LTCC：`/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth`
- LTCC SHA256：`f3f89f4359267699c8d7abd55d843d0f30a3b43d506652acb984f7d3b38b7688`
- PRCC：`/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/model_final.pth`
- PRCC SHA256：`74aed55925603adfdfb07325f5b6646608a8d60d23c613852c5951d521263772`

## 官方测试结果

百分比均为评估器原始比例乘以 100；差值为 NDP_RANK 减 Bio+GPC B0，单位为百分点。mINP 为仓库评估器不支持，记为 Unavailable。

| 数据集 / 协议 | Bio+GPC B0 Rank-1 | NDP_RANK Rank-1 | ΔRank-1 | Bio+GPC B0 mAP | NDP_RANK mAP | ΔmAP | NDP_RANK Rank-5 / Rank-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 84.179 | -0.203 | 54.186 | 55.193 | +1.007 | 91.075 / 93.103 |
| LTCC Clothes-Changing | 55.357 | 53.061 | -2.296 | 33.937 | 32.235 | -1.701 | 69.388 / 75.000 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.503 | +0.024 | 100.000 / 100.000 |
| PRCC C→A | 64.098 | 64.380 | +0.282 | 63.414 | 64.103 | +0.689 | 71.211 / 74.457 |

B0 来源为已完成的正式 Bio+GPC 模型：LTCC `/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth`，PRCC `/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/model_final.pth`。它们与本轮使用相同 seed、正式初始化来源、训练协议和 GPC context，但不是同一进程内逐步复制的 bitwise 分支；PPU/AMP 非确定性仍是可比性限制。

## 论文结果表（当前可核验数据）

Bio-only 的 PRCC 正式 60 epoch 结果没有在本轮复测，不能填入猜测值。LTCC Bio-only 数值来自历史报告，非本轮严格配对；因此论文正文使用前应重新核对其初始化和协议。

| 数据集 / 协议 | Bio-only | Bio+GPC | Bio+GPC+NDP_RANK |
|---|---:|---:|---:|
| LTCC General Rank-1 / mAP | 历史：N/A | 84.381 / 54.186 | 84.179 / 55.193 |
| LTCC Clothes-Changing Rank-1 / mAP | 历史：49.490 / 28.793（非本轮配对） | 55.357 / 33.937 | 53.061 / 32.235 |
| PRCC Same-Clothes Rank-1 / mAP | N/A（未复测） | 100.000 / 99.479 | 100.000 / 99.503 |
| PRCC C→A Rank-1 / mAP | N/A（未复测） | 64.098 / 63.414 | 64.380 / 64.103 |

## 训练统计

最后一个 epoch 的统计来自 `metrics_per_epoch.csv`；关系图数量在整个训练中固定，避免把关系重建变化误当作性能证据。

| 数据集 | train 图像 | 在线关系数 | 最后 epoch NDP active batch | 最后 epoch active rate | selected / active relations | NDP loss | NDP/main feature-grad ratio | skipped / effective updates（全程） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LTCC | 9,576 | 28,728 | 488 / 596 | 81.879% | 2,128 / 1,260 | 0.017059 | 0.042806 | 14 / 35,746 |
| PRCC | 17,896 | 53,688 | 130 / 551 | 23.593% | 289 / 190 | 0.006535 | 0.091150 | 13 / 33,047 |

全程关系 active 比例约为 LTCC `138,862 / 177,567 = 78.20%`、PRCC `19,593 / 25,337 = 77.33%`（按 epoch 汇总的 selected/active 统计）。这说明实现确实产生了梯度和约束，但“激活”本身没有带来检索性能增益。

## 文件与安全核验

- LTCC 最终模型：`LTCC_NDP_RANK_full/LTCC/model_final.pth`
- LTCC SHA256：`0f92102db0da32e39f5823393e26703e054455f690c1e68e45a17efca251dd86`
- PRCC 最终模型：`PRCC_NDP_RANK_full/PRCC/model_final.pth`
- PRCC SHA256：`a2c685df02375118ec2cfec859c6a759303f3caaef831a3f3a796f481ad45d3d`
- LTCC、PRCC 均存在 `RUN_COMPLETE.json`，`epochs_completed=60`。
- 训练结束后 `resume_latest.pth` 已清理；最终模型保留。
- 输出目录扫描 `rank*state.pt`、`rank*state.pth`、`rank*state*`：无匹配。
- `mINP`：Unavailable，评估器未暴露该指标。

## 决策

本轮回答了“能否完成正式 60 epoch 并真实参与训练”：可以。NDP_RANK 的 NDP loss、非零特征梯度、AMP 统计和最终模型均可核验。

但回答“是否已经获得稳定性能提升”：不能。LTCC Clothes-Changing 的两个主要指标均恶化，PRCC C→A 仅有小幅正向变化，且 LTCC General 也出现 Rank-1 轻微下降。当前不建议把 NDP_RANK 作为论文主结果，也不建议在没有新的监督增量证据前继续扩大到第二 seed 或 60 epoch 扫描。

下一步若继续，应优先做严格的 Bio+GPC 同步分支与局部监督增量验证；不应仅根据更高激活率继续加强静态历史关系约束。
