# NTEACH 与当前正式 B0 重核对（阶段报告）

## 状态

本轮仍未完成：LTCC 已完成 B0 60 epoch、B0/NTEACH 5 epoch 配对及官方评估；PRCC B0 正在从正式 common initialization 进行 60 epoch 重建，截至本报告为 5/60 epoch，NTEACH 尚未启动。

## 代码与约束

- GitHub main commit：`755779d`
- GPC context：`phase7i_relation_specificity`
- 基础损失：`CE + Triplet + clipBio + 0.1 * GPC`
- NTEACH：冻结 B0 起点教师图，`lambda_ndp=0.001`，从第 2 epoch 生效
- 训练关系仅来自 train split；未使用 query/gallery
- 新输出未生成任何 `rank*state.pt`、`rank*state.pth` 或逐 rank checkpoint
- NTEACH 教师图已写入单文件 `resume_latest.pth`，恢复时不再从学生当前特征重建

## LTCC 正式 B0 评估

本次重建 B0 模型 SHA256：`f3f89f4359267699c8d7abd55d843d0f30a3b43d506652acb984f7d3b38b7688`

| 协议 | Rank-1 | mAP |
|---|---:|---:|
| General | 84.381% | 54.186% |
| Clothes-Changing | 55.357% | 33.937% |

与此前记录的 57.14% / 33.42% 不一致；因此两者不是同一可复用模型节点，不能直接混为严格配对基线。

## LTCC 5 epoch 配对结果

两臂均从上述 LTCC B0 60 epoch 模型权重启动，并新建相同 optimizer/scheduler/scaler/RNG。

| 协议 / arm | Rank-1 | mAP |
|---|---:|---:|
| General B0 | 84.381% | 54.431% |
| General NTEACH | 85.598% | 54.101% |
| Clothes-Changing B0 | 54.337% | 33.411% |
| Clothes-Changing NTEACH | 55.357% | 33.015% |

相对配对 B0：LTCC CC Rank-1 `+1.020 pp`，mAP `-0.396 pp`；General Rank-1 `+1.217 pp`，mAP `-0.330 pp`。因此尚未得到 Rank-1 与 mAP 同时提升的稳定证据。

## PRCC 当前进度

PRCC B0 目录：`PRCC_B0_full/PRCC/`

- 正式训练：5/60 epoch
- 第 5 epoch：loss `8.204030`，NDP `0`，AMP skip `0`
- 单文件恢复点：`PRCC_B0_full/PRCC/resume_latest.pth`
- PRCC NTEACH：尚未启动，等待 B0 60 epoch 完成

## 结论（阶段性）

NTEACH 教师恢复逻辑已修复并通过 LTCC 训练验证，但 LTCC 短程只表现为 Rank-1 上升、mAP 下降。PRCC 的最终结论必须等待 B0 60 epoch 与同起点 NTEACH 配对完成后再给出；当前禁止宣称 NTEACH 已在两个数据集有效。
