# NTEACH / NTEACH_SAFE LTCC 与 PRCC 配对续训结果

## 结论

本轮完成了 PRCC B0 60 epoch 官方评估、PRCC B0/NTEACH 五轮配对续训，以及 LTCC/PRCC 的 NTEACH_SAFE 五轮短程验证。NTEACH_SAFE 没有同时改善两个数据集的 Rank-1 和 mAP，因此不启动正式 NTEACH_SAFE 60 epoch。

GPC-v1、权重 0.1、EVA02、CE、Triplet、Bio、采样器和评估协议未修改。所有关系图只使用 train split；未使用 query/gallery 训练或选参。

## 运行与代码

- 实验目录：`/home/DIFFER/log_NDP/ndp_nteach_safe_pair_20261031_180000/`
- GitHub commit：`acb1d9d`（Add reliable-positive NTEACH gate）
- `ndp_v5_prcc_runner.py` SHA256：`d143e28d0f056269ae3d805af2e49f61fa2fda7b60824e2a0edb8c09f0b7e08e`
- `train_ndp_listwise_full60.py` SHA256：`6e5b6b715c00f767851254b87c21df98d780763fefddc2b6152ee085da046ecf`
- NTEACH_SAFE 唯一改动：在已有冻结教师、teacher margin 和当前负 margin 门控外，要求 `safe_positive_reliable`，即正样本端点由至少两个不同 clothes ID 支持。门控 detached，当前 anchor 仍保留梯度。
- NTEACH 教师图继续写入单文件 `resume_latest.pth`；没有生成逐 Rank 状态文件。

## 官方评估结果

百分比单位；PRCC 主要看 C→A，LTCC 主要看 Clothes-Changing（CC）。五轮 B0 与 NTEACH/NTEACH_SAFE 均从对应数据集已完成 60 epoch B0 模型权重开始，B0/NDP 使用同一初始化模型、学习率和 GPC phase。该短程续训是 model-only warm-start 新片段，不冒充恢复历史 optimizer 状态。

| 数据集/协议 | 起点 B0 60 epoch | 五轮 B0 | 五轮 NTEACH | NTEACH-B0 | 五轮 NTEACH_SAFE | NTEACH_SAFE-B0 |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General Rank-1 | 84.381 | 84.381 | 85.598 | +1.217 | 83.367 | -1.014 |
| LTCC General mAP | 54.186 | 54.431 | 54.101 | -0.330 | 54.193 | -0.238 |
| LTCC CC Rank-1 | 55.357 | 54.337 | 55.357 | +1.020 | 55.612 | +1.276 |
| LTCC CC mAP | 33.937 | 33.411 | 33.015 | -0.396 | 33.133 | -0.278 |
| PRCC Same-Clothes Rank-1 | 100.000 | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 |
| PRCC Same-Clothes mAP | 99.479 | 99.426 | 99.347 | -0.079 | 99.447 | +0.021 |
| PRCC C→A Rank-1 | 64.098 | 64.014 | 63.731 | -0.282 | 63.506 | -0.508 |
| PRCC C→A mAP | 63.414 | 62.829 | 62.891 | +0.062 | 62.876 | +0.047 |

## 训练侧证据

- LTCC NTEACH_SAFE：每轮约 353–407 个有效关系，active batch 约 40.5%–45.1%；NDP/main 特征梯度范数比约 0.057–0.117；第 1、2 轮各 1 个 AMP skip，其余无 skip。
- PRCC NTEACH_SAFE：每轮 145–161 个有效关系，active batch 约 21.0%–23.8%；NDP/main 梯度范数比约 0.042–0.668；第 5 轮 1 个 AMP skip，其余无 skip。
- NDP loss 均有限且非零，NTEACH_SAFE 确实参与训练；B0 的 NDP loss、梯度和激活均为 0。
- LTCC train-only 图规模为 9,576 anchor / 28,728 关系；PRCC 为 17,896 anchor / 53,688 关系。参考图摘要中的 `query_gallery_used=False`。

## 判定

1. PRCC B0 60 epoch 已完成并完成 Same-Clothes 与 C→A 评估；此前“未评估”问题已补齐。
2. 原始 NTEACH 在 LTCC 保持 CC Rank-1 提升但 mAP 下降，在 PRCC C→A Rank-1 下降；不能称为稳定增益。
3. NTEACH_SAFE 略微减轻 LTCC mAP 损失，但仍为负；PRCC Rank-1 进一步下降。可靠正样本门控没有产生跨数据集增量价值。
4. 由于没有同时改善 Rank-1 和 mAP，不继续 NTEACH_SAFE 60 epoch，也不继续扫描静态阈值或权重。

## 文件安全与未完成事项

- 新实验目录中未发现 `rank*state.pt` 或 `rank*state.pth`。
- 使用了 `resume_latest.pth` 与 `model_final.pth`；NTEACH 教师图保存在单文件恢复点中。
- 本轮未运行 NTEACH/NTEACH_SAFE 正式 60 epoch；原因是五轮配对结果未达到继续训练门槛。
- PPU/A-A 既有非确定性限制仍然适用；小幅 mAP 变化不视为已证实有效。
