# NDP_GPC_IMPACT LTCC / PRCC 正式 60 Epoch 实验报告

本轮 `NDP_GPC_IMPACT` 使用 seed=`1234`，LTCC 与 PRCC 均从既有 Bio+GPC 共同初始化开始完成 60 epoch。训练与关系挖掘只使用 train split，官方 query/gallery 仅用于最终评估。GPC 保持原实现、原样本选择与固定权重 `0.1` 不变。

## 方法

对当前 batch 中 GPC 实际选出的同身份正样本对 `(a,p)`，在学生当前归一化特征空间中选择平均相似度最高的异身份样本 `n`；不再排除 Batch-Hard 负 PID。计算当前局部 margin：

`M = cos(a,p) - 0.5 * [cos(a,n) + cos(p,n)]`。

同时计算 GPC loss 对当前学生特征的梯度，并用局部 margin 梯度与 GPC 下降方向的内积估计一阶影响。只有当该影响为负、即 GPC 更新会使该局部 margin 变差时，才施加 `ReLU(0.01 - M)`，NDP 权重固定为 `0.001`。监督只回传当前学生的 `a/p/n` 特征，不使用冻结教师特征，也不使用测试集关系。

## 官方结果

差值为 `NDP_GPC_IMPACT - Bio+GPC`，单位为百分点。

| 数据集 / 协议 | Bio+GPC Rank-1 | 新方法 Rank-1 | ΔRank-1 | Bio+GPC mAP | 新方法 mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 85.801 | +1.420 | 54.186 | 54.792 | +0.606 |
| LTCC CC | 55.357 | 57.398 | +2.041 | 33.937 | 32.620 | -1.317 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.496 | +0.016 |
| PRCC C→A | 64.098 | 64.268 | +0.170 | 63.414 | 64.253 | +0.839 |

结论：PRCC C→A 的 Rank-1 和 mAP 同时小幅改善；LTCC Rank-1 改善明显，但 CC mAP 下降，因此不能宣称 LTCC/PRCC、Rank-1/mAP 全面稳定增益。

## 训练状态

| 数据集 | Epoch | Global step | Optimizer updates | AMP skipped | 最终 NDP loss | 最高 NDP/main 梯度比 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| LTCC | 60/60 | 35760 | 35746 | 14 | 0.000000 | 0.000000 | 仅第 14 epoch 出现极少量非零 raw NDP，计数因双 rank 均值取整显示为 0 |
| PRCC | 60/60 | 33060 | 33048 | 12 | 0.000142 | 0.047956 | 51/60 epoch 有非零 raw NDP，激活率峰值 43.83% 后收缩到个位数关系 |

严格配对控制位于 `/home/DIFFER/log_NDP/ndp_gpc_paired_control_20261103_021427/`：LTCC、PRCC 的 B0、NDP 权重 0、正常 NDP 均完成 20-step PPU smoke，基础损失路径、AMP 和优化器无异常。候选 train-only smoke 位于 `/home/DIFFER/log_NDP/ndp_gpc_impact_smoke_20261103_ppuenv/`：LTCC 0 条关系激活，PRCC 选中 39 条、激活 23 条，NDP/main 特征梯度比约 1.67%。

正式输出仅保留每个数据集的 `model_final.pth`、训练 CSV/日志和 JSON 元数据；没有生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。最终模型 SHA256：

- LTCC：`8945f4fe44f1f97755077cf736226d4f57c38aa1acf770b70259372a849054fe`
- PRCC：`58026a2edd134cd1b6ab5d476cdc6f65c62d2b0f7c100c63a0ba1887dece4475`

## 后续结论

本轮已经验证了“GPC 实际影响驱动、低干扰、当前学生空间”的方向，但没有取得两个数据集所有核心指标同时改善的稳定证据。按实验约束，不再进行第二个 seed 或继续堆叠相似 Margin Loss；后续优先整理固定 GPC 的论文实验与复现结果。

正式日志目录：`/home/DIFFER/log_NDP/ndp_gpc_impact_full60_20261103/`。
