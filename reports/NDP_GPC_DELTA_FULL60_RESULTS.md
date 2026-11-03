# NDP_GPC_DELTA 正式实验报告

日期：2026-11-04  
Seed：`1234`  
Arm：`NDP_GPC_DELTA`  
状态：LTCC、PRCC 均完成正式 60/60 epoch，并完成官方评估。

## 方法

本轮没有修改 GPC。基础目标仍为 `CE + Triplet + clipBio + 0.1 * GPC-v1`，GPC 的代码、正样本选择、matched-budget 选择器和权重 `0.1` 均保持不变。

`NDP_GPC_DELTA` 只使用 train split：对 GPC 实际选中的同身份关系，在当前学生特征中寻找最近的异身份负样本；在真实 optimizer step 前后重新计算同一局部 identity margin。只有 margin 真实下降的关系才写入 train-only 风险表，并以稳定的 GPC `state_key` 跨 batch/epoch 复用。关系再次出现时，用该关系自身的 pre-step margin 作保护目标，学生当前 anchor、正端点和负端点均保持可反向传播。NDP 系数固定为 `0.001`，没有进行测试集选参或大规模搜索。

## 正式结果

差值为新方法减固定 Bio+GPC，单位为百分点。

| 数据集 / 协议 | Bio+GPC Rank-1 | NDP_GPC_DELTA Rank-1 | ΔRank-1 | Bio+GPC mAP | NDP_GPC_DELTA mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 84.179 | -0.203 | 54.186 | 54.899 | +0.713 |
| LTCC CC | 55.357 | 54.592 | -0.765 | 33.937 | 33.033 | -0.904 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.487 | +0.008 |
| PRCC C→A | 64.098 | 63.872 | -0.226 | 63.414 | 63.836 | +0.422 |

## 训练状态与归因审计

- LTCC：60/60 epoch，最终 loss `5.198964`；最后 epoch `35` 个 active batch、`36` 个 active relation，未乘系数的 NDP/主干特征梯度比约 `0.269`。
- PRCC：60/60 epoch，最终 loss `7.180637`；最后 epoch `549`/`551` active batch、`5416` 个 active relation，未乘系数的 NDP/主干特征梯度比约 `0.370`。
- 两套正式运行均为 PPU 双进程、seed `1234`，训练过程中未出现非有限 loss；AMP skip 均为常规可恢复状态。
- `final_model.json` 明确记录 `lambda_gpc=0.1`、`lambda_ndp=0.001`、`train_only_reference=true`、`query_gallery_used_during_training=false`，且 `per_rank_checkpoint_files_created=false`。
- 训练过程中只使用单文件 `resume_latest.pth` 恢复，完成后保存 `model_final.pth`；本仓库归档不包含约 1.2 GB 的模型文件。未生成 `rank*state.pt/pth` 或逐 Rank checkpoint。

此前 LTCC `NDP_GPC_IMPACT` 的结构性归因也已完成核对：同一 LTCC 初始化张量哈希为 `c593b713...cc83202`，GPC 正式源 SHA256 为 `69ce0b48...68156`。用相同训练协议完成的 `lambda_NDP=0` 60 epoch 对照，LTCC CC Rank-1 为 `55.357`，与固定 Bio+GPC 一致；因此此前 NDP_GPC_IMPACT 的 LTCC CC Rank-1 上升不能归因于 NDP。新方法的 LTCC CC 结果也没有形成增益，PRCC C→A 只表现为 mAP 上升而 Rank-1 下降，不能宣称跨数据集稳定提升。

## 结论

`NDP_GPC_DELTA` 成功解决了旧风险门控“几乎不触发”和前向诊断断图的问题：它确实在 LTCC/PRCC 以不同密度激活，且学生端梯度有效。但正式指标没有同时改善 LTCC 或 PRCC 的换装 Rank-1 与 mAP。因此本轮应作为真实的负结果和归因审计材料保留，不再继续堆叠相似 NDP Margin 变体；后续优先完成固定 GPC 的论文实验与复现。

正式小体积归档：[`runs/ndp-gpc-delta-full60-seed1234-20261103`](../runs/ndp-gpc-delta-full60-seed1234-20261103)  
验证与 lambda=0 对照归档：[`runs/ndp-gpc-delta-validation-20261103`](../runs/ndp-gpc-delta-validation-20261103)
