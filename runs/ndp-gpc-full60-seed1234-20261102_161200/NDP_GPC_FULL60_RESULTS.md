# NDP_GPC LTCC / PRCC 正式 60 Epoch 实验报告

本轮 `NDP_GPC` 使用 seed=`1234`，LTCC 与 PRCC 均从项目既有共同初始化完成 60 epoch，并使用仓库官方评估流程。训练和邻域挖掘只读取 train split；官方 query/gallery 只在最终评估阶段使用。正式输出未生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。

## 固定条件与方法

- Bio+GPC 主干完全固定为 `CE + Triplet + clipBio + 0.1 * GPC-v1`。GPC 代码、样本选择、selector/context 和权重 `0.1` 未修改，也没有重新训练已有 B0。
- LTCC 复用 `/data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth`；PRCC 复用 `/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth`。
- NDP_GPC 直接使用当前 batch 中 GPC 实际选出的同身份正样本 `(anchor, positive)`。对每个正样本对，在学生当前归一化特征中排除原始 Batch-Hard 负 PID，再从剩余异身份样本中选取平均相似度最高的近邻负样本。
- 仅当学生当前局部 margin `M = cos(a,p) - 0.5[cos(a,n)+cos(p,n)] <= 0` 时激活，损失为 `ReLU(0.01 - M)`，`lambda_ndp=0.005`。索引选择不回传梯度，`a/p/n` 均由学生当前特征计算，避免教师/学生特征空间不一致。
- 训练使用 PPU/GPU 两进程；恢复和最终保存只使用单文件 `resume_latest.pth`、`model_final.pth`。正式完成后只保留 `model_final.pth`。

## 官方评估结果

差值为 `NDP_GPC - Bio+GPC B0`，单位为百分点；mINP 为仓库评估器不支持。

| 数据集 / 协议 | Bio+GPC Rank-1 | NDP_GPC Rank-1 | ΔRank-1 | Bio+GPC mAP | NDP_GPC mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 83.773 | -0.608 | 54.186 | 52.599 | -1.587 |
| LTCC Clothes-Changing / CC | 55.357 | 52.041 | -3.316 | 33.937 | 29.159 | -4.778 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.493 | +0.015 |
| PRCC C→A | 64.098 | 63.534 | -0.564 | 63.414 | 63.761 | +0.347 |

Rank-5 / Rank-10：LTCC General `89.858 / 92.292`，LTCC CC `65.306 / 71.173`；PRCC Same-Clothes `100.000 / 100.000`，PRCC C→A `70.279 / 74.203`。

最终模型 SHA256：LTCC `219bbee063f5ef4d5675d7f7dfa2621bf0ef6fa6a76507c8a4e9e652747255eb`；PRCC `6348ffe7fa01f286efd13095395c8455445ad1e7ccbcd2bb4c13e429a6fe3d20`。

## 训练状态与有效性

| 数据集 | 完成 epoch | 最终 global step | 总 optimizer updates | AMP skipped | 总选中/激活关系 | 总激活 batch | 最终 NDP loss | 最终 NDP/main grad ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LTCC | 60/60 | 35760 | 35744 | 16 | 100 / 100 | 100 | 0.0000725 | 0.000000 |
| PRCC | 60/60 | 33060 | 33050 | 10 | 1500 / 1500 | 1014 | 0.0003739 | 0.096428 |

LTCC 全程 NDP 关系实际激活 36 个 epoch，最大激活率 3.10%；虽然最终诊断窗口的梯度比为 0，但训练中曾记录非零 NDP feature gradient，最大 NDP/main ratio 为 0.019117。PRCC 60 个 epoch 均有激活，最大激活率 38.57%，最大 NDP/main ratio 为 0.123225。Smoke Test 两个数据集均通过 20 steps：LTCC 选中/激活 `1/1`，PRCC `37/37`，均标记 `no_per_rank_checkpoint=true`。单元测试验证了 NDP_GPC 的负样本排除、margin 计算和梯度非零；仓库测试中除一个已有的缺失模块收集错误外，74 项测试通过。

## 结论

NDP_GPC 已解决“直接把冻结教师关系与学生特征混算”的特征空间问题，并且确实把监督绑定到 GPC 当前选择的同身份正样本关系；PRCC 的 mAP 有 `+0.347 pp`，但 C→A Rank-1 下降 `0.564 pp`。LTCC General 和 CC 的 Rank-1、mAP 均下降，尤其 CC 下降明显。因此本轮没有形成 LTCC/PRCC、Rank-1/mAP 同时可靠的增益，不启动第二个 seed，也不再继续堆叠相似 NDP margin 模块；后续优先完成固定 GPC 的论文实验和复现。

## 日志与代码

- 服务器正式日志：`/home/DIFFER/log_NDP/ndp_gpc_full60_seed1234_20261102_161200/`
- GitHub 归档：`runs/ndp-gpc-full60-seed1234-20261102_161200/`
- 归档仅包含 `train.log`、`metrics_per_epoch.csv`、`online_neighbor_summary.csv`、`RUN_COMPLETE.json`、`final_model.json`、`eval.log`、`eval_final.json` 和 smoke 输出，不包含约 1.2GB 的 `model_final.pth`。
