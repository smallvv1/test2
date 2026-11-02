# NDP_LATE LTCC / PRCC 正式 60 Epoch 结果

本报告对应 `NDP_LATE` 正式实验：seed=`1234`，LTCC 与 PRCC 均完成 60 epoch，训练和邻域参考只使用 train split。Smoke Test 和正式训练均使用 GPU/PPU；评估只在最终阶段使用官方 query/gallery。正式输出没有生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。

## 方法与公平性

基础 Bio+GPC 完全固定：`CE + Triplet + clipBio + 0.1 * GPC-v1`。未修改 GPC 代码、样本选择或权重，`lambda_gpc=0.1`。两套实验复用既有共同初始化和已经完成的 Bio+GPC 教师，没有重新训练 B0。

NDP_LATE 只在 NDP 分支中增加了最小的延迟和风险门控：

- epoch 1–15 完全关闭 NDP；从 epoch 16 开始激活。
- 冻结教师只选择 train-only 的可靠局部关系，教师 margin gate 为 `[0.02, 0.10]`。
- 正端点、负端点和参考特征每 5 epoch 用当前学生特征刷新；学生用自身 anchor、正端点和负端点计算 margin，避免跨特征空间比较。
- 只有当前学生 margin `<=0` 的真实身份混淆关系才施加 `ReLU(0.01 - M_student)`；每个 anchor 只保留最危险的一条关系；`lambda_ndp=0.005`。

固定输入：LTCC common init 为 `/data/DIFFER/outputs/new_method_track/phase5b_id_damage_ds/common_init/COMMON_PHASE5B_INIT.pth`，PRCC common init 为 `/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/PRCC_COMMON_INIT.pth`。冻结教师分别复用 LTCC `/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/model_final.pth` 和 PRCC `/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/model_final.pth`。

## 官方评估结果

差值为 `NDP_LATE - Bio+GPC B0`，单位为百分点；mINP 为仓库评估器不支持。

| 数据集 / 协议 | Bio+GPC Rank-1 | NDP_LATE Rank-1 | ΔRank-1 | Bio+GPC mAP | NDP_LATE mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 84.179 | -0.203 | 54.186 | 54.333 | +0.147 |
| LTCC Clothes-Changing | 55.357 | 55.612 | **+0.255** | 33.937 | 32.706 | -1.231 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.452 | -0.027 |
| PRCC C→A | 64.098 | 64.126 | +0.028 | 63.414 | 63.374 | -0.040 |

Rank-5/Rank-10：LTCC General `90.264/92.292`，LTCC Clothes-Changing `69.898/75.255`；PRCC Same-Clothes `100.000/100.000`，PRCC C→A `70.731/74.118`。

最终模型 SHA256：LTCC `a96b8cb5f3b6265161d2a2050bd563d4a9d1806e397affd33081805cbfdbfcee`；PRCC `ea45d6a44337933cdf53c2471a49a06aa31544057e5acf1fc8fa222809914e47`。

## 训练有效性

前 15 epoch 的 `ndp_loss` 和 NDP lambda 均为 0。epoch 16 后 NDP 正常激活并逐步变稀疏：

| 数据集 | epoch 16 激活 batch / 总 batch | epoch 60 激活 batch / 总 batch | epoch 60 选中关系 | epoch 60 NDP loss | NDP/main feature-grad ratio | AMP skipped |
|---|---:|---:|---:|---:|---:|---:|
| LTCC | 497/595 | 325/596 | 466 | 0.023077 | 0.091829 | 15 |
| PRCC | 124/551 | 55/551 | 62 | 0.004062 | 0.045217 | 11 |

Smoke Test（LTCC/PRCC 各 20 steps）均通过，学生参考特征均为 train-only bank，shape 分别为 `(28728,1024)` 和 `(53688,1024)`；正式训练均完成 `60/60`，无非有限 loss 或异常终止。

## 结论

NDP_LATE 确实修复了 NDP_RANK 的跨特征空间问题，也解决了训练早期介入过多的问题；但正式结果没有形成跨 LTCC/PRCC、Rank-1/mAP 同时可靠的增益。LTCC Clothes-Changing Rank-1 小幅上升，但 mAP 明显下降；PRCC C→A 只有接近零的 Rank-1 变化，mAP 反而略降。因此不再继续堆叠相似 NDP 模块，后续优先完成固定 GPC 的论文实验和复现；不启动第二个 seed 或 NDP 消融。

## 代码与日志

- 正式日志目录：`/home/DIFFER/log_NDP/ndp_late_full60_seed1234_20261102_014655/`
- GitHub 小体积归档：`runs/ndp-late-full60-seed1234-20261102_014655/`
- 归档只包含日志、CSV、JSON 和 smoke 输出，不包含约 1.2GB 的 `model_final.pth`。
- 本轮代码只扩展 `NDP_LATE` 分支；运行时 `/home/DIFFER/scripts` 与仓库实现保持一致，GPC 原代码不变。
