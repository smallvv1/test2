# NDP-Teacher（NTEACH）LTCC / PRCC 短程配对结果

## 结论

本轮完成 seed=1234 的 LTCC、PRCC 各自 B0/NTEACH 五个 raw epoch 训练和固定官方评估。NTEACH 真实产生了有限 NDP 梯度，且没有修改 GPC-v1；但结果是跨数据集不一致：LTCC Clothes-Changing 有明显正向变化，PRCC C→A 反而下降。因此当前不启动 LTCC/PRCC 正式 60 epoch，也不把 NTEACH 宣称为已验证有效。

## NTEACH 定义

NTEACH 在训练开始前，用同一实验初始化的 Bio+GPC B0 模型提取 train-only 特征并只构建一次固定教师邻域。教师 margin 至少为 0.02 的关系才作为可靠排序；训练中仅当当前实时 anchor 对该固定负邻居的 margin 进入 `(-0.05, 0)` 时激活，目标为 `min(0.05, max(0.01, 0.25 * teacher_margin))`。NDP 权重固定为 0.001，从第 2 轮开始生效。参考端点停止梯度，当前 anchor 保留梯度。

这不是全量历史 margin 恢复，也不随当前训练重选关系；目标是只修复已发生的轻度身份混淆。

固定不变：`CE + Triplet + clipBio + 0.1 * GPC`、EVA02、GPC-v1 选择和损失、sampler、优化器、评估协议。训练关系只来自 train split，未使用 query/gallery。

## 官方评估结果

以下 B0 与 NTEACH 从相同的已完成 Bio+GPC 模型权重 warm-start，并新建相同 optimizer/scheduler/scaler/RNG；因此是模型权重起点配对，不是旧 60 epoch optimizer 状态的无缝续训。百分比单位。

| 数据集 / 协议 | B0 Rank-1 | B0 mAP | NTEACH Rank-1 | NTEACH mAP | Δ Rank-1 (pp) | Δ mAP (pp) |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.976 | 52.531 | 83.773 | 52.813 | -0.203 | +0.283 |
| LTCC Clothes-Changing | 47.959 | 28.714 | 51.531 | 29.454 | +3.571 | +0.740 |
| PRCC Same-Clothes | 100.000 | 99.167 | 100.000 | 99.196 | +0.000 | +0.030 |
| PRCC C→A | 64.437 | 63.104 | 64.268 | 62.840 | -0.169 | -0.265 |

评估器不提供 mINP，均记录为 `Unavailable`。用户给出的正式 60 epoch B0（LTCC CC 57.14/33.42、PRCC C→A 64.35/63.91）与本轮 warm-start 后短程控制不是同一个评估节点，不能直接将上表的短程 B0 数字与正式基线混为严格比较；上表只用于同起点 NTEACH/B0 差值。

## 训练统计

| 数据集 / arm | 最后一轮 raw NDP loss | 最后一轮 selected/active | active batch 比例 | 总 AMP skip | effective updates |
|---|---:|---:|---:|---:|---:|
| LTCC B0 | 0 | 0/0 | 0 | 3 | 2977/2980 |
| LTCC NTEACH | 0.014784 | 392/392 | 44.46% | 2 | 2978/2980 |
| PRCC B0 | 0 | 0/0 | 0 | 1 | 2754/2755 |
| PRCC NTEACH | 0.013320 | 177/177 | 25.05% | 0 | 2755/2755 |

最后一轮 NDP 特征梯度 / 主损失特征梯度比约为 LTCC `0.0680`、PRCC `0.1054`；没有观察到 NDP 梯度压过主损失。LTCC 在线教师图为 28,728 条关系、9,576 个有效 anchor；PRCC 为 53,688 条关系、17,896 个有效 anchor。教师图只在 epoch 1 前捕获一次，后续没有重选。

## 运行与完成状态

- 代码语法检查：`/usr/local/bin/python3.10 -m py_compile scripts/ndp_safe.py scripts/ndp_v5_prcc_runner.py scripts/train_ndp_listwise_full60.py` 通过。
- LTCC/PRCC B0 与 NTEACH 均生成 `SHORT_RUN_COMPLETE.json`，五轮训练实际完成。
- 正式 60 epoch 未运行，因此没有 `RUN_COMPLETE.json`。
- 四个新输出目录均未生成 `rank*state.pt`、`rank*state.pth` 或其他逐 Rank checkpoint。
- 单文件 `resume_latest.pth` 机制保留；短程结束时未将短程结果伪装成正式完成。

## 文件摘要

- LTCC B0：`LTCC_B0/LTCC/model_final.pth`，SHA256 `a075644bad3f88a4d11bbb78a2a555c19e4ee151220351ca9eaf302b01f90bc8`。
- LTCC NTEACH：`LTCC_NTEACH/LTCC/model_final.pth`，SHA256 `0354e12dc435f19362e215e32b71ff46308066aaf7deca57f18c2138a8f1691a`。
- PRCC B0：`PRCC_B0/PRCC/model_final.pth`，SHA256 `e05cc0198d96272e6923bcc0a18227bb0d40f23b79f09464f8e0f97b041ae0fc`。
- PRCC NTEACH：`PRCC_NTEACH/PRCC/model_final.pth`，SHA256 `eb9eee87d53ebc096f9099954024548f6ce6b9c4ff5b4b73434625dbd3b30400`。
- `scripts/ndp_v5_prcc_runner.py` SHA256 `29f9a35ea7ef47d849a41df552359eef6677a59ddcb7221299e85c152ba417d2`。
- `scripts/train_ndp_listwise_full60.py` SHA256 `179d3207ee74af317b74d200b75f15c296352ff636a8a9228c823be434e1dbe1`。
- `scripts/ndp_safe.py` SHA256 `65d96ed9415db9c3c210410b21d6ee398446d91eaf02f437b5124fc8230f8f09`。

## 决策

**短程门禁：混合结果，NO-GO 扩展正式 60 epoch。** LTCC 的 +3.571 pp CC Rank-1 值得保留为线索，但 PRCC C→A 同时下降 `0.169 pp Rank-1 / 0.265 pp mAP`，尚不能证明冻结教师约束具有跨数据集增量价值。下一步不继续扫描 margin 或 lambda；若继续研究，应先解释 LTCC/PRCC 的正样本可靠性差异，或停止关系保持分支。
