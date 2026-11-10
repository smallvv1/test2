# DIFFER + GPC + NDP：论文收尾审计与实验章节草稿

日期：2026-11-10  
正式 seed：`1234`  
评估：仓库统一 evaluator，LTCC CC / PRCC C→A 为主要换装协议  
GPC：代码、样本选择和权重 `0.1` 固定不变

## 1. 论文证据结论

目前已经具备一套可以投稿前整理的 GPC 主实验和 NDP 单 seed 消融证据，但还不适合把 NDP 的小幅变化表述为稳定、严格因果的普遍增益。

- GPC 的 Bio → Bio+GPC 主结果完整，LTCC CC 的 Rank-1/mAP 同时提升，PRCC C→A 的 mAP 提升，评估协议一致。
- `NDP_SHN_RELIABLE_RISK` 已完成 LTCC、PRCC 各 60 epoch，使用相同 seed、正式共同初始化、GPC 权重和 train-only 邻域机制，结果可作为完整候选和消融。
- RISK 在 LTCC CC 为 `55.61/34.22`，相对 Bio+GPC 为 `+0.26/+0.28` pp；但 Query 级净增加的 Rank-1 正确查询只有 1 个。
- RISK 在 PRCC C→A 为 `64.04/63.88`，相对 Bio+GPC 为 `-0.06/+0.47` pp；Rank-1 正确查询反而净减少 2 个。
- 因此，RISK 的 mAP 改善可以作为“局部排序质量改善”的证据，但不能宣称对两个数据集的 Rank-1 都有稳定贡献。

### 真正缺少的关键实验

现有正式 B0 已经是 NDP=0 的强匹配控制：seed=1234、60 epoch、相同正式共同初始化、相同步数、`lambda_gpc=0.1`、train-only reference、无逐 Rank checkpoint。它足以支撑主表中固定 Bio+GPC 对照。

但若论文要写“RISK 的提升严格由 NDP 导致”，仍缺一项更严格的控制：在当前提交代码、当前同一启动协议下，使用 `NDP_SHN_RELIABLE_RISK` 分支但将 `lambda_ndp=0`，LTCC/PRCC 各完成 60 epoch，并记录当前代码哈希、初始化张量哈希、采样计划、AMP 与优化器状态。

现有 B0 与 RISK 的历史日志不是逐步 bit-identical replay：例如 LTCC 第 1 epoch loss 为 `7.663791` vs `7.669555`，PRCC 为 `8.995804` vs `8.995719`；这可能来自不同运行时间的 PPU/数据加载非确定性，也可能包含运行代码快照差异。故本报告将 B0 称为“固定匹配基线”，不把 RISK 的微小提升写成强因果证明。

除上述严格 NDP=0 控制外，不建议再补新的 NDP 变体、权重扫描或多 Seed GPC 实验。

## 2. Bio → Bio+GPC → Bio+GPC+NDP 主结果

格式为 Rank-1 / mAP，差值单位为百分点（pp）。本表的 NDP 候选采用完整的 `NDP_SHN_RELIABLE_RISK`。

| 协议 | Bio | Bio+GPC | Bio+GPC+NDP | GPC 提升 | NDP 提升 | 总提升 |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.57 / 52.48 | 84.38 / 54.19 | 84.58 / 54.59 | +0.81 / +1.71 | +0.20 / +0.40 | +1.01 / +2.11 |
| LTCC Clothes-Changing | 48.98 / 29.17 | 55.36 / 33.94 | 55.61 / 34.22 | +6.38 / +4.77 | +0.26 / +0.28 | +6.63 / +5.05 |
| PRCC Same-Clothes | 100.00 / 99.21 | 100.00 / 99.48 | 100.00 / 99.57 | +0.00 / +0.27 | +0.00 / +0.09 | +0.00 / +0.36 |
| PRCC Clothes-Changing C→A | 64.61 / 63.00 | 64.10 / 63.41 | 64.04 / 63.88 | -0.51 / +0.41 | -0.06 / +0.47 | -0.56 / +0.88 |

精确的 RISK 结果：LTCC General `84.5841765/54.5899803`，LTCC CC `55.6122422/34.2189425`；PRCC Same-Clothes `100.0000000/99.5667005`，PRCC C→A `64.0417755/63.8848221`。

## 3. NDP 消融汇总

各单元格为“结果（相对固定 Bio+GPC 的 Rank-1/mAP 变化）”。全部为已有 seed=1234 正式 60 epoch 结果，不涉及新的参数搜索。

| NDP 版本 | LTCC General | LTCC CC | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| SHN_MP | 86.00/55.33（+1.62/+1.15） | 57.40/33.41（+2.04/-0.52） | 100.00/99.52（+0.00/+0.05） | 64.15/63.78（+0.06/+0.36） |
| SHN_RELIABLE | 84.38/54.05（+0.00/-0.14） | 55.87/33.71（+0.51/-0.22） | 100.00/99.50（+0.00/+0.02） | 64.47/63.87（+0.37/+0.46） |
| SHN_RELIABLE_RISK | 84.58/54.59（+0.20/+0.40） | 55.61/34.22（+0.26/+0.28） | 100.00/99.57（+0.00/+0.09） | 64.04/63.88（-0.06/+0.47） |
| SHN_RELIABLE_ANCHOR | 84.18/52.14（-0.20/-2.05） | 50.51/28.50（-4.85/-5.44） | 100.00/99.55（+0.00/+0.07） | 63.93/63.76（-0.17/+0.35） |
| SHN_BANK | 85.19/55.05（+0.81/+0.87） | 56.38/32.48（+1.02/-1.46） | 100.00/99.43（+0.00/-0.05） | 63.34/63.59（-0.76/+0.17） |
| SHN λ=0.005 | 85.80/52.59（+1.42/-1.60） | 51.53/28.88（-3.83/-5.05） | 100.00/99.41（+0.00/-0.06） | 64.80/63.73（+0.71/+0.31） |

消融说明：SHN_MP 的 LTCC CC Rank-1 最高，但 mAP 低于 GPC；RISK 的 LTCC CC mAP 最高，说明它改善了部分查询的整体排序，却没有显著扩大首位正确集合；RELIABLE 的 PRCC C→A Rank-1 最高；Anchor-view 修正明显退化，支持停止继续堆叠相似 Bank/Margin 分支。

## 4. Query 级检索分析

分析脚本复用了官方 evaluator 的距离、摄像头过滤和衣服过滤规则，只对已完成模型做事后审计，不用于选参。详细 CSV 保存在：

`/home/DIFFER/log_NDP/paper_audit_20261110/{LTCC,PRCC}/query_metrics.csv`

### LTCC Clothes-Changing

官方查询 493 个，其中 392 个有有效跨衣正样本，Rank-1 分母为这 392 个查询。

| 模型 | Rank-1 正确数 / 有效查询 | Rank-1 | mAP | 相对 Bio+GPC：Rank-1 gained/lost | AP 上升/下降/不变 | 平均 AP 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Bio+GPC | 217 / 392 | 55.36 | 33.94 | — | — | — |
| SHN_MP | 225 / 392 | 57.40 | 33.41 | 29 / 21 | 183 / 209 / 0 | -0.525 pp |
| SHN_RELIABLE | 219 / 392 | 55.87 | 33.71 | 20 / 18 | 191 / 201 / 0 | -0.224 pp |
| SHN_RELIABLE_RISK | 218 / 392 | 55.61 | 34.22 | 15 / 14 | 202 / 189 / 1 | +0.282 pp |

RISK 的 mAP 增益来自较分散的 AP 重排：改善 202 个查询、退化 189 个查询，Rank-1 仅净增加 1 个查询。因此它是有可复核的排序改善，但稳定性证据较弱。

### PRCC Clothes-Changing C→A

3543 个查询全部有有效正样本。

| 模型 | Rank-1 正确数 / 有效查询 | Rank-1 | mAP | 相对 Bio+GPC：Rank-1 gained/lost | AP 上升/下降/不变 | 平均 AP 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Bio+GPC | 2271 / 3543 | 64.10 | 63.41 | — | — | — |
| SHN_MP | 2273 / 3543 | 64.15 | 63.78 | 60 / 58 | 1585 / 1350 / 608 | +0.363 pp |
| SHN_RELIABLE | 2284 / 3543 | 64.47 | 63.87 | 72 / 59 | 1657 / 1245 / 641 | +0.456 pp |
| SHN_RELIABLE_RISK | 2269 / 3543 | 64.04 | 63.88 | 71 / 73 | 1681 / 1243 / 619 | +0.471 pp |

RISK 的 PRCC mAP 增益同样主要来自大量查询的排序变化，而不是首位命中改善；Rank-1 净减少 2 个查询。由此可见，RISK 更适合描述为“弱局部排序正则”，而不是可靠的 Rank-1 提升模块。

## 5. 局部风险与可靠性筛选证据

RISK 的 train-only 选择器包括三层约束：

1. Anchor 与 bank 正样本的实时相似度必须不明显低于同 batch 正样本均值（容差 `0.05`）。
2. bank 正样本与自身 PID 中心的支持度至少为 `0.55`，排除明显不可靠的冻结样本。
3. 局部风险定义为 `sim(Anchor, bank-positive) - max sim(Anchor, current negative identity center)`；在可靠候选中采用 40% 分位附近的中等风险关系，避免单独强推最困难关系；最多两个 Anchor/PID 不重复的关系。

正式日志显示：

- LTCC/PRCC 在第 3 epoch 后分别有 58/58 个可激活 epoch 使用 NDP；平均 active rate 为 `99.54%/99.96%`。
- LTCC 平均每 epoch 为 4745 条实时关系 + 1186 条 bank 辅助关系；PRCC 为 4407 条实时关系 + 1102 条 bank 辅助关系。bank 辅助约占总 NDP 关系 20%，并未替代实时 SHN 主损失。
- 第 3–60 epoch 的 bank 辅助可靠性均值为 LTCC `0.854`、PRCC `0.876`；最后 epoch 为 `0.771/0.823`，说明可靠性门控确实筛掉了一部分低支持样本，但不能保证筛选关系对测试检索有益。
- 第 3–60 epoch 的 NDP/main feature-gradient ratio 均值为 LTCC `0.177`、PRCC `0.265`；最后 epoch 为 `0.243/0.363`。NDP 梯度有效且没有“损失未激活”的归因问题。
- AMP skip 总数为 LTCC `14`、PRCC `13`，没有 NaN 或异常中断；训练和参考库刷新均只使用 train split。

这些结果证明“关系被筛选并反向传播”成立，但 Query 级结果表明“被筛选关系一定有测试身份判别价值”并不成立，这正是 NDP 增益不稳定的原因之一。

## 6. 训练和推理开销

### 训练

依据正式 `train.log` 的首尾 epoch 时间：

| 数据集 | Bio+GPC B0 | SHN_RELIABLE_RISK | 额外时间 | 相对开销 |
|---|---:|---:|---:|---:|
| LTCC | 2.245 h | 5.514 h | +3.269 h | +145.6% |
| PRCC | 2.202 h | 4.450 h | +2.248 h | +102.1% |

主要开销来自每 5 个 epoch 的 train-only 特征库刷新，以及每个 batch 的局部关系和中心计算；RISK 没有新增可学习网络参数。

### 推理

NDP 只存在于训练 loss 和 train-only bank，最终模型的 forward 结构与 Bio+GPC 完全相同，因此没有额外的推理参数、FLOPs 或测试时邻域检索。PPU 统一评估中，LTCC 单模型特征提取约 51 秒、PRCC C→A 约 48 秒，四个模型的评估耗时基本一致；检索矩阵和官方评估开销也不变。

## 7. 方法总结、失败实验与局限

### 可写入论文的方法总结

GPC 提供冻结协议下的同身份关系一致性约束；NDP_SHN_RELIABLE_RISK 在不改变 GPC 的前提下，使用训练集 bank 提供少量跨衣同身份正样本，并以当前 Anchor 到最近负身份中心的局部 margin 衡量潜在混淆，再通过身份中心可靠性筛选和稀疏关系约束减少噪声传播。该设计的核心是“实时 SHN 主损失 + 低权重可靠 bank 辅助”，而不是额外网络或测试时模块。

### 应如实保留的失败证据

- 早期静态 Margin、教师/学生跨空间关系、LATE、GPC-耦合、DELTA_FIX、特征库等方案没有稳定提升。
- SHN_BANK 的跨 batch 正样本直接混合导致 LTCC mAP 下降、PRCC Rank-1 下降。
- NDP 权重 `0.005` 在 PRCC 略有收益，但 LTCC 明显退化；因此不应继续扫描权重。
- Anchor-view 修正后的 `NDP_SHN_RELIABLE_ANCHOR` 在 LTCC CC 退化至 `50.51/28.50`，证明更强的局部边界筛选并不自动带来有效监督。

### 局限性

- 所有 NDP 正式结果为单 seed=1234，不能给出跨 seed 方差。
- RISK 与已有 B0 的初始化、步数和 GPC 协议匹配，但缺少当前代码同一运行快照下的完整 `lambda_ndp=0` 配对控制；强因果 NDP 结论应暂缓。
- 当前 evaluator 不输出 mINP；论文应只报告 Rank-1/mAP，或另行实现并固定 mINP 评估协议。
- NDP 对训练时间增加约 2–2.5 倍，却没有稳定改善 Rank-1；从工程角度不应把它作为 GPC 的必要组成。

## 8. 复现与归档

- Query 审计脚本：[scripts/paper_query_audit.py](/home/DIFFER/github_latest_test2_lXaBlM/scripts/paper_query_audit.py)
- Query 审计日志：`/home/DIFFER/log_NDP/paper_audit_20261110/`
- RISK 正式日志：`/home/DIFFER/log_NDP/ndp_shn_reliable_risk_full60_seed1234_20261108/`
- RISK 正式模型：上述目录下 `LTCC/model_final.pth`、`PRCC/model_final.pth`
- RISK 正式报告：[NDP_SHN_RELIABLE_RISK_FULL60_RESULTS_20261109.md](/home/DIFFER/github_latest_test2_lXaBlM/reports/NDP_SHN_RELIABLE_RISK_FULL60_RESULTS_20261109.md)
- Anchor 失败实验报告：[NDP_SHN_RELIABLE_ANCHOR_FULL60_RESULTS_20261109.md](/home/DIFFER/github_latest_test2_lXaBlM/reports/NDP_SHN_RELIABLE_ANCHOR_FULL60_RESULTS_20261109.md)

结论：论文可以以 GPC 为主要贡献，以 SHN/RELIABLE/RISK 作为单 seed NDP 探索性消融；不要把 RISK 的小幅 mAP 增长包装成稳定的 NDP 普适增益。若投稿前只补一个实验，应补当前代码下 LTCC/PRCC 的严格配对 NDP=0，而不是继续开发新的 NDP 变体。
