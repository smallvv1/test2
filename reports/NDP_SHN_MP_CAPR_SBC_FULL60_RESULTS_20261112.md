# NDP-SHN-MP-CAPR-SBC 正式实验报告

日期：2026-11-12  
仓库：`smallvv1/test2`，分支：`main`  
代码快照：`bea5d8f`（本报告提交后为最新提交）  
正式 seed：1234；训练：双 PPU；epoch：60；邻域关系：仅 train split。

## 1. 本轮结论

本轮在历史 CAPR-B 上增加了稀疏边界混淆关系（SBC）监督。它不改变 GPC、Bio、CE、Triplet、原始 SHN-MP 或 CAPR-B 的关系选择，仅对 CAPR 已选中的额外 Anchor，筛选非 Batch-Hard、可靠跨衣正样本和接近局部排序边界的负身份关系。

正式 PRCC C→A 结果为：

- CAPR-SBC：`63.5055/63.3411`（Rank-1/mAP）。
- 严格配对 `eta=0`：`63.9853/64.0642`。
- SBC 相对配对控制：`-0.4803/-0.7231 pp`。

因此，SBC 没有带来独立正向贡献，且相对固定 Bio+GPC 也下降（`-0.5927/-0.0732 pp`）。本分支不应替代历史 CAPR-B，也不建议继续围绕同一类静态局部 Margin 约束做大规模搜索。

## 2. 方法实现

候选分支的损失为：

```text
L = L_Bio + 0.1 L_GPC + 0.001 (L_SHN_MP + 0.5 L_CAPR_B + 0.25 L_SBC)
```

其中：

1. 原始 SHN-MP 的 8 Anchor、Top-3 身份中心负样本、`tau=0.2`、多正样本 Softplus 和权重 `0.001` 完全保留。
2. CAPR-B 的最多 2 个额外 Anchor 和跨衣正样本排序完全保留。
3. SBC 只使用 CAPR 已选中的额外 Anchor；不新增 Anchor，不覆盖 CAPR-B。
4. 负身份排除当前 Batch-Hard PID，只从其余 Top-3 身份中心中寻找局部风险。
5. 正样本必须是同 PID、不同衣服，并满足 Batch 内同 PID 中心可靠性比例至少 `0.50`。
6. 用 detached 的 `s(anchor, negative_center)-s(anchor, positive)` 判断局部边界，只保留 `[-0.25 tau, +0.25 tau]` 内的关系；每个额外 Anchor 至多选一条，未满足条件时新增项严格为零。
7. `eta=0` 控制保留完全相同的 CAPR-B、SHN-MP、GPC 和训练协议，只关闭 `0.25 L_SBC`。

该模块补充的是 Triplet/CAPR-B 未明确覆盖的“非 Batch-Hard、但仍接近局部正负排序边界”的少量关系；它不是替换 Batch-Hard Triplet，也不是把 CAPR-B 关系重新平均。

## 3. 代码和基本测试

- `scripts/ndp_v5_prcc_runner.py`：新增 SBC 选择器、损失组合和训练统计。
- `scripts/train_ndp_listwise_full60.py`：注册独立 arm、SBC 统计字段和 `--capr-sbc-eta-override` 配对控制。
- `tests/test_ndp_shn.py`：新增 SBC 可选、边界过滤和实时梯度测试。
- `py_compile`：通过。
- `pytest -q tests/test_ndp_shn.py`：`12 passed, 7 warnings`。
- 双 PPU Smoke Test：通过；无 NaN/Inf，GPC/CE/Triplet 正常。
- PRCC 真正身份不重叠的 train-only 短程验证：留出 PID，不读取官方 query/gallery；SBC 有非零梯度且关系筛选有效。

之前一次 `idsplit5` 因留出身份后固定 sampler 数据量不足而安全失败，没有被当成性能结果；随后使用满足训练量要求的 `idsplit20` 完成短程检查。

## 4. 正式结果

所有数值为 Rank-1 / mAP（百分比）。历史模型和 Bio/Bio+GPC/CAPR-B 数值来自仓库既有正式报告；本轮新增模型均使用统一 evaluator。仓库 evaluator 不提供 mINP，因此不编造 mINP。

| 方法 | LTCC General | LTCC Clothes-Changing | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| Bio | 83.5700 / 52.4800 | 48.9796 / 29.1699 | 100.0000 / 99.2069 | 64.6063 / 63.0032 |
| Bio+GPC | 84.3813 / 54.1862 | 55.3571 / 33.9367 | 100.0000 / 99.4787 | 64.0982 / 63.4143 |
| SHN-MP | 86.0041 / 55.3313 | 57.3980 / 33.4117 | 100.0000 / 99.5250 | 64.1547 / 63.7777 |
| CAPR-B（历史正式） | 85.3955 / 55.0222 | 57.6531 / 34.3787 | 100.0000 / 99.5509 | 64.7756 / 63.8407 |
| CAPR-SBC（本轮） | — | — | 100.0000 / 99.4644 | 63.5055 / 63.3411 |
| CAPR-SBC，`eta=0` 配对控制 | — | — | 100.0000 / 99.4340 | 63.9853 / 64.0642 |

LTCC 没有重新进行正式 60 epoch，因为本轮新增模块的可行关系由训练数据自动决定，历史 CAPR-B 正式模型必须保持不变。新的 LTCC 3 epoch train-only 检查中，epoch 1–3 的 `CAPR extra Anchor=0`、`SBC candidate=0`、`SBC selected=0`，因此 SBC 不会改变 LTCC 的 CAPR-B 路径；LTCC 正式论文结果仍使用历史 CAPR-B：`57.6531/34.3787`。

### 相对固定 Bio+GPC 的变化（pp）

| 方法 | LTCC General | LTCC CC | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| SHN-MP | +1.6227 / +1.1451 | +2.0408 / -0.5249 | +0.0000 / +0.0463 | +0.0565 / +0.3634 |
| CAPR-B | +1.0142 / +0.8359 | **+2.2959 / +0.4421** | +0.0000 / +0.0722 | **+0.6774 / +0.4264** |
| CAPR-SBC（PRCC） | — | — | +0.0000 / -0.0143 | -0.5927 / -0.0732 |
| CAPR-SBC，`eta=0`（PRCC） | — | — | +0.0000 / -0.0447 | -0.1129 / +0.6499 |

### SBC 的严格配对独立贡献（SBC − eta=0）

| 协议 | Rank-1 pp | mAP pp |
|---|---:|---:|
| PRCC Same-Clothes | 0.0000 | +0.0304 |
| PRCC C→A | **-0.4803** | **-0.7231** |

这组控制说明：本轮候选与控制之间的差异确实来自 SBC 新增项，而不是初始化、采样、优化器或 AMP 的整体差异；但方向为负。

## 5. PRCC 风险与覆盖诊断

正式训练 epoch 3–60 的均值如下，全部来自 train split。

| 指标 | SBC `eta=0.25` | `eta=0` |
|---|---:|---:|
| NDP active rate | 99.9625% | 99.9625% |
| SHN-MP live relations/epoch | 4406.90 | 4406.90 |
| CAPR extra Anchor/epoch | 859.29 | 859.21 |
| CAPR cross-clothes positives/epoch | 1480.67 | 1479.74 |
| CAPR Top-3 与 Triplet hard overlap | 49818/49839 = 99.9583% | 49815/49834 = 99.9619% |
| SBC candidate nonhard relations/epoch | 1606.00 | 1605.88 |
| SBC boundary relations/epoch | 184.07 | 176.69 |
| SBC selected relations/epoch | 124.95 | 120.53 |
| SBC selected violation/epoch | 13.98 | 13.57 |
| 选中/候选边界关系 | 7.78% | 7.50% |
| 选中关系中 violation | 11.19% | 11.26% |

现象解释：CAPR Top-3 与 Triplet 的 PID 集合重合率几乎 100%，所以 SBC 虽然排除了 Batch-Hard，但它面对的可用候选仍主要来自与 Triplet 相邻的负身份集合。边界带在训练后迅速收缩到极少数关系，说明筛选器没有过密激活；然而这些关系仍未提供足够独立的排序信息，且其梯度可能改变已经有效的 CAPR-B 局部排序，导致 PRCC 首位命中和整体 AP 同时下降。

## 6. 梯度和训练稳定性

正式 PRCC epoch 3–60 的诊断：

| 指标 | SBC `eta=0.25` | `eta=0` |
|---|---:|---:|
| NDP feature grad norm | 0.02513 | 0.02368 |
| Main feature grad norm | 0.09551 | 0.09427 |
| 原始 NDP/Main grad ratio | 0.3151 | 0.3067 |
| 梯度 cosine | 0.4800 | 0.4871 |
| skipped steps | 11/58 | 13/58 |

SBC 并非没有梯度：它在候选训练中实际激活，梯度比例有限且余弦为正；问题是监督内容与主干已经学习的边界关系不够互补，而不是简单的“权重太小导致未生效”。因此不应据此盲目增加 NDP 全局权重。

## 7. 训练、评估和文件审计

- PRCC SBC 正式训练：60/60 epoch，seed=1234，双 PPU，退出码 0。
- PRCC `eta=0` 严格控制：60/60 epoch，seed=1234，双 PPU，退出码 0。
- 两个正式模型均使用同一 train 初始化、训练 batch、AMP、优化器、scheduler 和 evaluator；模型 SHA256 已写入各自 `eval_final.json`。
- LTCC SBC train-only：3/3 epoch；没有额外 Anchor 或 SBC 关系。
- 训练和邻域挖掘没有使用官方 query/gallery 做关系构建或选择。
- 训练完成后只保留单文件 `model_final.pth`；`resume_latest.pth` 在正式 60 epoch 结束后清理。
- `/home/DIFFER/log_NDP/` 的本轮输出、仓库 archive 和整个仓库均扫描未发现 `rank*state.pt` 或 `rank*state.pth`。
- 大模型 checkpoint 未上传 GitHub；仓库只归档小体积日志、CSV、JSON、初始化 manifest 和报告。

## 8. 最终判断与建议

1. SBC 的确补充了一个不同于 Batch-Hard 的非硬负身份边界关系，并且有真实梯度；但在 PRCC C→A 上相对严格配对控制下降，不能作为最终 NDP 方法。
2. LTCC 原有 CAPR-B 正式结果没有被修改，新的 SBC train-only 检查也显示它在 LTCC 自动为零，因此 LTCC 收益保持来自历史 CAPR-B/SHN-MP 路径，而不是 SBC。
3. PRCC Rank-1 未改善的最可能原因是：CAPR Top-3 与 Triplet 的负身份候选高度重合，新增边界关系虽然排除了 hard PID，却仍与已有监督共享同一局部几何；同时边界带关系在训练后过快变稀疏，无法稳定改变首位排序，反而扰动了 CAPR-B 已有的排序结构。
4. 本轮 SBC 不值得作为论文最终 NDP；论文主结果应继续使用历史 CAPR-B，并把 SBC 与 eta=0 作为失败的低干扰消融，真实说明“非 Batch-Hard 边界监督并不自动带来独立增益”。
5. 下一步不建议继续堆叠 Margin/Bank/HardSim 变体。若论文需要进一步实验，优先做 Query 级错误类型分析和现有 CAPR-B 的复现/消融整理；只有发现明确的非重合监督来源后，才考虑新的 NDP 设计。

