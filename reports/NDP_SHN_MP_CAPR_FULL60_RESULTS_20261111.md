# NDP_SHN_MP_CAPR 正式实验报告

日期：2026-11-11  
仓库：`smallvv1/test2`  代码分支：`main`  
正式 seed：1234；训练：双 PPU；epoch：60；邻域关系：仅 train split。

## 1. 结论

本轮锁定 `NDP_SHN_MP_CAPR_B`。它保留原始 SHN-MP 的 8 个 Anchor、Top-3 身份中心负样本、`tau=0.2`、多正样本 Softplus 和 `lambda_ndp=0.001`，仅在原始关系之外补充最多 2 个未覆盖 PID 的 Anchor，并对同 Batch 的跨衣正样本施加独立的局部排序项。GPC 代码、正样本选择、预算和权重 `0.1` 均未修改。

在官方换装协议上，CAPR-B 同时超过 Bio+GPC 的 LTCC CC Rank-1/mAP 和 PRCC C→A Rank-1/mAP：

- LTCC CC：`57.6531/34.3787`，相对 Bio+GPC `+2.2959/+0.4421 pp`。
- PRCC C→A：`64.7756/63.8407`，相对 Bio+GPC `+0.6774/+0.4264 pp`。

严格配对的当前代码 `lambda_ndp=0` 控制进一步表明新增 NDP 的贡献为：

- LTCC CC：`+1.0204 Rank-1 / +0.7757 mAP pp`。
- PRCC C→A：`+1.3548 Rank-1 / -0.0091 mAP pp`。

因此，LTCC 的 Rank-1 与 mAP 增益能够归因于 NDP；PRCC 的首位检索增益也能够归因于 NDP，但 PRCC mAP 相对严格控制基本持平，不能宣称 NDP 带来 PRCC mAP 增益。

## 2. 方法与分支

原始 SHN-MP 保持不变。CAPR 的总损失为：

```text
L = L_Bio + 0.1 L_GPC + 0.001 (L_SHN_MP + 0.5 L_CAPR)
```

CAPR 候选要求 Anchor 在 Batch 内存在同 PID、不同衣服的正样本，并且 PID 未被原始 8 Anchor 覆盖；最多补充 2 个 Anchor，按每个 Anchor 的局部正负混淆风险排序，不使用 ABM 的全局中位数/MAD 硬筛选。风险权重 detached 并限制在 `[0.75, 1.25]`。

- **A**：只增加额外 Anchor，沿用原 SHN-MP 局部损失。
- **B**：A 加入跨衣正样本逐项 Softplus 排序；这是正式候选。
- **C**：B 再加入 `beta=0.3` 的图像级 smooth HardSim 负样本混合；只做短程验证，未作为正式结果。

内部 train-only identity-held-out 验证固定了 B：从训练身份中按确定规则留出 30/150 个 PID，未使用官方 query/gallery。A/B/C 的 3 epoch 结果只用于排除明显错误和选择候选，不用于预测正式排名：

| 分支 | held-out Rank-1 | held-out mAP | 说明 |
|---|---:|---:|---|
| A | 70.4687 | 64.4860 | 有效梯度 |
| B | 70.4958 | 64.5364 | 选择正式实验 |
| C | 70.2520 | 64.1353 | HardSim 首轮下降，未正式训练 |

## 3. 四协议正式主结果

所有数值为 Rank-1 / mAP（百分比）。mINP 按仓库当前 evaluator 不提供，未编造。

| 方法 | LTCC General | LTCC CC | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| Bio | 83.5700 / 52.4800 | 48.9796 / 29.1699 | 100.0000 / 99.2069 | 64.6063 / 63.0032 |
| Bio+GPC | 84.3813 / 54.1862 | 55.3571 / 33.9367 | 100.0000 / 99.4787 | 64.0982 / 63.4143 |
| SHN-MP | 86.0041 / 55.3313 | 57.3980 / 33.4117 | 100.0000 / 99.5250 | 64.1547 / 63.7777 |
| **CAPR-B** | **85.3955 / 55.0222** | **57.6531 / 34.3787** | **100.0000 / 99.5509** | **64.7756 / 63.8407** |

### 相对 Bio+GPC 的增量（pp）

| 方法 | LTCC General | LTCC CC | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| SHN-MP | +1.6227 / +1.1451 | +2.0408 / -0.5249 | +0.0000 / +0.0463 | +0.0565 / +0.3634 |
| CAPR-B | +1.0142 / +0.8359 | **+2.2959 / +0.4421** | +0.0000 / +0.0722 | **+0.6774 / +0.4264** |

### GPC 独立贡献（Bio+GPC − Bio）

| 协议 | Rank-1 pp | mAP pp |
|---|---:|---:|
| LTCC General | +0.8114 | +1.7062 |
| LTCC CC | +6.3775 | +4.7668 |
| PRCC Same-Clothes | +0.0000 | +0.2718 |
| PRCC C→A | -0.5080 | +0.4111 |

### CAPR 相对 SHN-MP

| 协议 | Rank-1 pp | mAP pp |
|---|---:|---:|
| LTCC General | -0.6085 | -0.3091 |
| LTCC CC | +0.2551 | +0.9670 |
| PRCC Same-Clothes | +0.0000 | +0.0259 |
| PRCC C→A | +0.6209 | +0.0630 |

## 4. 严格 NDP=0 配对控制

控制使用 CAPR-B 当前代码、相同共同初始化、seed、采样、AMP、优化器和 60 epoch 协议，仅设置 `--ndp-lambda-override 0.0`。控制没有使用测试集关系，也没有生成逐 Rank checkpoint。

| 方法 | LTCC General | LTCC CC | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| CAPR-B, lambda=0 | 85.3955 / 55.2558 | 56.6327 / 33.6031 | 100.0000 / 99.5312 | 63.4208 / 63.8498 |
| CAPR-B, lambda=0.001 | 85.3955 / 55.0222 | 57.6531 / 34.3787 | 100.0000 / 99.5509 | 64.7756 / 63.8407 |
| **NDP 独立贡献** | **+0.0000 / -0.2336** | **+1.0204 / +0.7757** | **+0.0000 / +0.0197** | **+1.3548 / -0.0091** |

这组控制也说明：PRCC C→A 的主要新增收益是首位排序，mAP 并未同步超过 lambda=0 控制；论文中应分别报告 Rank-1 与 mAP，不能把 PRCC mAP 的微小差异写成增益。

## 5. Query 级分析

分析使用官方 evaluator 产生的有效 query，训练过程中没有使用这些 query/gallery。`gained/lost` 表示 CAPR-B 相对比较模型的 Rank-1 首位变化；AP 统计按每个 query 的 AP 变化。

| 数据集/比较 | Rank-1 gained/lost/net | AP improved/decreased/unchanged | 平均 AP 变化 |
|---|---:|---:|---:|
| LTCC CC：CAPR-B vs Bio+GPC | 25 / 16 / +9 | 203 / 188 / 1 | +0.4421 pp |
| LTCC CC：CAPR-B vs SHN-MP | 22 / 21 / +1 | 222 / 170 / 0 | +0.9670 pp |
| LTCC CC：CAPR-B vs lambda=0 | 26 / 22 / +4 | 224 / 168 / 0 | +0.7757 pp |
| PRCC C→A：CAPR-B vs Bio+GPC | 71 / 47 / +24 | 1634 / 1273 / 636 | +0.4264 pp |
| PRCC C→A：CAPR-B vs SHN-MP | 71 / 49 / +22 | 1569 / 1340 / 634 | +0.0630 pp |
| PRCC C→A：CAPR-B vs lambda=0 | 78 / 30 / +48 | 1455 / 1438 / 650 | -0.0091 pp |

LTCC 的 mAP 改善不是单个 Rank-1 query 的偶然变化：相对 lambda=0 有 224 个 query 的 AP 改善、168 个下降，平均 AP 提升 0.7757 pp。PRCC 的 Rank-1 获益更集中且更清楚（相对 lambda=0 净增加 48 个 query），但 AP 改善与下降几乎抵消。

## 6. 训练统计与覆盖关系

正式训练 epoch 3–60 的平均诊断如下；统计仅来自 train split。

| 数据集 | NDP active rate | NDP 关系/epoch | CAPR 额外 Anchor/epoch | 跨衣正样本/epoch | 风险权重均值 | 原始 NDP/Main 梯度比 | lambda 加权梯度比 | 梯度余弦 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LTCC | 99.54% | 4745.2 | 0 | 0 | — | 0.1485 | 0.000148 | 0.5711 |
| PRCC | 99.96% | 5266.3 | 859.4 | 1481.1 | 0.8994 | 0.3092 | 0.000309 | 0.4867 |

LTCC 当前 Batch 采样几乎没有满足“未覆盖 PID + 同 PID 跨衣正样本”的额外候选，因此 CAPR 没有替换或删除原 SHN-MP 的 8 个 Anchor；LTCC 的提升来自原始 SHN-MP 关系上的 NDP 梯度，而不是额外 Anchor 数量。PRCC 则平均每个 epoch 有约 859 个额外 Anchor 计数和 1481 个跨衣正样本计数，风险权重没有失控，且额外关系与 Batch-Hard 困难关系诊断重合率约为 99.96%。

一次 train-only synthetic probe 显示原始 SHN-MP 的 8 个名额可能由少数 PID 重复占用（示例 PID 为 `[1,1,1,1,9,9,9,9]`）；CAPR 保留这 8 个原始关系，只从未覆盖 PID 补充最多 2 个，避免 ABM “每 PID 至多一个”造成的过度削减。该 probe 不涉及官方测试集。

## 7. 训练状态与复现归档

- A/B/C 单元测试和 Smoke Test 已通过；最终 pytest：`40 passed`。
- 代码通过 `py_compile` 和 `git diff --check`。
- 正式 CAPR-B：LTCC/PRCC 均 `60/60`、seed=1234、双 PPU、无训练跳步异常。
- 严格 lambda=0 控制：LTCC/PRCC 均 `60/60`、seed=1234、双 PPU。
- 每个实验只保留 `model_final.pth`；训练完成后没有保留逐 Rank checkpoint，扫描未发现 `rank*state.pt/pth`。
- 训练和邻域挖掘的 `RUN_COMPLETE.json` 标记 `query_gallery_used_during_training=false`、`train_only_reference=true`。
- GPC 权重仍为 `0.1`；CAPR 的 NDP 权重为 `0.001`。
- 小体积日志、评估 JSON、epoch CSV、paired query summary 已归档到 `runs/ndp-shn-mp-capr-20261110/`；模型文件仍保留在 `/home/DIFFER/log_NDP/`，未上传大模型。

## 8. 局限与论文建议

CAPR-B 的结果已经足以作为 NDP 的单 seed 正式对照，但 PRCC 的 mAP 相对严格控制没有提升，不能夸大为全面检索质量改善；PRCC 的可信增益应表述为 C→A Rank-1 首位排序改善。LTCC CAPR 的额外 Anchor 为 0，论文中应明确该数据集主要验证了 SHN-MP 主关系和局部排序保护，而不是声称额外 Anchor 在两个数据集都同等激活。

建议论文主表使用 Bio、Bio+GPC、SHN-MP、CAPR-B，并把 lambda=0 配对控制作为 NDP 因果消融；A/C 仅作为轻量实现消融。当前不建议继续堆叠新的 Margin、Bank 或 HardSim 变体，优先完成 GPC 主结果、NDP 严格控制和失败方案整理。
