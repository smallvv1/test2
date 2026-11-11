# NDP-CAPR 严格配对控制与论文收尾报告

日期：2026-11-11  
仓库：`smallvv1/test2`，分支：`main`，当前代码提交：`2d3744a`  
正式配置：seed=1234、双 PPU、60 epoch、GPC 权重 `0.1`、NDP 权重 `0.001`；所有邻域关系只来自 train split。

## 1. 本次任务状态

本次先检查了 `/home/DIFFER/` 下的进程、日志和模型，没有发现仍在运行的训练进程，因此没有重复启动训练，也没有从头重跑任何已有正式实验。

四个正式目录均已完整结束 60 epoch，并通过统一 evaluator：

| 目录 | 配置 | epoch | `lambda_ndp` |
|---|---|---:|---:|
| `ndp_shn_mp_capr_nr_ltcc_b_full60_20261111/LTCC` | CAPR-NR-B | 60/60 | 0.001 |
| `ndp_shn_mp_capr_nr_ltcc_ndp0_full60_20261111/LTCC` | 严格配对控制 | 60/60 | 0.0 |
| `ndp_shn_mp_capr_nr_prcc_b_full60_20261111/PRCC` | CAPR-NR-B | 60/60 | 0.001 |
| `ndp_shn_mp_capr_nr_prcc_ndp0_full60_20261111/PRCC` | 严格配对控制 | 60/60 | 0.0 |

控制实验使用同一 CAPR-NR-B 代码、seed、共同初始化、采样协议、AMP、优化器和 scheduler，仅将 `lambda_ndp` 置零。LTCC/PRCC 均使用共同的 2 epoch warmup 分叉；共同初始化摘要 SHA256 为 `e3cc88e940eec00bfae797e56c09d4c7a56009197b43f376aa1fea2eaca78449`。控制的 `RUN_COMPLETE.json` 已核对为 `lambda_ndp=0.0`。

四个正式模型的 SHA256：

- LTCC CAPR-NR-B：`074ad2545c5134a68f06232c778ea8adfc97f3ab949f1ad3d8e857e6e19242e2`
- LTCC NDP=0：`c08f5e334d8451c1c8f3fd57150b15da3f39c3a62d0ae8c8992c3e0fd6396482`
- PRCC CAPR-NR-B：`94dac146de2f07062c8c0f21245806baf0b48bb58f628515ba003dae8dba835f`
- PRCC NDP=0：`d3dbee3bca883349a636ce4aeef592e7a586ea352f9ff741ebdbb974d21b3dcd`

没有生成或上传 `rank*state.pt`、`rank*state.pth`，训练结束后本机仅保留必要的 `model_final.pth`；仓库只归档小体积 JSON、CSV、日志和本报告。

## 2. CAPR-NR-B 与原方法的关系

CAPR-NR-B 保留原始 SHN-MP 的 8 个 Anchor、Top-3 身份中心、多正样本 Softplus、`tau=0.2` 和 `lambda_ndp=0.001`。CAPR-B 的额外跨衣局部排序保留为独立项；NR 只从 Top-3 中挑选非 Batch-Hard、仍对跨衣正样本构成局部排序威胁的负身份，采用 detached 的有界冗余权重，不使用历史 Margin 恢复、Teacher 或 Bank 固定目标。

实现目标是：

```text
L = L_Bio + 0.1 L_GPC + 0.001 (L_SHN_MP + 0.5 L_CAPR_B + 0.25 L_NR)
```

GPC 的代码、正样本选择、预算和权重均未改动。

需要注意一个归档事实：CAPR-NR-B 正式训练发生在最新关系计数修复提交之前，因此其 `metrics_per_epoch.csv` 中部分 `shn_live_relations`/`ndp_active_relations` 字段被旧统计分支记录为 0；这只是计数器问题，不代表损失没有计算。正式日志中的 `ndp_loss` 非零，PRCC 也记录了 CAPR-NR 选择关系。修复后的代码已提交，后续报告不把旧的 0 计数伪装成真实激活率；NDP=0 的严格评估只据实际模型和统一 evaluator 判断。

## 3. 四协议正式结果

数值为 Rank-1 / mAP（百分比）。当前仓库 evaluator 不输出 mINP，因此不编造 mINP。

| 方法 | LTCC General | LTCC Clothes-Changing | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| Bio | 83.5700 / 52.4800 | 48.9796 / 29.1699 | 100.0000 / 99.2069 | 64.6063 / 63.0032 |
| Bio+GPC | 84.3813 / 54.1862 | 55.3571 / 33.9367 | 100.0000 / 99.4787 | 64.0982 / 63.4143 |
| SHN-MP | 86.0041 / 55.3313 | 57.3980 / 33.4117 | 100.0000 / 99.5250 | 64.1547 / 63.7777 |
| CAPR-B（历史正式） | 85.3955 / 55.0222 | 57.6531 / 34.3787 | 100.0000 / 99.5509 | 64.7756 / 63.8407 |
| **CAPR-NR-B** | **84.3813 / 55.3774** | **54.8469 / 34.1691** | **100.0000 / 99.4786** | **64.3805 / 64.1874** |
| CAPR-NR-B，NDP=0 | 85.3955 / 54.6456 | 57.1429 / 32.8960 | 100.0000 / 99.6124 | 63.8724 / 64.0707 |

## 4. 相对固定 Bio+GPC 的变化

差值单位为百分点（pp），格式为 Rank-1 / mAP。

| 方法 | LTCC General | LTCC CC | PRCC Same-Clothes | PRCC C→A |
|---|---:|---:|---:|---:|
| SHN-MP | +1.6227 / +1.1451 | +2.0408 / -0.5249 | +0.0000 / +0.0463 | +0.0565 / +0.3634 |
| CAPR-B | +1.0142 / +0.8359 | +2.2959 / +0.4421 | +0.0000 / +0.0722 | +0.6774 / +0.4264 |
| **CAPR-NR-B** | **+0.0000 / +1.1912** | **-0.5102 / +0.2324** | **+0.0000 / -0.0001** | **+0.2823 / +0.7737** |

CAPR-NR-B 没有保持历史 CAPR-B 的 LTCC CC Rank-1，虽然 CC mAP 仍略高于 Bio+GPC。PRCC C→A 的 mAP 是本轮最好的正式值，但 Rank-1 仅提升 0.2823 pp，远低于此前设定的 +1.5 pp 目标。

## 5. 严格配对的 NDP 独立贡献

以下是 CAPR-NR-B 减去同代码、同协议的 NDP=0 控制：

| 协议 | NDP 独立 Rank-1 / mAP 增量（pp） | 解释 |
|---|---:|---|
| LTCC General | **-1.0142 / +0.7318** | 排序平均质量改善，但首位命中下降 |
| LTCC CC | **-2.2959 / +1.2731** | mAP 明显改善，但 Rank-1 下降，不是全面增益 |
| PRCC Same-Clothes | **0.0000 / -0.1338** | Rank-1 饱和，mAP 略降 |
| PRCC C→A | **+0.5080 / +0.1168** | NDP 对首位和平均排序均有小幅独立贡献 |

因此，本轮不能把 CAPR-NR-B 的 LTCC Rank-1 变化写成 NDP 成功提升；唯一较清楚的独立正向证据是 LTCC mAP 与 PRCC C→A 的小幅改善。PRCC 的结果有正向趋势，但不足以声称达到稳定的 +1.5 pp 提升。

## 6. 训练与诊断摘要

- 所有正式运行均为 60/60 epoch、seed=1234、双 PPU。
- LTCC NDP=0 后期每 epoch 约 4,745 条 SHN-MP live relations，active rate 约 99.54%；这确认对照训练路径确实执行了原 SHN-MP 关系计算，而不是空跑。
- PRCC CAPR-NR-B 的旧日志记录平均约 859 个 CAPR-NR 额外关系/epoch；其原始 NDP 梯度与主干梯度均非零，且没有出现 NaN/Inf 或 AMP 跳步异常。
- 由于 CAPR-NR-B 正式训练早于计数器修复，旧日志的逐关系激活字段不作为精确激活率证据；已将这个限制写入归档报告，避免把日志 bug 解释成方法收益。
- 训练、邻域挖掘和内部检查没有读取官方 query/gallery 做关系构建或参数选择。

## 7. 结论与后续建议

本次 LTCC NDP=0 严格配对控制已经安全完成，正式结果和模型均可复核。CAPR-NR-B 的实际结论是“PRCC C→A 有小幅、可归因的独立增益，LTCC mAP 有改善但 Rank-1 退化”，不是达到验收目标的成功版本。

与历史 CAPR-B 相比，CAPR-NR-B 在 LTCC CC Rank-1 低 2.8062 pp、mAP 低 0.2097 pp；在 PRCC C→A Rank-1 低 0.3951 pp、mAP 高 0.3467 pp。因此不应覆盖或替换历史 CAPR-B，论文主结果应保留 CAPR-B，CAPR-NR-B 作为严格配对失败/边界约束消融。

在当前证据下不建议继续启动新的 NDP 60 epoch 变体。更合理的收尾方式是：保留历史 CAPR-B、SHN-MP 和本轮 CAPR-NR-B/NDP=0 的完整证据，论文中分别报告 Rank-1 与 mAP，并明确说明 NDP 对“首位命中”和“整体排序”的影响可能不一致。
