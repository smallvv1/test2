# NDP_SHN_RELIABLE：正式 60 epoch 实验报告

日期：2026-11-07  
仓库：`smallvv1/test2`  
正式 seed：`1234`  
训练设备：2 × PPU-ZW810E  
GPC：原代码、样本选择与权重均保持不变，`lambda_gpc=0.1`

## 1. 方法

本轮以效果较好的 `NDP_SHN_MP` 实时多正样本/困难负身份损失为主项。新增的特征库只作为轻量辅助：

- 每个 batch 最多选择 2 个跨 batch 同身份正样本；优先跨衣样本。
- 只保留经过身份中心支持（可靠性下限 0.55）、且不明显差于当前 batch 正样本的关系。
- bank 正样本端点 detach，辅助项固定乘 0.25；总 `lambda_ndp=0.001`。
- 负身份及 Top-3 软负身份混合始终由学生当前 batch 特征计算，bank 负中心不替代实时负样本。
- bank 仅使用 train split，按 epoch 刷新；不保存历史 Margin 目标，也不使用 query/gallery 选参。

## 2. 三阶段正式结果

数值为换装协议的 Rank-1 / mAP，差值为百分点（pp）。

| 协议 | Bio | Bio+GPC | Bio+GPC+NDP_SHN_RELIABLE | GPC 独立提升 | NDP 独立提升 | 总提升 |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.57 / 52.48 | 84.38 / 54.19 | 84.38 / 54.05 | +0.81 / +1.71 | +0.00 / -0.14 | +0.81 / +1.57 |
| LTCC Clothes-Changing | 48.98 / 29.17 | 55.36 / 33.94 | 55.87 / 33.71 | +6.38 / +4.77 | +0.51 / -0.22 | +6.89 / +4.54 |
| PRCC Same-Clothes | 100.00 / 99.21 | 100.00 / 99.48 | 100.00 / 99.50 | +0.00 / +0.27 | +0.00 / +0.02 | +0.00 / +0.29 |
| PRCC Clothes-Changing C→A | 64.61 / 63.00 | 64.10 / 63.41 | 64.47 / 63.87 | -0.51 / +0.41 | +0.37 / +0.46 | -0.14 / +0.87 |

完整原始值（便于复核）：

- LTCC：Bio-only 与当前统一评估器结果位于 `bio_only_unified_eval_20261106/LTCC/eval_final.json`；固定 B0 位于 `ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/eval_final.json`。
- PRCC：Bio-only 与当前统一评估器结果位于 `bio_only_unified_eval_20261106/PRCC/eval_final.json`；固定 B0 位于 `ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/eval_final.json`。
- 新模型评估位于 `/home/DIFFER/log_NDP/ndp_shn_reliable_full60_seed1234_20261107/{LTCC,PRCC}/eval_final.json`。

## 3. 与前序 NDP 消融的关系

| 方法 | LTCC CC | PRCC C→A |
|---|---:|---:|
| Bio+GPC | 55.36 / 33.94 | 64.10 / 63.41 |
| NDP_SHN_MP | 57.40 / 33.41 | 64.15 / 63.78 |
| NDP_SHN_BANK | 56.38 / 32.48 | 63.34 / 63.59 |
| NDP_SHN_RELIABLE（本轮） | 55.87 / 33.71 | 64.47 / 63.87 |

本轮相对 `NDP_SHN_BANK` 的变化为：LTCC CC `-0.51 / +1.23` pp，PRCC C→A `+1.13 / +0.28` pp；说明减少冻结 bank 监督、保留实时负样本路径确实修复了 BANK 版本的部分退化。相对 `NDP_SHN_MP`，LTCC CC Rank-1 仍低 `1.53` pp，但 mAP 高 `0.30` pp；PRCC C→A Rank-1/mAP 分别高 `0.31/0.09` pp。

## 4. 训练与完整性状态

- LTCC：60/60 epoch，35,760 optimizer steps；PRCC：60/60 epoch，33,060 optimizer steps。
- 两个正式运行均从既有共同 Bio 初始化开始，未重新训练 Bio 或 GPC；seed=1234。
- 训练中只使用 train split 的实时特征与 train-only bank；正式评估严格使用现有统一检索器和原始 query/gallery 协议。
- LTCC 最后 epoch：实时 SHN 4,736 条关系、bank 辅助 1,184 条关系，辅助可靠性均值约 0.895，NDP/主干特征梯度范数比约 0.211。
- PRCC 最后 epoch：实时 SHN 4,408 条关系、bank 辅助 1,102 条关系，辅助可靠性均值约 0.906，NDP/主干特征梯度范数比约 0.337。
- CE、Triplet、GPC 和 NDP 均有有限值；AMP 仅出现少量短暂 skip，没有中断、恢复或异常重启。
- Smoke Test、3 epoch train-only 验证通过；代码测试 `37 passed`。
- 正式目录只保留 `model_final.pth`、日志、CSV、JSON；没有生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。正式训练完成后未保留 `resume_latest.pth`，因此不存在多份恢复副本。

## 5. 结论

本轮方案在 PRCC C→A 上同时提升 Rank-1 和 mAP，相对固定 Bio+GPC 为 `+0.37/+0.46` pp；在 LTCC CC 上 Rank-1 提升 `+0.51` pp，但 mAP 下降 `-0.22` pp，LTCC General mAP 也下降 `-0.14` pp。因此目前不能宣称在两个数据集、所有主要指标上取得稳定增益。

该结果支持“实时 SHN 为主、可靠 train-only bank 为弱辅助”优于冻结 bank 等权补充，但不足以继续堆叠相似 NDP 模块。后续论文应保留本轮完整对照，同时以固定 Bio+GPC 与表现更强的 `NDP_SHN_MP` 作为主要消融证据，停止继续扫描 NDP 权重或扩展特征库损失。
