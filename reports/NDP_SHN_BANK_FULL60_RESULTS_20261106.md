# NDP_SHN_BANK 正式实验报告

日期：2026-11-06  
正式实验目录：`/home/DIFFER/log_NDP/ndp_shn_bank_full60_seed1234_20261106/`  
统一 Bio-only 评估目录：`/home/DIFFER/log_NDP/bio_only_unified_eval_20261106/`

## 1. 实验目标与方法

本轮实现轻量的 `NDP_SHN_BANK`，用于补足 NDP_SHN_MP 在单个 Batch 中同身份正样本数量不足的问题。GPC 完全复用固定实现，代码、正样本选择、采样器和权重 `lambda_gpc=0.1` 均未修改；NDP 权重为 `lambda_ndp=0.001`。

- 保留 SHN_MP 的身份中心、Top-3 困难负身份、8 个 Anchor、`tau=0.2` 和均衡多正样本 softplus 排序。
- 每 5 个 epoch 从当前学生模型刷新一次 detached 的训练集特征库；首次在 epoch 3 建库。
- 特征库只包含 train split：LTCC 为 9576 张图、77 个身份；PRCC 为 17896 张图、150 个身份。
- 对每个 Anchor 使用当前 Batch 的 live 特征，同时补充跨 Batch 同 PID 正样本；有衣物标签时优先选择不同衣物的训练正样本，最多补充 8 个。
- 负身份使用特征库身份中心的 Top-3 困难负身份；不读取 query/gallery，不用测试集选参。
- 不引入额外网络、分类器或逐 Rank checkpoint。

## 2. Bio-only 模型与评估协议核对

已有正式 Bio-only 模型，没有重新训练：

- LTCC：`/data/DIFFER/outputs/new_method_track/phase6c_full_schedule_confirmation/B0_full/model_final.pth`，`RUN_COMPLETE.json` 显示 60 epoch、35760 次 optimizer update，SHA256 `01fc2141de86c0d9f5806af0dc933c1210ad2346bfcdbfd89bd64483b3ebb011`。
- PRCC：`/data/DIFFER/outputs/new_method_track/phase7a_prcc_relation_validation/P0_prcc_bio/model_final.pth`，`RUN_COMPLETE.json` 显示 60 epoch、33060 次 optimizer update，SHA256 `6ce6da7431c259aa92215cdca014b3df858302a8182929e577aa31505c21bf18`。

两套 Bio-only 模型均通过当前仓库的 `train_ndp_listwise_full60.py --mode eval` 评估。Bio+GPC 和 NDP_SHN_BANK 也使用同一 `eval_protocol`，同一 LTCC General/Clothes-Changing、PRCC Same-Clothes/C→A 检索协议和相同 query/gallery 数量。因此本报告没有混用旧评估器数值。历史报告中的 LTCC `48.98/29.17` 仅作交叉核对，不作为本表来源。

## 3. 三阶段正式结果

表中数值均为百分比，格式为 `Rank-1 / mAP`；所有差值为百分点（pp）。

| 数据集 / 协议 | Bio | Bio+GPC | Bio+GPC+NDP_SHN_BANK | GPC 独立提升（GPC−Bio） | NDP 独立提升（NDP−GPC） | 总提升（NDP−Bio） |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 83.5700 / 52.4800 | 84.3813 / 54.1862 | 85.1927 / 55.0529 | +0.8114 / +1.7062 | +0.8114 / +0.8666 | +1.6227 / +2.5729 |
| LTCC Clothes-Changing | 48.9796 / 29.1699 | 55.3571 / 33.9367 | 56.3776 / 32.4804 | +6.3775 / +4.7668 | +1.0204 / −1.4563 | +7.3980 / +3.3105 |
| PRCC Same-Clothes | 100.0000 / 99.2069 | 100.0000 / 99.4787 | 100.0000 / 99.4325 | +0.0000 / +0.2718 | +0.0000 / −0.0462 | +0.0000 / +0.2256 |
| PRCC Clothes-Changing C→A | 64.6063 / 63.0032 | 64.0982 / 63.4143 | 63.3362 / 63.5865 | −0.5080 / +0.4111 | −0.7621 / +0.1722 | −1.2701 / +0.5832 |

原始评估 JSON：

- Bio-only：`/home/DIFFER/log_NDP/bio_only_unified_eval_20261106/{LTCC,PRCC}/eval_final.json`
- Bio+GPC：`/home/DIFFER/log_NDP/ndp_nteach_b0_recovery_20261030_210612/LTCC_B0_full/LTCC/eval_final.json`；`/home/DIFFER/log_NDP/ndp_nteach_current_b0_pair_20261031_125146/PRCC_B0_full/PRCC/eval_final.json`
- NDP_SHN_BANK：`/home/DIFFER/log_NDP/ndp_shn_bank_full60_seed1234_20261106/{LTCC,PRCC}/eval_final.json`

## 4. 训练与完整性状态

- LTCC：60/60 epoch，35760 steps；PRCC：60/60 epoch，33060 steps。
- Seed：`1234`；使用已有共同初始化；PPU-ZW810E 双进程训练；未重新训练 Bio+GPC B0。
- 固定 GPC 上下文：`phase7i_relation_specificity`，`lambda_gpc=0.1`。
- NDP_SHN_BANK：`lambda_ndp=0.001`，epoch 3 后激活，每 5 epoch 刷新训练集学生特征库。
- Smoke Test：LTCC、PRCC 各 20 steps 通过；3 epoch train-only 验证通过。epoch 3 的 NDP 特征梯度范数为 LTCC `0.005279`、PRCC `0.006707`，主干梯度分别为 `0.367321`、`0.247739`，均有限且非零。
- 正式训练最后一 epoch 的 NDP/主干特征梯度范数：LTCC `0.017114/0.125148`，PRCC `0.016097/0.052780`；NDP 实际参与反向传播。
- 正式训练累计激活：LTCC 34568 个 batch、276544 条关系；PRCC 31958 个 batch、255664 条关系。AMP 跳步分别为 15、12，训练完成且无 NaN。
- 仅保留正式目录中的 `resume_latest.pth`（训练期间）和 `model_final.pth`（最终）；训练完成后未保留 resume 文件；没有任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。
- `RUN_COMPLETE.json` 明确记录 `query_gallery_used_during_training=false`、`train_only_reference=true`、`per_rank_checkpoint_files_created=false`。

模型 SHA256：

- LTCC NDP_SHN_BANK：`4ae69fcc68ddf53a9c670a84a4227b3535dc206965b7d46aa49b571a1ab16d6b`
- PRCC NDP_SHN_BANK：`219ac703e3d67ccdf8cf81187749c391884e2e51d456b66401a7b21237561ac8`

## 5. 模块贡献与消融分析

GPC 的独立贡献由 Bio→Bio+GPC 给出：LTCC Clothes-Changing 提升 `+6.3775/+4.7668 pp`，PRCC C→A 的 Rank-1 为 `−0.5080 pp`、mAP 为 `+0.4111 pp`。这说明固定 GPC 是主要的 LTCC 换装收益来源。

NDP_SHN_BANK 相对固定 Bio+GPC 的独立贡献为：LTCC General `+0.8114/+0.8666 pp`，LTCC Clothes-Changing `+1.0204/−1.4563 pp`，PRCC Same-Clothes `0/−0.0462 pp`，PRCC C→A `−0.7621/+0.1722 pp`。新增 NDP 路径确实激活且梯度有效，但没有同时改善 LTCC CC 的 Rank-1 和 mAP，也没有同时改善 PRCC C→A 的 Rank-1 和 mAP。

因此，本轮不能宣称 NDP_SHN_BANK 带来稳定的跨数据集全面增益。它对 LTCC CC Rank-1 有真实提升，但 CC mAP 明显下降；PRCC C→A mAP 小幅提升而 Rank-1 下降。与上一轮 SHN_MP 相比，特征库补充正样本并未解决 LTCC CC mAP 瓶颈，说明问题不只是 Batch 内正样本数量不足，跨 Batch 正样本的可靠性/分布和排序目标仍可能与官方换装检索不一致。

在严格因果意义上，本轮没有另跑一套同代码、同初始化、`lambda_ndp=0` 的完整 60 epoch 配对控制；因此“新增 NDP 路径参与优化”有日志和梯度证据，但应将具体指标差异表述为固定 Bio+GPC 对照下的单 Seed 结果，而不是过强的因果结论。

## 6. 结论

NDP_SHN_BANK 已完成要求的 Smoke、train-only 验证、LTCC/PRCC 正式 60 epoch 训练、统一评估、Bio-only 补评估和完整结果整理。最终结果支持将 Bio、固定 Bio+GPC 和 NDP_SHN_BANK 三阶段如实纳入论文消融表，但当前 NDP_SHN_BANK 不是稳定的主结果：LTCC CC mAP 下降 `1.4563 pp`，PRCC C→A Rank-1 下降 `0.7621 pp`。建议论文主线继续使用固定 GPC 的已验证结果，并将 NDP_SHN_BANK 作为失败/边界条件消融，暂不继续堆叠相似的 NDP Margin 变体。
