# NDP_SHN_MP_ABM：完整实验与论文收尾报告

日期：2026-11-10  
代码实现提交：`750fa29`（`Add anchor-aware SHN-MP boundary mining arm`）  
正式实验：seed=1234，双 PPU，60 epoch，统一 evaluator。

## 1. 结论先行

本轮实现并完成了独立的 `NDP_SHN_MP_ABM` 分支。它保留原 SHN-MP 的多正样本均衡 Softplus、Top-3、`tau=0.2`、最多 8 个 Anchor、`lambda_ndp=0.001` 和 epoch 3 启用，仅改变局部关系选择：每个 Anchor 独立计算到其他 PID 中心的相似度，并在稳健局部 Margin 边界带内选择关系，同时限制同一 PID 的重复占位。

代码、Smoke Test、短程 train-only 诊断和 LTCC/PRCC 正式 60 epoch 均已完成；但 ABM 没有取得可信增益：

- LTCC CC：`51.0204/29.2912`，相对 Bio+GPC 为 `-4.3367/-4.6455 pp`。
- PRCC C→A：`63.5337/63.3451`，相对 Bio+GPC 为 `-0.5645/-0.0692 pp`。

因此本轮不再继续堆叠边界权重、历史 Margin、Teacher 或 Bank 变体。当前最有价值的 NDP 对照仍是原始 SHN-MP，但它只有单 seed，且 LTCC 的 Rank-1 增益伴随 mAP 下降，尚不足以作为稳定的普适提升结论。

## 2. 方法与实现变更

ABM 与原 SHN-MP 的差异仅限于关系选择：

1. 对每个 Anchor 单独计算 Anchor 到其他 PID 中心的相似度，独立选 Top-3 软困难负身份；不再让同一 PID 下的 Anchor 共用负身份排序。
2. 使用 Anchor 到同 PID 正样本的平均相似度减去最近负身份中心相似度，构成 Anchor 级局部 Margin。
3. 以 batch 内 Margin 的稳健中位数/MAD 构造边界带，排除极端困难和低质量离群 Anchor；最多每个 PID 选择一个 Anchor。
4. 选中的关系仍使用原 SHN-MP 的实时特征、多正样本均衡 Softplus；没有冻结教师、历史 Margin 恢复、跨 epoch 风险缓存或测试集信息。
5. GPC 代码、正样本选择、权重 `0.1`、Bio/CE/Triplet、采样器均未修改。

实现文件：`scripts/ndp_v5_prcc_runner.py`、`scripts/train_ndp_listwise_full60.py`、`tests/test_ndp_shn.py`。原始 `NDP_SHN_MP` 仍可通过原 arm 完全复现，没有覆盖历史代码或模型。

## 3. 验证状态

### 3.1 代码和 Smoke Test

- `38 passed`：`tests/test_ndp_shn.py` 与 `tests/test_ndp_v5_b0_loss_contract.py`。
- 双 PPU LTCC Smoke Test 通过，epoch 1 按设计不激活 NDP。
- LTCC/PRCC 各完成 3 epoch train-only 检查；CE、Triplet、GPC 正常，NDP 梯度有限且非零。
- 未生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。

### 3.2 短程 train-only 诊断

| 数据集 | epoch 3 候选 Anchor | 选中 Anchor/PID（跨 batch 累计） | Top-3 变化 batch | NDP 梯度 | Main 梯度 | 梯度比 | 梯度余弦 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LTCC | 4748 | 2284 / 2284 | 594 | 0.02210 | 0.36371 | 0.06075 | 0.55584 |
| PRCC | 8816 | 2181 / 2181 | 551 | 0.01495 | 0.24795 | 0.06028 | 0.35421 |

正式训练最后一个 epoch 的关系统计也表明 ABM 并非“没有激活”：

- LTCC：596 个 batch 中 592 个激活，候选 Anchor 4736，选中 2284 个 Anchor/PID，Top-3 变化 592 个 batch。
- PRCC：551 个 batch 全部激活，候选 Anchor 8816，选中 2179 个 Anchor/PID，Top-3 变化 551 个 batch。

这说明实现满足了 Anchor 级选择、PID 去重和反向传播要求；失败来自选择到的局部关系与最终换装检索目标不一致，而不是 NDP 没有执行或梯度为零。

## 4. 四协议正式结果

数值为 Rank-1/mAP（百分比）。差值均为百分点（pp）。Bio 和 Bio+GPC 使用已有正式模型及统一 evaluator；没有重新训练已有 Bio 或 Bio+GPC。

| 协议 | Bio | Bio+GPC | SHN-MP | ABM |
|---|---:|---:|---:|---:|
| LTCC General | 83.5700/52.4800 | 84.3813/54.1862 | 86.0041/55.3313 | 85.5984/52.7603 |
| LTCC Clothes-Changing | 48.9796/29.1699 | 55.3571/33.9367 | 57.3980/33.4117 | 51.0204/29.2912 |
| PRCC Same-Clothes | 100.0000/99.2069 | 100.0000/99.4787 | 100.0000/99.5250 | 100.0000/99.5534 |
| PRCC C→A | 64.6063/63.0032 | 64.0982/63.4143 | 64.1547/63.7777 | 63.5337/63.3451 |

### 4.1 模块增量

| 协议 | GPC 提升（GPC−Bio） | SHN-MP 增量（SHN-MP−GPC） | ABM 增量（ABM−GPC） | ABM 总提升（ABM−Bio） |
|---|---:|---:|---:|---:|
| LTCC General | +0.8114/+1.7062 | +1.6227/+1.1451 | +1.2170/−1.4259 | +2.0284/+0.2803 |
| LTCC Clothes-Changing | +6.3775/+4.7668 | +2.0408/−0.5249 | −4.3367/−4.6455 | +2.0408/+0.1213 |
| PRCC Same-Clothes | +0.0000/+0.2718 | +0.0000/+0.0463 | +0.0000/+0.0747 | +0.0000/+0.3465 |
| PRCC C→A | −0.5080/+0.4111 | +0.0565/+0.3634 | −0.5645/−0.0692 | −1.0725/+0.3418 |

ABM 的 LTCC General mAP 也比 Bio+GPC 下降 1.4259 pp；因此不能把它解释为只改善换装局部关系。

## 5. Query 级检索分析

分析只在正式 evaluator 生成的 query/gallery 特征上进行，属于结果审计，不参与训练、选参或关系挖掘。

### LTCC Clothes-Changing（392 个有效 query）

| 模型 | Rank-1 正确数 | 相对 Bio+GPC 的 Rank-1 gained/lost | AP improved/decreased/unchanged | 平均 AP 变化 |
|---|---:|---:|---:|---:|
| Bio+GPC | 217/392 | — | — | — |
| SHN-MP | 225/392 | 29/21（净 +8） | 183/209/0 | −0.5249 pp |
| ABM | 200/392 | 25/42（净 −17） | 103/289/0 | −4.6455 pp |

ABM 相对 SHN-MP 另有 13 个 Rank-1 gained、38 个 lost，净减少 25 个；AP 改善/下降为 90/302，平均 AP 再下降 4.1205 pp。ABM 的平均首个正样本排名从 Bio+GPC 的 48.88、SHN-MP 的 48.06 退化到 73.83，且 AP 中位数从 0.2953 降到 0.2316，说明 mAP 下降来自大量 query 的整体排序变差，而不是少数 Rank-1 异常。

### PRCC C→A（3543 个有效 query）

| 模型 | Rank-1 正确数 | 相对 Bio+GPC 的 Rank-1 gained/lost | AP improved/decreased/unchanged | 平均 AP 变化 |
|---|---:|---:|---:|---:|
| Bio+GPC | 2271/3543 | — | — | — |
| SHN-MP | 2273/3543 | 60/58（净 +2） | 1585/1350/608 | +0.3634 pp |
| ABM | 2251/3543 | 60/80（净 −20） | 1491/1432/620 | −0.0692 pp |

ABM 相对 SHN-MP 为 60 个 Rank-1 gained、82 个 lost，净减少 22 个；AP 改善/下降为 1387/1537，平均 AP 下降 0.4326 pp。PRCC 的平均首个正样本排名虽然为 35.61，优于 Bio+GPC 的 38.64，但 Rank-1 和 mAP 并未改善，说明单一首正样本排名不能替代完整 AP 分布。

## 6. 开销与可归因性

- ABM 不改变模型结构，推理时不执行 NDP，故没有新增推理模块或额外检索开销；评估仍使用原始 1024 维特征和官方协议。
- 最终模型大小：LTCC 1,215,890,852 bytes，PRCC 1,217,202,916 bytes，与原架构一致。
- 双 PPU 实际训练墙钟时间约为 LTCC 2 小时 23 分、PRCC 2 小时 26 分；历史 SHN-MP 约为 LTCC 2 小时 26 分、PRCC 2 小时 26 分，未观察到有意义的额外训练开销（受机器负载和 AMP skip 影响，不能视为严格 benchmark）。
- 正式模型 SHA256：
  - LTCC ABM：`231a998c0f7a3f91c8f252286da399b7f68176264d9abff46704c96ef8241c59`
  - PRCC ABM：`52d43fcc738b0e4518fd2852097c0ab409ada332c60e10989dc6bd3f516f0769`
- 两套正式运行均使用共同初始化、seed=1234、`lambda_gpc=0.1`、`lambda_ndp=0.001`，并在元数据中记录 `query_gallery_used_during_training=false`、`per_rank_checkpoint_files_created=false`。

## 7. 为什么 ABM 失败

ABM 的代码路径、梯度和 Anchor 级负身份排序均已验证有效，但它把 batch 内即时身份中心当作了局部判别边界的可靠代理。LTCC 的跨衣变化使单张图片到其他 PID 中心的相似度更容易受姿态、衣服和 batch 组成影响；稳健边界带虽然排除了极端样本，却仍然可能持续选中不稳定的伪边界关系。与此同时，“每个 PID 至多一个 Anchor”改变了原 SHN-MP 的关系覆盖方式，导致有限的 NDP 监督预算被分配给与最终跨衣 AP 不一致的局部排序。

这与历史结果一致：增加冻结 Bank、历史 Margin、风险缓存或更强门控都没有稳定解决 LTCC mAP；直接把 Anchor 视角的边界关系作为监督也出现明显退化。因此本轮不再继续做第二个边界权重消融或扫描 NDP 权重。

## 8. 论文证据是否充分、仍缺什么

当前证据足以支持：GPC 是有效的固定模块；SHN-MP/RELIABLE/RISK 等 NDP 变体的结果和失败模式可以作为消融材料；ABM 是一个实现正确但效果失败的 Anchor-aware 负身份选择消融。

当前证据不足以支持“新增 NDP 在两个数据集上稳定提升”的论文结论，原因是：

1. SHN-MP 只有 seed=1234，LTCC 虽然 Rank-1 +2.0408 pp，但 mAP −0.5249 pp；PRCC 仅有小幅增益。
2. 尚缺一组与当前 SHN-MP 代码、初始化、步数、AMP 和优化器完全一致的严格 `lambda_ndp=0` 配对对照。这是确认 SHN-MP 增益是否真正来自 NDP 的首要缺口。
3. 若要把 NDP 写成稳定贡献，还需要至少一个额外 seed；但这不是本轮继续开发的必要条件，也不应通过测试集选择方案替代。

建议论文主线将 GPC 作为主要有效贡献，NDP 作为经过系统探索的局部判别增强尝试：报告 SHN-MP 的单 seed 正向/负向双面结果、RISK 的 LTCC mAP 小幅提升，以及 ABM 的失败证据；不要宣称 NDP 已经稳定改善 LTCC 和 PRCC。

## 9. 归档位置

正式结果目录：

- `/home/DIFFER/log_NDP/ndp_shn_mp_abm_full60_seed1234_20261110/LTCC`
- `/home/DIFFER/log_NDP/ndp_shn_mp_abm_full60_seed1234_20261110/PRCC`

Query 审计：

- `/home/DIFFER/log_NDP/paper_audit_abm_20261110/LTCC`
- `/home/DIFFER/log_NDP/paper_audit_abm_20261110/PRCC`

GitHub 仅归档小体积的配置、指标、评估 JSON、Query 汇总和本报告；大型 `model_final.pth` 保留在本地正式结果目录，不上传仓库。
