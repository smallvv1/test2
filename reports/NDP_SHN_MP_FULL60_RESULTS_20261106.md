# NDP_SHN_MP 正式实验报告

日期：2026-11-06  
实验目录：`/home/DIFFER/log_NDP/ndp_shn_mp_full60_seed1234_20261105/`  
数据协议：沿用仓库原始 LTCC、PRCC 官方检索评估协议；训练和邻域选择只使用 train split。

## 方法变更

本轮只修改 SHN 的正样本聚合方式，GPC 代码、正样本选择、采样方式和权重 `0.1` 均未改变。

- 保留身份中心、Top-3 困难负身份、8 个 Anchor、`tau=0.2` 和 epoch 3 后激活策略。
- 原 NDP_SHN 对同 PID 正样本使用 `logsumexp`，梯度容易集中到最相似的一个正样本。
- NDP_SHN_MP 对每个同 PID 正样本分别计算同一负身份混合下的 `softplus` 排序项，再取均值，使合法正样本均衡参与；没有固定 Margin 目标，也没有额外网络或分类器。
- NDP 权重恢复为 `lambda_ndp=0.001`。

## 训练与完整性检查

- LTCC：60/60 epoch，35,760 steps，PPU 双进程训练。
- PRCC：60/60 epoch，33,060 steps，PPU 双进程训练。
- Seed：`1234`；使用已有共同初始化，未重新训练 Bio+GPC B0。
- Smoke Test、3 epoch train-only 验证通过；新增损失可反向传播，CE/Triplet 无 NaN 或异常爆炸。
- 正式训练中 NDP 梯度持续有效：最终 LTCC `ndp_feature_grad_norm=0.02504`，PRCC `0.01658`；最终 NDP 原始损失分别为 `0.25508`、`0.19708`。
- NDP 从 epoch 3 开始实际激活：LTCC 共 34,409 个 active batch、275,220 条关系；PRCC 共 31,946 个 active batch、255,600 条关系。AMP 跳步分别为 16、11，训练正常完成。
- 只生成单文件 `model_final.pth`（训练期间使用 `resume_latest.pth`）；未生成任何 `rank*state.pt/pth`。
- `RUN_COMPLETE.json` 标记 `query_gallery_used_during_training=false`、`per_rank_checkpoint_files_created=false`。

## 正式评估结果

数值均为百分比，格式为 `Rank-1 / mAP`。

| 数据集协议 | 固定 Bio+GPC | NDP_SHN（λ=.001） | NDP_SHN_MP | MP 相对 Bio+GPC |
|---|---:|---:|---:|---:|
| LTCC General | 84.3813 / 54.1862 | 84.1785 / 54.1941 | **86.0041 / 55.3313** | **+1.6227 / +1.1451** |
| LTCC CC | 55.3571 / 33.9367 | 56.6327 / 33.4036 | **57.3980 / 33.4117** | **+2.0408 / -0.5249** |
| PRCC Same-Clothes | 100.0000 / 99.4787 | 100.0000 / 99.4692 | **100.0000 / 99.5250** | **+0.0000 / +0.0463** |
| PRCC C→A | 64.0982 / 63.4143 | 64.1547 / 63.6552 | **64.1547 / 63.7777** | **+0.0565 / +0.3634** |

模型 SHA256：

- LTCC：`1202b35434b953e5f5de6fbaf2bbc7ebbb32a84845668973bf70d0cd54c5d990`
- PRCC：`784e75df9b35a67a9b3c229488101efa6cdbea0a043d427dae81dac3c5317b41`

## 结论

NDP_SHN_MP 相比固定 Bio+GPC 在 LTCC General 和 PRCC 两个协议上取得提升；LTCC CC 的 Rank-1 也提升 2.04 个百分点，但 mAP 下降 0.52 个百分点，因此不能宣称 LTCC 换装检索质量全面改善。相比旧 NDP_SHN λ=.001，LTCC CC Rank-1 进一步提升 0.7653 个百分点、mAP 基本持平（+0.0082），PRCC C→A mAP 提升 0.1225 个百分点。

由于本轮只有新增 NDP_SHN_MP 的正式单 Seed，没有同一轮 `lambda_ndp=0` 的配对重跑，结果支持“新增 NDP 路径有效且确实参与优化”，但对 LTCC CC Rank-1 的单独因果归因仍应保守表述。当前最可靠的论文结论是：均衡多正样本聚合能改善总体/跨衣 Rank-1，并在 PRCC 同时改善 mAP；LTCC CC 的 mAP 仍是未解决问题，不应继续仅靠扫描 NDP 权重宣称稳定增益。
