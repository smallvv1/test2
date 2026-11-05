# NDP-SHN λ=0.005 正式实验报告（seed=1234）

本轮仅验证提高 NDP-SHN 权重是否有效：将 `lambda_ndp` 从 `0.001` 改为 `0.005`，多正样本损失、当前学生身份中心、Top-3 困难负身份、最多 8 个 Anchor、训练采样和 GPC 全部保持不变。LTCC、PRCC 均完成正式 60 epoch 和官方评估。

## 公平性与实现变更

- 基础目标保持 `CE + Triplet + clipBio + 0.1 * GPC-v1`；GPC 代码、正样本选择器、采样方式和权重 `0.1` 未修改。
- NDP-SHN 仍从第 3 epoch 起工作：使用当前学生 Batch 特征建立身份中心，选择最多 3 个不同 PID 的近邻负身份，最多 8 个 Anchor 和全部同 PID 正样本，负身份采用 detached softmax 权重（`tau=0.2`）。本轮只改变 `lambda_ndp=0.005`。
- 训练、身份中心和困难负身份选择只使用 train split；未读取官方 query/gallery 进行选参。seed=`1234`，复用既有 Bio+GPC 共同初始化。
- 训练只使用 PPU。正式输出仅保留单文件 `resume_latest.pth`（恢复期间）和 `model_final.pth`（最终模型）；`RUN_COMPLETE.json` 标记 `per_rank_checkpoint_files_created=false`，扫描未发现 `rank*state.pt/pth`。

## Smoke 与 train-only 检查

Smoke Test 20 steps 通过。3 epoch train-only 验证中，LTCC 第 3 epoch 的原始 NDP loss 为 `0.625796`、PRCC 为 `0.247829`，对应加权贡献约为 `0.003129` 和 `0.001239`；NDP 有有效梯度，未出现非有限值、CE/Triplet 异常或 AMP 异常。

## 正式训练状态

| 数据集 | epoch / global step | 最终 train loss | 最终 NDP raw loss | NDP 特征梯度范数 | 主特征梯度范数 | NDP/主梯度比 | 最终激活关系 | skipped steps | model SHA-256 前 16 位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LTCC | 60 / 35760 | 5.294614 | 0.284960 | 0.027566 | 0.145929 | 0.188897 | 4736/4736 | 14 | `81103dd1c2ed1927` |
| PRCC | 60 / 33060 | 7.180291 | 0.064240 | 0.005641 | 0.052546 | 0.107353 | 4408/4408 | 12 | `72ce71d9bbf23079` |

AMP 在两个正式实验中均保持有效；训练完成 60/60，最终模型和评估 JSON 均已生成。

## 官方评估结果

百分比单位；差值为相应 NDP 方法减固定 Bio+GPC，单位为百分点。固定 Bio+GPC 和 λ=0.001 NDP-SHN 均来自已完成的历史正式 JSON。

| 数据集 / 协议 | 固定 Bio+GPC Rank-1 | λ=0.001 Rank-1 | λ=0.005 Rank-1 | Δ Rank-1 vs B0 | 固定 Bio+GPC mAP | λ=0.001 mAP | λ=0.005 mAP | Δ mAP vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.3813 | 84.1785 | 85.8012 | +1.4199 | 54.1862 | 54.1941 | 52.5867 | -1.5996 |
| LTCC CC | 55.3571 | 56.6327 | 51.5306 | **-3.8265** | 33.9367 | 33.4036 | 28.8818 | **-5.0549** |
| PRCC Same-Clothes | 100.0000 | 100.0000 | 100.0000 | +0.0000 | 99.4787 | 99.4692 | 99.4141 | -0.0646 |
| PRCC C→A | 64.0982 | 64.1547 | 64.8038 | +0.7056 | 63.4143 | 63.6552 | 63.7275 | +0.3132 |

相对 λ=0.001 NDP-SHN，λ=0.005 在 LTCC CC 进一步下降 `5.1020` Rank-1 / `4.5218` mAP；PRCC C→A 上升 `0.6492` Rank-1 / `0.0722` mAP。

## 结论

提高 NDP 权重确实使 NDP 分支保持有效梯度，但没有带来可信的跨数据集增益：PRCC C→A 有小幅改善，LTCC CC 却出现明显且同时发生的 Rank-1、mAP 退化，LTCC General 的 Rank-1 上升也伴随 mAP 下降。因此 λ=0.005 不应作为正式主方法，也不再继续扫描 NDP 权重。

下一步若继续 NDP，应改进多正样本损失本身的可靠性/低干扰形式，而不是继续调整静态权重；固定 Bio+GPC 仍是当前论文主对照。完整日志保存在服务器目录：

`/home/DIFFER/log_NDP/ndp_shn_lambda005_full60_seed1234_20261105/`

本报告只上传小体积摘要，正式模型未上传仓库。
