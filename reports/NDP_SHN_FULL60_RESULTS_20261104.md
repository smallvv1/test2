# NDP-SHN 正式实验报告（seed=1234）

NDP-SHN 已完成 LTCC、PRCC 各 60 epoch 训练和官方评估。该方法参考 [ANFI](https://arxiv.org/abs/2607.25407) 的邻域关系可靠性视角与 [MINER](https://arxiv.org/abs/2610.28941) 的软困难样本、多正样本对比思想；未加入复杂网络。

## 方法与公平性

- GPC 代码、采样选择、正样本选择和权重 `0.1` 完全不变，基础损失仍为 `CE + Triplet + clipBio + 0.1 * GPC-v1`。
- 第 1–2 epoch warm-up；从第 3 epoch 起仅用当前固定训练 Batch 的学生特征构造身份中心，选择最多 3 个不同 PID 的软困难负身份，最多 8 个锚点，使用全部同身份正样本。负身份为 detached softmax 权重（`tau=0.2`），损失使用 live student 特征，`lambda_ndp=0.001`。
- 训练、身份中心和负身份选择只使用 train split，未读取官方 query/gallery；seed=`1234`，复用既有共同初始化，两张 PPU 正式训练 60 epoch。
- LTCC state hash：`c593b713f631fcc78087095d160b62de9ef3196cb7cefd0495d0daa62cc83202`；PRCC state hash：`6567bd0d54e6919be355757c5a2d7fb2e5d1c2128368357e51b7047f5ded97fc`。
- 训练期间只使用单文件 `resume_latest.pth`，完成后保存 `model_final.pth` 并清理 resume。正式目录 `RUN_COMPLETE.json` 标记 `per_rank_checkpoint_files_created=false`，扫描未发现 `rank*state.pt/pth`。

Smoke 和 3 epoch train-only 验证通过。第 3 epoch：LTCC `ndp=0.625882`、`active=4748/4748`；PRCC `ndp=0.248358`、`active=4408/4408`，证明新增损失有实际梯度路径。

## 正式训练状态

| 数据集 | epoch / global step | 最终 train loss | 最终 NDP loss | 最终有效关系 | AMP skipped steps | model SHA-256 前 16 位 |
|---|---:|---:|---:|---:|---:|---|
| LTCC | 60 / 35760 | 5.049144 | 0.223405 | 4736/4736 | 15 | `0e52ceebb11a892d` |
| PRCC | 60 / 33060 | 7.179844 | 0.064462 | 4408/4408 | 12 | `409722453541985f` |

## 官方结果

百分比单位；差值为 `NDP-SHN - 固定 Bio+GPC`，单位为百分点。固定 Bio+GPC 数值来自历史正式评估 JSON。

| 数据集 / 协议 | 固定 Bio+GPC Rank-1 | NDP-SHN Rank-1 | ΔRank-1 | 固定 Bio+GPC mAP | NDP-SHN mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.3813 | 84.1785 | -0.2028 | 54.1862 | 54.1941 | +0.0079 |
| LTCC CC | 55.3571 | 56.6327 | **+1.2755** | 33.9367 | 33.4036 | **-0.5331** |
| PRCC Same-Clothes | 100.0000 | 100.0000 | +0.0000 | 99.4787 | 99.4692 | -0.0095 |
| PRCC C→A | 64.0982 | 64.1547 | +0.0565 | 63.4143 | 63.6552 | +0.2409 |

仓库评估器未提供 mINP。官方日志目录为 `/home/DIFFER/log_NDP/ndp_shn_full60_seed1234_20261104/{LTCC,PRCC}/`。

## 归因判断

NDP-SHN 在训练中持续激活并通过梯度检查，因此 PRCC C→A 的小幅变化可以归因于新增 SHN 路径，而不是未激活分支。但 LTCC CC 只有 Rank-1 上升、mAP 下降，且 General/PRCC Same-Clothes 没有普遍增益。本轮结论是“部分有效、尚不足以作为稳定论文主方法”；固定 GPC 仍应作为论文主结果，NDP-SHN 作为真实单 seed 探索/边界证据保留。

最终模型 SHA-256：LTCC `0e52ceebb11a892d3305129b499f85fc606be7a56ff4dd5d6b9f3213872dc4e7`；PRCC `409722453541985fc5516dbb58c9b9c2a94e6973c0f2a7dc33d0bdd80ac88469`。未将模型上传仓库。
