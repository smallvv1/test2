# NDP_RANK LTCC / PRCC 正式 60 Epoch 结果

本报告对应代码提交 `3a1c2ffe394cf346d24ac2842c15919d0222b4b7`，seed=1234。LTCC 与 PRCC 的 NDP_RANK 均从正式初始化完成 60 epoch；训练只使用 train split 建图，不使用 query/gallery。输出目录未生成任何 `rank*state.pt`、`rank*state.pth` 或其他逐 Rank 状态文件。

## 方法与固定条件

基础损失固定为 `CE + Triplet + clipBio + 0.1 * GPC-v1`。NDP_RANK 使用冻结完成的 Bio+GPC 教师：教师 margin 在 `[0.02, 0.10]`，学生当前 margin `<0.05`，目标为 `clamp(0.5 * M_teacher, 0.01, 0.05)`，NDP 权重为 `0.005`。正负端点停止梯度，当前 anchor 保留梯度。EVA02、CE、Triplet、clipBio、GPC-v1、GPC 关系选择、采样器和评估协议均未修改。

代码 SHA256：

- `scripts/ndp_v5_prcc_runner.py`: `4804c776a937a382aaa4de7dee5766757b884acf7677b58293ddccce1edcfd31`
- `scripts/train_ndp_listwise_full60.py`: `45705bdc6c9123af401a44d7d8b34ee84293c7b29b303a2579d9a73a131ed31`

## 官方评估结果

差值为 NDP_RANK 减 Bio+GPC B0，单位为百分点；mINP 为仓库评估器不支持。

| 数据集 / 协议 | Bio+GPC B0 Rank-1 | NDP_RANK Rank-1 | ΔRank-1 | Bio+GPC B0 mAP | NDP_RANK mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| LTCC General | 84.381 | 84.179 | -0.203 | 54.186 | 55.193 | +1.007 |
| LTCC Clothes-Changing | 55.357 | 53.061 | -2.296 | 33.937 | 32.235 | -1.701 |
| PRCC Same-Clothes | 100.000 | 100.000 | +0.000 | 99.479 | 99.503 | +0.024 |
| PRCC C→A | 64.098 | 64.380 | +0.282 | 63.414 | 64.103 | +0.689 |

Rank-5/Rank-10：LTCC General `91.075/93.103`，LTCC Clothes-Changing `69.388/75.000`；PRCC Same-Clothes `100.000/100.000`，PRCC C→A `71.211/74.457`。mINP：Unavailable。

## 训练统计与结论

最后 epoch：LTCC NDP active batch `488/596`，active rate `81.879%`，selected/active relation `2128/1260`，NDP loss `0.017059`，NDP/main feature-grad ratio `0.042806`；PRCC 为 `130/551`、`23.593%`、`289/190`、`0.006535`、`0.091150`。全程 AMP skipped/effective updates：LTCC `14/35746`，PRCC `13/33047`。最终模型 SHA256：LTCC `0f92102db0da32e39f5823393e26703e054455f690c1e68e45a17efca251dd86`；PRCC `a2c685df02375118ec2cfec859c6a759303f3caaef831a3f3a796f481ad45d3d`。

LTCC Clothes-Changing 的 Rank-1/mAP 均下降，PRCC C→A 只有小幅上升，因此 NDP_RANK 没有得到稳定跨数据集增益，不应作为论文主结果。Bio-only 的 PRCC 数值本轮没有复测，不能填入猜测值；LTCC 历史 Bio-only 数值也不是本轮严格配对数据。

完整服务器报告：`/home/DIFFER/log_NDP/ndp_rank_full60_seed1234_20261031_183800/NDP_RANK_FULL60_RESULTS.md`。
