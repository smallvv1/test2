# 真实 Batch-Hard Triplet 覆盖审计

源码复核结果：原 Triplet 使用欧氏距离；hard positive 是 batch 内同 PID 最大欧氏距离，hard negative 是异 PID 最小欧氏距离；配置 `NO_MARGIN=True`，损失为 `SoftMarginLoss(dist_an-dist_ap)`；make_loss 没有把 feat[0] 先归一化后再交给 Triplet。此前用 cosine ReLU 代理的 T0 不能当作真实 Triplet。

本批次通过原 sampler 代码重建 epoch 60/61、rank0/1 的 batch，使用 update0 特征只读计算 miner，不调用 optimizer.step。没有历史逐步 batch/miner 日志，因此下面是“重建采样近似”，不是历史真实梯度覆盖。

| Seed | 重建 batch | miner 行 | NDP 固定关系 | 有共同 batch 机会 | 完整 miner 选中 | 有效 Triplet 直接选中 | 有任意其他 Triplet 的 anchor |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|2384|19072|2560|0|0|0|398/512|
|4321|2384|19072|2560|0|0|0|398/512|

直接固定关系覆盖为 0，不表示 Triplet 对这些 anchor 没有任何作用：两种子都有 398/512 个 NDP anchor 在其他关系上出现过 active Triplet。正确结论是：固定的多 PID 局部关系没有被该重建 Batch-Hard 直接覆盖，而 anchor 级间接约束存在；真实历史梯度覆盖仍为 `NOT_AVAILABLE`。

证据：`/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/triplet_coverage/seed1234/triplet_miner_coverage_summary.json`、逐关系 `ndp_triplet_coverage_rows.csv` 及 Seed4321 对应文件。
