# NDP 多层级真实 Triplet 覆盖审计

## 结论

当前没有历史逐步 miner 日志，因此本报告是按固定 DIFFER sampler 重建 epoch 60/61、两个 rank 的只读近似；不能标记为 observed historical coverage，也不能推出真实梯度覆盖率。

| seed | batches | miner rows | 固定关系 | B 身份共同出现 | D hard-negative PID | C 固定图像共同出现 | F 完整关系选中 | 有任意 Triplet 的 anchor | 随机 PID 匹配 hard-PID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 2384 | 19072 | 2560 | 8.3984% | 7.6953% | 0.1953% | 0 | 398/512 | 6.8359% |
| 4321 | 2384 | 19072 | 2560 | 8.5938% | 7.8125% | 0.0000% | 0 | 398/512 | 6.8750% |

层级定义：A 是 anchor 曝光；B 是任意该负 PID 同 batch；C 是固定负图像同 batch；D 是 hard negative PID 等于目标 PID；E 是 hard negative 图像等于固定图像；F 要求 hard positive 与 hard negative 同时落入固定候选。两 seed 的 B/D 约为 8.40%/7.70% 与 8.59%/7.81%，但 C/E/F 为 5/2560、0/2560 和 0。398/512 anchor 通过其他关系获得 Triplet 约束，说明图像精确覆盖为零不等于 anchor 完全没有 Triplet 训练。

随机 PID 数量匹配控制的 D 覆盖为 seed1234 `6.8359%`、seed4321 `6.8750%`；固定困难关系分别高出约 `0.86` 和 `0.94` 个百分点，但这是重建采样的描述性差异，不是历史因果证据。

## 重要限制

脚本当前使用 update0 归一化检索特征做近似 miner；P1 已证明这与真实 raw Triplet 输入不总相同。因此本报告可支持“关系访问机会口径”，不能支持“真实训练中 Triplet 实际覆盖比例”。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py --seed 1234 --features outputs/ndp_true_multipid_20261021_103952/full_features/seed1234/update0_full_features_seed1234.npz --graph outputs/ndp_true_multipid_20261021_103952/reference_graph/seed1234/multi_pid_reference_graph.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/miner_coverage/seed1234 --epochs 60 61`
`/usr/local/bin/python3.10 -u scripts/ndp_real_triplet_miner_coverage.py --seed 4321 --features outputs/ndp_true_multipid_20261021_103952/full_features/seed4321/update0_full_features_seed4321.npz --graph outputs/ndp_true_multipid_20261021_103952/reference_graph/seed4321/multi_pid_reference_graph.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/miner_coverage/seed4321 --epochs 60 61`
