# NDP 阶段 A/A 数值可重复性审计

## 结论

`A/A_REPRODUCIBILITY: FAIL`

本轮没有启动 NDP optimizer-step，也没有启动 30/60 epoch 训练。A/A 只使用固定的
Bio+GPC-v1 B0 E60 完整状态、LTCC train-only 物化输入和原始 I3 matched-budget
关系选择。

## 实验对象与来源

- seed1234 checkpoint：`/home/DIFFER/outputs/gpc_v2_ltcc_20261015_155822/training/two_seed/B0_seed1234/rank0_state_epoch060.pt`，SHA256 `e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`。
- seed4321 checkpoint：同目录 `B0_seed4321/rank0_state_epoch060.pt`，SHA256 `d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 两个 seed 均从完整 model、optimizer、scheduler、GradScaler 和 RNG 字段恢复；模型计算使用两个 `PPU-ZW810E`，未发生 CPU fallback。
- 采样和关系先物化到 `/home/DIFFER/outputs/ndp_aa_reproducibility_20261021_030947/materialized_seed{1234,4321}.pt`，每个分支使用相同物化张量。
- 实际诊断源码：`/home/DIFFER/scripts/ndp_aa_reproducibility.py`，SHA256 `5b654d3b8fdfdaebd3c4e94adbfd81343b1f0f9ee4ea9c80b4e7caaf460044ca`。

## 逐 seed 结果

| seed | 有效更新 A0/A1 | AMP skip | 输入哈希 | 关系哈希 | 首处分叉 | 第1步 forward | 第2步 forward |
|---:|---:|---:|---|---|---|---|---|
| 1234 | 20/20 | 0/0 | 一致 | 一致 | step 1 梯度 | 一致 | 不一致 |
| 4321 | 20/20 | 0/0 | 一致 | 一致 | step 1 梯度 | 一致 | 不一致 |

seed1234 的第1步最大差异包括：`parameter_grad_norm=54.4483`、`base_loss=0.00693`；
seed4321 包括：`parameter_grad_norm=415.9699`、`base_loss=0.01111`。这些是两条完整
轨迹在 20 步窗口内的最大逐步差异，不是把差异隐藏在最终指标中。

两种 seed 的第1步都表现为：模型 forward 输出哈希和标量 loss 尚一致，但抽样层梯度
哈希已经不同；`patch_embed.proj`、`blocks.0.attn.q_proj/proj`、
`blocks.22.attn.q_proj` 等层出现差异，`blocks.22.attn.proj` 和 `head` 在第1步仍可
相同。第2步开始 forward hook/输出也分叉。因此证据将首处分叉定位到 PPU backward
数值路径，而不是输入、GPC 关系或 AMP found_inf 决策；现有记录还不足以归因到单个
具体算子。

## 运行异常的边界

首次 parent 启动时没有留下完整父级 summary；随后原始 seed4321 child JSON 也已落盘，
并显示各自 20/20 更新、无 skip。为使最终配对不依赖 parent 的中断，另外在两个新独立
进程中完成 retry2/retry3 的 seed4321 `aa0/aa1`，并用该完整 pair 纳入上表。父级
summary 缺失属于监督器可观测性问题，不作为模型或算法证据；A/A 结论来自两个完整
child JSON。不能仅凭该现象归因于具体 PPU 算子。

## 科学判定

输入、关系、初始状态和有效更新数均可控，但反向梯度不具备逐位或可接受的严格重复性。
历史 A/A 的 found_inf 差异与本轮的无 skip 梯度分叉是两类现象：本轮证明即使不跳步，
PPU backward 仍会先产生数值差异。因此当前不允许开展 B0/NDP 因果效果比较。

