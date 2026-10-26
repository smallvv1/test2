# NDP PRCC 严格配对三组对照结果

## 结论摘要

六组正式训练均完成，状态为 `PASS`。本实验确实消除了此前的 optimizer/model-only warm-start 混杂：每个 seed 先由修复后的 B0 三 raw epoch 模型权重建立一份新的、共享的 optimizer/scheduler/GradScaler/RNG 状态，再分别启动 B0、N5、NF0。

在严格配对结果中，没有出现两个 seed、两个节点评估都同时优于 B0 的 NDP 方案。N5 的影响较小但不一致；NF0 虽显著提高了激活关系比例，却在第3节点明显伤害 Seed4321，不能据此继续延长训练或扫描阈值。

## 实验身份与可复现信息

- 远端 main：`75cb59ddb63aee0bd31f0756fa28fa8d739e2b11`。
- 实验目录：[ndp_prcc_strict_pair_20261026_030056](/home/DIFFER/log_NDP/ndp_prcc_strict_pair_20261026_030056)
- 数据：PRCC train-only，17,896 张训练图；PRCC 官方 C→A 换装评估。
- 初始状态：修复后 B0 三 raw epoch、源 `epoch=3`、`global_step=1653` 的 model-only 权重 warm-start；之后为每个 seed 新建并共享完整 rank state。不是历史 E60 的无缝续训。
- Seed1234 源模型 SHA256：`b803504065e60185ced22654bdbb74f47bc4f404b4a3b8bf1e34a36c7ce70b18`。
- Seed4321 源模型 SHA256：`9c1bbfcd4830cd2f109a5c78a1ee6d59986af798d8c0f651e729b46e81d03cac`。
- 六组正式训练使用的 runner SHA256：`c8f3a67139a83d93f7f185ded0d23b738e3e055a30f274a2ca5cb70b2980dc93`。
- 训练完成后新增 raw-batch 摘要校验的 runner SHA256：`39e77b6e6d14eebea089d06055938eea9b33fc9636e5e51208d38876ebb61efc`；该改动只用于只读校验，没有改变已完成训练。
- 共同状态生成脚本 SHA256：`eff848d020ea7604c32b9a3131a0edd2ff0444882f2de99e7e84a48fc0b619d1`。
- 固定 GPC 源 SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`，未修改。
- B0/NDP Loss contract 单测：`4 passed`。

真实运行的核心命令形式为：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train \
  --seed {1234,4321} --arm {B0,N5,NF0} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --resume-checkpoint common_seed{seed} --epochs 5 \
  --eval-epochs 3,5 --ndp-epsilon 0.02
```

## 严格配对检查

固定 GPC phase 不包含 arm：
`ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed{seed}`。

前 100 个 raw batch、两个 rank 的 GPC 关系检查：

| Seed | 比较 batch | fixed GPC mismatch | 旧 arm-dependent phase mismatch | 图像索引计划摘要 |
|---|---:|---:|---:|---|
| 1234 | 100/rank | 0 | 200/200 | rank0 `d1aa410a...f5db`；rank1 `0935b231...19e18` |
| 4321 | 100/rank | 0 | 200/200 | rank0 `6fd74a74...732b`；rank1 `53fb44af...1934` |

图像索引摘要来自同一个 seed/run-id 下的共享 loader raw-batch 计划；三臂均使用该 arm-independent 计划。fixed GPC 关系完全一致，旧的把 arm 放入 PHASE 的做法则确实产生关系差异。正式训练每臂均为 2,755 raw batches、2,754 successful optimizer updates；每臂各有 1 次 AMP skip，位置保存在对应 `amp_events.csv`，未补跑掩盖。

## PRCC C→A 结果

数值为百分比；括号内为相对同 seed 同节点 B0 的百分点差。

| Seed | continuation 节点 | B0 Rank-1 / mAP | N5 Rank-1 / mAP | NF0 Rank-1 / mAP |
|---:|---:|---:|---:|---:|
| 1234 | 3 | 63.251 / 58.548 | 63.364 (+0.113) / 58.076 (-0.472) | 63.167 (-0.085) / 58.187 (-0.361) |
| 1234 | 5 | 63.195 / 59.581 | 63.251 (+0.056) / 59.162 (-0.419) | 63.675 (+0.480) / 59.558 (-0.022) |
| 4321 | 3 | 63.901 / 58.436 | 63.618 (-0.282) / 58.401 (-0.035) | 61.897 (-2.004) / 57.744 (-0.693) |
| 4321 | 5 | 64.409 / 60.155 | 64.409 (+0.000) / 60.005 (-0.150) | 64.183 (-0.226) / 59.665 (-0.490) |

仓库中的 PRCC evaluator 没有暴露 mINP，故本轮 mINP 记为 unavailable，没有用其他协议结果替代。

## NDP 实际参与情况

| Seed | Arm | 节点 | 激活 Batch 比例 | 选中关系 | 激活关系 | 激活/选中 | NDP mean loss | 当前 margin | Purity@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | N5 | 3 | 27.83% | 108,925 | 1,893 | 1.738% | 0.000126 | 0.01027 | 0.30654 |
| 1234 | N5 | 5 | 25.23% | 72,592 | 1,148 | 1.581% | 0.000108 | 0.01249 | 0.30661 |
| 1234 | NF0 | 3 | 91.68% | 108,925 | 14,594 | 13.398% | 0.000986 | 0.01033 | 0.30656 |
| 1234 | NF0 | 5 | 87.43% | 72,592 | 8,078 | 11.128% | 0.000845 | 0.01253 | 0.30658 |
| 4321 | N5 | 3 | 28.13% | 109,678 | 2,048 | 1.867% | 0.000150 | 0.01076 | 0.30517 |
| 4321 | N5 | 5 | 25.36% | 72,969 | 1,215 | 1.665% | 0.000128 | 0.01305 | 0.30523 |
| 4321 | NF0 | 3 | 92.77% | 109,678 | 14,907 | 13.592% | 0.001077 | 0.01075 | 0.30514 |
| 4321 | NF0 | 5 | 88.61% | 72,969 | 8,222 | 11.268% | 0.000902 | 0.01294 | 0.30518 |

B0 两个 seed 两个节点均为 NDP selected/active/loss = 0，且 B0 总损失严格为 `main_total`。Smoke Test 中 N5 与 NF0 均 20/20 更新完成；N5 的 NDP 路径有非零 loss，NF0 的 NDP feature gradient 在前20步最大约 `0.00262`，确认当前 anchor 到 EVA02 的反向路径有效。

正式训练前20步的 NDP feature-gradient 统计如下：

| Seed | Arm | mean | max |
|---:|---|---:|---:|
| 1234 | N5 | 0.0001466 | 0.001064 |
| 4321 | N5 | 0.0002877 | 0.001416 |
| 1234 | NF0 | 0.001094 | 0.002622 |
| 4321 | NF0 | 0.001324 | 0.005670 |

NF0 的确显著增加了触发和梯度，但 PRCC 性能没有随激活增加而稳定改善；这说明“保护更多关系”本身不是足够的性能依据。

## 失败与排除项

- 初始并发 Smoke Test 目录曾被中断；只采用随后串行重跑的 `smoke_n5_seed1234_retry` 与 `smoke_nf0_seed1234_retry`，两者均 PASS，未纳入正式结果。
- AMP skip 位置因 arm/seed 不同而不同，因此不能把六条数值轨迹宣称为逐 batch 数值完全相同；raw batch 预算和固定 loader/GPC 计划保持一致，skip 位置如实记录。
- PPU/EVA02 既有 A/A 非确定性仍存在，本实验没有重新进行算子审计；小幅变化不视为已证实收益。
- 未执行 LTCC、更多 seed 或 60 epoch；也没有根据 PRCC 测试结果改参数。

## 决策

本次严格统一起点后，N5 和 NF0 都不能通过“双 seed、双节点 Rank-1 与 mAP 同时提升”的门禁。NF0 只证明提高了激活率和 NDP 梯度，并未证明更好的身份判别保护；Seed4321 第3节点的 Rank-1 下降 `2.004 pp`，应停止继续强化该方向。

下一步不建议继续扫描 epsilon、lambda、Floor 或训练时长，也不建议直接扩展到 60 epoch。若继续研究，应转向关系级筛选：找出确实在后续退化、且 NDP 梯度与主 Triplet 方向不冲突的局部身份关系，并在独立 seed 上进行预注册的小规模验证。当前结论是：NDP 已真实参与训练，但在统一起点的 PRCC 三臂对照中尚无稳定性能增量证据。
