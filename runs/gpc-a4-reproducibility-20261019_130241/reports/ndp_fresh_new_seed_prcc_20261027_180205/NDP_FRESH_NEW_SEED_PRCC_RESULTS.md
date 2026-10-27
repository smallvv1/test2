# NDP-Fresh Seed3456 PRCC 独立种子实验结果

## 结论

Seed3456 的独立起点、独立 t0/t1 特征、独立 K=5 图和 Top5 关系均已完成；B0、TOP5、NFRESH 三组均从同一个 t1 common state 训练 5 个 continuation raw epoch。

Seed3456 的 NFRESH 结果：

- continuation epoch 3：Rank-1 `63.53%`、mAP `59.58%`，相对 B0 为 `-0.368 / +0.354 pp`。
- continuation epoch 5：Rank-1 `64.15%`、mAP `60.60%`，相对 B0 为 `+0.734 / +0.615 pp`。

因此 Seed3456 在第 5 节点支持 NFRESH 的正向信号，但没有在第 3 和第 5 节点都同时优于 B0。结合 Seed1234、4321，NFRESH 仍然存在 seed 依赖，暂不新增 Seed5678，也不直接扩展 60 epoch。

## 实验身份与固定条件

- GitHub `main` 已核对为：`fff4d2f2b3d6b330ee155a8aabca0b5862d27d87`。
- 新实验目录：`/home/DIFFER/log_NDP/ndp_fresh_new_seed_prcc_20261027_180205/`。
- Seed：3456；训练集：17896 张 PRCC train-only 图像；未使用 query/gallery。
- 固定 run-id：`ndp_prcc_baseline_fix_20261025_130457`。
- GPC-v1 源快照 SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`。
- NDP runner SHA256：`599adbce3ad23eacd2c4dd59115feae50ccd5bba5bb82bb8353c16df939e02db`。
- Top5 reference builder SHA256：`d3b46d92767f8e1029473543496c9d792ff83d8adb38ec86a423469376b78617`。
- Common warmstart builder SHA256：`efaee6e3206ab715e1fa8e433669b6e772fe246d006309ed96d86fbbb1f14c42`。
- 测试文件 SHA256：`3bf5a90f15ff6674f66aba657dfb6b7430e6d4f767c7fb3fe01a6935f5dc48ef`。
- t1 模型 SHA256：`44bed39e6659602354f40042840334cda19619d2628a4daaadd7ff6e2fa346d3`。
- t0 特征 SHA256：`981ae3f6891aa6dedf4c169b431ff201a54581d9be31f359665e4a939a1e84e6`。
- t1 特征 SHA256：`e5a791b7aa7e2f3ab0b61d8f4468838044ca633b260e7ca8e6a24eabc635bff9`。
- Top5 图 SHA256：`f8ca6fc91e04dddf736888c644e785e29ee6734cb12f309aee86d97721640455`。

本轮仅把 runner 的 seed 参数从 `(1234,4321)` 扩展为 `(1234,3456,4321)`；没有修改 GPC、EVA02、CE、Triplet、Bio、NDP Loss 或 sampler。

## Seed3456 独立起点与参考图

先从 PRCC Bio-only common initialization 运行 B0 前缀 3 raw epoch，再从完整节点继续 1 raw epoch：

- 前缀 3 raw epoch：`effective_updates=1652/1653`，保存完整 rank state；C→A Rank-1/mAP 为 `57.9%/51.4%`。
- t1 继续 1 raw epoch：`effective_updates=551`，保存完整 rank state；C→A Rank-1/mAP 为 `62.0%/55.7%`。
- t0：前缀第 3 epoch 模型，source update `1652`。
- t1：第 4 absolute epoch 模型，source update `2204`。

t0/t1 特征均为 `17896 x 1024`，image index 完全一致。独立 Top5 关系构建结果：

- t0 合法关系：10229。
- 动态候选 PID：5167。
- Top5 PID 关系：259。
- Top5 anchor：215。
- `top5_fraction=0.05`。
- `query_gallery_used=false`。

由 t1 模型新建 optimizer、scheduler、GradScaler 和每 rank RNG，生成：

`/home/DIFFER/log_NDP/ndp_fresh_new_seed_prcc_20261027_180205/common_seed3456/`

B0、TOP5、NFRESH 三组均从该目录启动，未复制其他 seed 的参考图。

## 实际运行命令

Smoke 和正式训练使用：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed 3456 \
  --arm {B0|NDYN_TOP5|NFRESH} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --resume-checkpoint /home/DIFFER/log_NDP/ndp_fresh_new_seed_prcc_20261027_180205/common_seed3456 \
  --epochs 5 --eval-epochs 1,3,5 \
  --save-updates 551,1102,1653,2755 --diagnostic-steps 20
```

TOP5 和 NFRESH 额外使用 Seed3456 自己的 `t1_features.npz` 与 `top5_graph.json`。B0 不加载 NDP 关系图。

## Seed3456 PRCC C→A 结果

单位为百分比，格式为 `Rank-1 / mAP`：

| Epoch | B0 | TOP5 | NFRESH | NFRESH−B0 |
|---:|---:|---:|---:|---:|
| 1 | 63.20 / 57.15 | 63.36 / 57.18 | 62.63 / 57.16 | -0.565 / +0.007 pp |
| 3 | 63.90 / 59.22 | 64.52 / 59.67 | 63.53 / 59.58 | -0.368 / +0.354 pp |
| 5 | 63.42 / 59.98 | 64.49 / 59.50 | **64.15 / 60.60** | **+0.734 / +0.615 pp** |

NFRESH−TOP5：epoch1 `-0.734 / -0.023 pp`，epoch3 `-0.988 / -0.091 pp`，epoch5 `-0.339 / +1.092 pp`。

当前 PRCC evaluator 未暴露 mINP，因此没有编造 mINP。

## NFRESH 激活、Fresh Bank 与 AMP

| Epoch | active batch rate | selected relations | active relations | NDP mean loss |
|---:|---:|---:|---:|---:|
| 1 | 9.98% | 257 | 143 | 0.000546 |
| 3 | 9.75% | 516 | 265 | 0.000522 |
| 5（恢复段） | 8.58% | 516 | 220 | 0.000519 |

正式 NFRESH 日志中存在非零 NDP 特征梯度；诊断行最大值约 `0.01249`。B0 正式日志的 NDP Loss、关系数和 NDP 梯度均为零。

Fresh bank：

- 每个 bank 形状 `17896 x 1024`。
- 固定 Top5 关系数量 259，没有在训练中重新选图。
- 所有正式 bank manifest 的 `rng_state_preserved=true`。
- Seed3456 bank Margin 均值约从 `0.004797` 变化到 `0.009870`，变化 `+0.005073`。

NFRESH 主进程在第 3 节点评估完成后被系统终止；update1653 的 rank0/rank1 完整状态存在，因此从该节点在 `nfresh_seed3456_5raw_recovery_from03/` 恢复剩余两 raw epoch。恢复段完成 `1102/1102` effective updates；没有重跑前三轮，也没有合并不同起点的指标。

AMP skip 和 effective update 分开记录。B0、TOP5 存在少量正常 AMP skip；NFRESH 主段和恢复段均完成，未出现 NaN/Inf 或参数非有限。

## 与既有三个种子对照

下面沿用上一份报告中已完成的严格配对结果；4321 使用重建 common state 后的正式 B0：

| Seed | Epoch | B0 | TOP5 | NFRESH | NFRESH−B0 |
|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 62.38 / 56.66 | 63.20 / 57.00 | 63.03 / 56.76 | +0.649 / +0.097 pp |
| 1234 | 3 | 63.67 / 59.21 | 63.70 / 59.46 | 63.73 / 59.34 | +0.056 / +0.126 pp |
| 1234 | 5 | 62.86 / 59.51 | 63.90 / 60.29 | 64.07 / 60.22 | +1.214 / +0.715 pp |
| 4321 | 1 | 62.07 / 56.96 | 61.45 / 56.86 | 61.78 / 56.66 | -0.282 / -0.295 pp |
| 4321 | 3 | 63.22 / 58.90 | 62.57 / 59.10 | 63.65 / 58.74 | +0.423 / -0.168 pp |
| 4321 | 5 | 64.41 / 60.95 | 64.04 / 60.51 | 63.76 / 60.25 | -0.649 / -0.706 pp |
| 3456 | 1 | 63.20 / 57.15 | 63.36 / 57.18 | 62.63 / 57.16 | -0.565 / +0.007 pp |
| 3456 | 3 | 63.90 / 59.22 | 64.52 / 59.67 | 63.53 / 59.58 | -0.368 / +0.354 pp |
| 3456 | 5 | 63.42 / 59.98 | 64.49 / 59.50 | 64.15 / 60.60 | +0.734 / +0.615 pp |

在 epoch5，NFRESH 相对 B0 的三 seed 平均差值约为 Rank-1 `+0.433 pp`、mAP `+0.208 pp`；这是描述性均值，不是统计显著性结论。三 seed 中 2 个 seed 的两个指标同时提升，但 Seed4321 仍明显下降，不能称为稳定收益。

## 测试与最终决策

执行：

```bash
/usr/local/bin/python3.10 -m py_compile \
  scripts/ndp_v5_prcc_runner.py scripts/ndp_dynamic_top5_reference.py \
  scripts/ndp_make_common_warmstart.py
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
```

结果：`17 passed`，三份脚本编译通过；B0 loss contract、NFRESH RNG contract 和已有梯度/状态约束均通过。

最终判断：

1. **Seed3456 独立实验完成：是。** 三个正式 arm 均完成 5 raw epoch，NFRESH 通过断点恢复完成。
2. **NFRESH 是否在第3、第5 epoch都优于 B0：否。** 第5 epoch两个指标均改善，但第3 epoch Rank-1 下降。
3. **是否新增 Seed5678：暂不新增。** 按预注册门禁未满足“epoch3 与 epoch5 均优于 B0 且 mAP 改善”。
4. **是否值得继续 NFRESH 长训练：不建议直接延长。** 先把三个 seed 的 seed-dependent 结果与真实 Triplet 的增量监督对应起来。
5. **下一步：** 优先做 relation-level 增量监督分析，确认 NFRESH 实际保护的关系是否不在当前 Batch-Hard Triplet 的有效监督集合中；不要继续扫描 Margin 阈值或新增相似 NDP Loss。

**最终结论：Seed3456 给出一个有价值的 epoch5 正向复现，但没有消除 NFRESH 的不稳定性；NDP-Fresh 仍是“真实可训练、存在正向信号、尚未稳定验证”的候选。**
