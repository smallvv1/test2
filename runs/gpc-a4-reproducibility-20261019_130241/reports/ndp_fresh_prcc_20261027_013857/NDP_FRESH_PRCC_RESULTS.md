# NDP-Fresh PRCC 实验结果

日期：2026-10-27
实验目录：`/home/DIFFER/log_NDP/ndp_fresh_prcc_20261027_013857/`

## 结论摘要

NDP-Fresh 已完成 Seed 1234、4321 的 B0/NFRESH 配对 3 raw epoch 训练，并按同一起点补跑了冻结 Top5 严格消融。

- 两个 seed 在第 3 节点 NFRESH 都高于本轮匹配 B0：Seed1234 为 `+0.960 pp Rank-1 / +0.759 pp mAP`，Seed4321 为 `+0.198 pp / +0.209 pp`。
- 同一起点的冻结 Top5 在两个 seed 的第 3 节点都低于 B0，分别为 `-0.424/-0.070 pp` 和 `-0.847/-0.160 pp`。
- 因而本轮结果支持“epoch 级 fresh endpoint 可能比固定 t1 endpoint 更有用”的探索性信号，但不能排除 PPU/AMP 数值轨迹差异造成的部分影响，不能据此宣称 NDP 已被证明有效。
- NDP 确实参与 NFRESH 训练：两 seed 都有有限 NDP loss、非零 feature gradient 和模型更新；B0 的 NDP contribution 始终为 0。
- `mINP`：当前仓库 `R1_mAP_eval` 未暴露该指标，本报告不填造数。

当前建议：**GO 进行一次独立 seed 或重复配对的 5 raw epoch 验证；NO-GO 直接扩展到 60 epoch。** 在进一步扩展前，应先处理 AMP skip 对严格配对的影响，并保持 fresh bank、冻结 Top5 的消融同时存在。

## 1. 代码与实验记录

上游版本按用户指定记录为：`8d27468d1d08d57e29ad348023af14ed1fe234be`。`/home/DIFFER` 不是 Git checkout，因此本地无法从该目录验证 Git commit；代码身份使用以下 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `scripts/ndp_v5_prcc_runner.py` | `2deb440986ea0ff3cce0c4f845564fc5ee8d7fae9f2d5f786b85caac2be13237` |
| `tests/test_ndp_v5_b0_loss_contract.py` | `b1ea9a842c06b228f517f501a908842bcdac40f5657dafb7ff98772c69df8470` |
| 固定 GPC 选择上下文 | `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5` |

执行过的验证命令：

```bash
/usr/local/bin/python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
```

结果：`py_compile=PASS`；`16 passed`，只有既有依赖弃用警告。

正式训练命令均使用：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --epochs 3 --eval-epochs 1,3 --save-updates 551,1102,1653 \
  --diagnostic-steps 20
```

B0、NFRESH、冻结 Top5 使用同一 seed-specific common state、同一 `run-id`、同一 GPC phase 和同一 raw batch 计划。NFRESH 仅在每个 epoch 开始增加一次 train-only deterministic feature-bank 提取；不修改 sampler、GPC 或关系图。

## 2. 数据、参考图与初始化

- 数据：PRCC train split，`17896` 张 train-only 图像。
- 参考图：沿用 NDYN-TOP5 的 seed-specific 图和 t1 特征；不使用 query/gallery。
- NFRESH bank：每个 epoch 使用当前 NFRESH 模型、测试预处理、`shuffle=False`、冻结 bank；图像索引哈希三轮均为 `e51a558451176a7dea849cb13ef9633ad5c75923334f28fd1b5fcb7ab25c4097`。
- 原 Top5 图结构、风险排序、目标 margin 和 `lambda=0.05` 均固定；NFRESH 不重新选择关系。
- 由于历史 Top5 common rank state 已被清理，本轮从旧 t1 model-only 权重重建了新的 common state，并新建 optimizer/scheduler/GradScaler/RNG，起点明确标记为“model-only warm-start 的新训练片段”，不是旧完整 optimizer continuation。

共同初始模型 SHA-256：

| Seed | common model SHA-256 |
|---|---|
| 1234 | `2cb7e822fdef9133aa250ac81fe2ab21295ad8af98f2e4c5811f8f56af4da98f` |
| 4321 | `3155a876f0b5b067ccf4c3f79a67ff31a93e81f0210ff693dd844f632956861b` |

## 3. PRCC C→A 结果

数值为百分比；评估节点是 continuation raw epoch 1 和 3。`Top5-strict` 是本轮同一起点补跑的冻结 Top5 消融，不是此前历史运行的结果。

| Seed | 节点 | B0 Rank-1 | B0 mAP | NFRESH Rank-1 | NFRESH mAP | Top5-strict Rank-1 | Top5-strict mAP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 63.477 | 56.970 | 62.941 | 57.113 | 61.812 | 56.298 |
| 1234 | 3 | 63.731 | 59.083 | 64.691 | 59.841 | 63.308 | 59.013 |
| 4321 | 1 | 62.038 | 56.788 | 62.433 | 57.002 | 61.389 | 56.924 |
| 4321 | 3 | 63.816 | 59.052 | 64.014 | 59.261 | 62.969 | 58.892 |

第 3 节点相对 B0：

| Seed | NFRESH - B0 Rank-1 | NFRESH - B0 mAP | Top5-strict - B0 Rank-1 | Top5-strict - B0 mAP |
|---:|---:|---:|---:|---:|
| 1234 | +0.960 pp | +0.759 pp | -0.423 pp | -0.070 pp |
| 4321 | +0.198 pp | +0.209 pp | -0.847 pp | -0.160 pp |

第 1 节点 Seed1234 的 Rank-1 NFRESH 低于 B0 `0.536 pp`，但 mAP 高 `0.143 pp`；Seed4321 两项均高。故“两个 seed 第3节点均为正”不等于“所有早期节点均为正”。

## 4. NFRESH 参考库变化

固定关系数未因 fresh bank 更新而改变。变化仅来自当前模型的 endpoint 特征。

| Seed | 固定有效 PID 数 | Epoch bank | t1 margin 均值 | fresh margin 均值 | fresh - t1 |
|---:|---:|---:|---:|---:|---:|
| 1234 | 256 | 1 | 0.005298 | 0.005298 | +0.000000009 |
| 1234 | 256 | 2 | 0.005298 | 0.007480 | +0.002183 |
| 1234 | 256 | 3 | 0.005298 | 0.008696 | +0.003398 |
| 4321 | 192 | 1 | 0.006332 | 0.006332 | +0.000000009 |
| 4321 | 192 | 2 | 0.006332 | 0.006137 | -0.000195 |
| 4321 | 192 | 3 | 0.006332 | 0.007155 | +0.000824 |

这说明参考关系的 endpoint margin 在训练过程中确实变化，且两个 seed 的变化方向不同；因此“固定历史参考过时”是一个可观测现象，但仍不能单凭 margin 变化证明性能因果关系。

## 5. NDP 激活、梯度与 AMP

### NFRESH 训练侧统计

| Seed | 节点 | NDP active batch rate | selected PID | active hinge PID | NDP mean loss | 最大 feature-grad（诊断行） |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 10.07% | 253 | 138 | 0.000488 | 0.010881 |
| 1234 | 3 | 8.98% | 509 | 239 | 0.000486 | 0.010881 |
| 4321 | 1 | 9.98% | 188 | 124 | 0.000435 | 0.010917 |
| 4321 | 3 | 7.53% | 377 | 182 | 0.000369 | 0.010917 |

正式训练中非零 NDP loss 的 step 数为 Seed1234 `157/1653`、Seed4321 `132/1653`；对应最大 probe parameter gradient 分别约 `0.067283`、`0.042186`。因此 NFRESH 不是只计算不反传，当前 anchor 的梯度路径确实存在。

B0 四个评估节点的 `ndp=0`、active batch=0、NDP contribution=0，满足 B0 不计算/不加入 NDP 的契约。

AMP skip：

- B0 Seed1234：1 次，raw batch 588；1653 raw / 1652 effective updates。
- NFRESH Seed1234：1 次，raw batch 158；1653 raw / 1652 effective updates。
- B0 Seed4321：1 次，raw batch 341；1653 raw / 1652 effective updates。
- NFRESH Seed4321：0 次；1653 raw / 1653 effective updates。

因此四组使用了相同 raw batch 预算和 GPC 上下文，但 effective optimizer update 并非完全相同；这也是本轮只能作探索性结论的主要限制。

## 6. 与冻结 Top5 的解释

冻结 Top5 在同一起点的第 3 节点没有优于 B0，而 NFRESH 两个 seed 均优于 B0。这一结果与“NFRESH 的收益来自参考端点随 epoch 更新，而不是单纯把 Top5 约束加进训练”相一致。

但该判断仍受以下因素限制：

1. PPU/AMP 非确定性造成 skip 位置不同，Seed4321 的 NFRESH 比 B0 多一个 effective update。
2. 当前只有两个 seed 和 3 raw epoch，不能估计稳定性或长程收益。
3. common state 是从 t1 model-only 权重重建的短程片段，不是原始训练完整状态的无缝 continuation。
4. PRCC evaluator 当前不输出 mINP。

## 7. 决策

**本轮结论：探索性 GO，正式方法结论仍为 INCONCLUSIVE。**

值得继续的最小下一步是：保持当前 NFRESH、同一起点冻结 Top5、B0 三臂，增加一个独立 seed 或重复配对，并延长到 5 raw epoch；必须继续记录 raw/effective update、AMP skip 和 fresh bank margin。若收益在额外重复中消失，则停止这个方向。

不建议：

- 直接跑 60 epoch；
- 再扫描 epsilon、lambda、floor/cap 或 gate 阈值；
- 修改 GPC、EVA02、Triplet、Bio、sampler 或评估协议；
- 把本轮两个 seed 的小幅提升表述为已证实的 NDP 泛化收益。

主要日志、bank、checkpoint 和逐步 CSV 均保存在：

`/home/DIFFER/log_NDP/ndp_fresh_prcc_20261027_013857/`
