# NDP-V5-EPS005 PRCC 双种子实验结果

实验日期：2026-10-25
实验目录：`/home/DIFFER/log_NDP/ndp_v5_eps005_prcc_20261025_095504/`
远端 GitHub 提交：`99947739c7acd8709f5c714d08496de2da4817d4`

## 结论先行

本轮按预注册的单变量修改，将 NDP-V5 的 `epsilon` 从 0.02 改为 0.005；
lambda、损失、参考图、K=5 图、GPC phase、优化器和训练协议均保持不变。

- EPS005 确实显著提高了 NDP 激活：激活负 PID 关系比例从旧 N5 的约
  1.5%～1.9% 提高到约 15.6%～19.6%。
- Seed1234 在第 5 raw epoch 相对 B0：Rank-1 `+1.27 pp`、mAP `+0.55 pp`。
- Seed4321 在第 5 raw epoch 相对 B0：Rank-1 `+0.37 pp`、mAP `-0.56 pp`。
- 因此 EPS005 没有解决两个 seed 的稳定性问题；激活率增加没有转化为双 seed、双指标的一致收益。
- 不建议用 EPS005 直接进入 10/60 epoch，也不继续在本轮扫描更多 epsilon。

## 1. 固定项与最小代码修改

损失保持：

```text
L_base = CE + Triplet + clipBio + 0.1 * original GPC-v1
L_total = L_base + 0.05 * NDP-V5
NDP-V5 = per-negative-PID ReLU(M_ref - epsilon - M_current)
```

本轮唯一算法参数为：

```text
--ndp-epsilon 0.005
```

runner 的默认值仍为 0.02，旧实验不受影响。当前 NDP runner 增加了显式
命令行参数，并将实际 epsilon 写入 `training_config.json` 和 reference
metadata。GPC-v1 的源文件和选择算法没有修改。

两个 N5-EPS005 run 都从此前相同的 B0 update1653 完整状态开始：

```text
Seed1234: /home/DIFFER/log_NDP/ndp_v5_prcc_20261024_042844/b0_seed1234_3e/node_update1653
Seed4321: /home/DIFFER/log_NDP/ndp_v5_prcc_20261024_042844/b0_seed4321_3e/node_update1653
```

旧 B0 结果直接复用，没有重复训练。新 N5 输出使用独立目录，但 GPC
`run-id` 有意复用旧 5-epoch run-id：

```text
ndp_prcc_5epoch_20261024_125141
```

对应 phase 为：

```text
Seed1234: ndp_v5_prcc_gpc_pair_ndp_prcc_5epoch_20261024_125141_seed1234
Seed4321: ndp_v5_prcc_gpc_pair_ndp_prcc_5epoch_20261024_125141_seed4321
```

这样新 EPS005 与已保存 B0/N5 使用相同的 GPC phase；新实验的物理日志仍
保存在独立目录，没有覆盖旧结果。

## 2. 实际运行命令

代码检查：

```bash
python3 -m py_compile scripts/ndp_v5_prcc_runner.py
```

Seed1234 Smoke Test：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --seed 1234 --arm N5 \
  --run-id ndp_prcc_5epoch_20261024_125141 \
  --run-dir /home/DIFFER/log_NDP/ndp_v5_eps005_prcc_20261025_095504/n5_seed1234_smoke \
  --feature <seed1234-warm-feature> --graph <seed1234-warm-graph> \
  --resume-checkpoint <seed1234-B0-update1653> \
  --steps 20 --ndp-epsilon 0.005 --diagnostic-steps 20
```

正式训练使用相同命令模板，分别运行 Seed1234/4321：

```bash
--arm N5 --epochs 5 --eval-epochs 3,5 \
--save-updates 1653,2755 --diagnostic-steps 20 \
--ndp-epsilon 0.005
```

两组正式输出：

```text
/home/DIFFER/log_NDP/ndp_v5_eps005_prcc_20261025_095504/n5_eps005_seed1234_5raw/
/home/DIFFER/log_NDP/ndp_v5_eps005_prcc_20261025_095504/n5_eps005_seed4321_5raw/
```

## 3. Smoke Test

Seed1234 20-step Smoke Test 为 PASS：

| 指标 | 数值 |
|---|---:|
| NDP selected relations | 1357 |
| NDP active relations | 300 |
| active relation ratio | 22.1% |
| active Batch rate | 95.0% |
| NDP mean loss | 0.001084 |
| 最大 NDP feature gradient | 0.003208 |
| 最大 NDP probe parameter gradient | 0.05712 |
| AMP skip | 0 |

总 loss、NDP loss、模型参数和梯度均有限；当前 anchor 的梯度路径没有被
reference stop-gradient 错误切断。

## 4. PRCC C→A 结果

评估器支持 Rank-1 和 mAP，不输出 mINP，因此 mINP 没有估计或填充。

| Seed | Raw epoch | B0 Rank-1 | B0 mAP | 旧 N5(ε=.02) Rank-1 | 旧 N5 mAP | EPS005 Rank-1 | EPS005 mAP | EPS005-B0 Rank-1 | EPS005-B0 mAP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 64.69% | 59.26% | 64.10% | 58.70% | 63.73% | 58.44% | -0.96 pp | -0.83 pp |
| 1234 | 5 | 62.94% | 58.82% | 63.73% | 59.71% | 64.21% | 59.37% | +1.27 pp | +0.55 pp |
| 4321 | 3 | 62.38% | 58.31% | 62.40% | 58.04% | 61.92% | 57.63% | -0.45 pp | -0.68 pp |
| 4321 | 5 | 63.00% | 59.85% | 63.14% | 59.58% | 63.36% | 59.29% | +0.37 pp | -0.56 pp |

旧 N5 数值来自 `ndp_prcc_5epoch_20261024_125141`，B0 也直接复用该实验
的固定 raw epoch 3/5 节点。三者使用同一组 warm-start 来源和旧 B0 GPC
phase；EPS005 只改变 NDP epsilon。

## 5. NDP 激活与梯度统计

### EPS005 正式 run

| Seed | Raw epoch | Active Batch rate | Selected relations | Active relations | Active/selected | NDP mean loss | 平均当前 Margin | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 92.95% | 105176 | 19603 | 18.64% | 0.001121 | 0.010650 | 0.30620 |
| 1234 | 5 | 89.11% | 70129 | 10975 | 15.65% | 0.000967 | 0.012883 | 0.30730 |
| 4321 | 3 | 93.44% | 109419 | 21431 | 19.59% | 0.001224 | 0.010796 | 0.30512 |
| 4321 | 5 | 92.01% | 72815 | 11941 | 16.40% | 0.001049 | 0.012951 | 0.30516 |

正式训练中最大 NDP feature gradient：

```text
Seed1234: 0.003208
Seed4321: 0.002263
```

### 与 epsilon=0.02 的激活对照

| Seed | Raw epoch | 旧 N5 active Batch rate | EPS005 active Batch rate | 旧 N5 active relation | EPS005 active relation |
|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 25.44% | 92.95% | 1.66% | 18.64% |
| 1234 | 5 | 23.41% | 89.11% | 1.47% | 15.65% |
| 4321 | 3 | 28.86% | 93.44% | 1.86% | 19.59% |
| 4321 | 5 | 26.45% | 92.01% | 1.62% | 16.40% |

因此本轮清楚验证了：epsilon 从 0.02 降至 0.005 确实扩大了保护作用的
覆盖和梯度激活；但覆盖扩大本身没有带来稳定的 mAP 改善。

## 6. AMP、raw batches 与 checkpoint

两个 EPS005 run 都运行了 2755 个 raw batches：

| Seed | Raw epoch 3 updates | Raw epoch 5 updates | AMP skip |
|---:|---:|---:|---|
| 1234 | 1653 | 2755 | 无 |
| 4321 | 1652 | 2754 | raw batch 498，epoch 4，scale 1024→512 |

没有因为 AMP skip 补跑额外 raw epoch。所有正式 run 的 `run_summary.json`
为 PASS，AMP 事件保存在各自 `amp_events.csv`。

最终模型 SHA256：

| Seed | EPS005 update1653 model | EPS005 update2755 model |
|---:|---|---|
| 1234 | `9760fb2a843218b9dde62ae241da634fe416b6f4609deef620461470d4764e66` | `ce81eb39f77eb552b0f9f171e1b304368b44d827f43eadcd683b9510d36cd6ed` |
| 4321 | `edf22c91f1ab6695f18ace3e72bc4bb15224a1953c7bc177f57c67a7dcb27737` | `a920c9e07682b48a649adf7328edf15b7f12895100b70602f1eb09af22ef839e` |

## 7. 代码与固定 GPC 哈希

- 修改后的 runner：
  `16017523360494d495750a1203118173f8f891d38d7ec139d6566d1b891b0398`
- 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py`：
  `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`
- Seed1234 warm feature：
  `cfa90437f80f917bf4911501c4ebf7b1ddb2f147e087a9df51aac3a8db520955`
- Seed1234 K=5 graph：
  `475596ab2e47ffeb4905d11e3fbb059ff0882d7752c93441fcbff2e6b4a84a4c`
- Seed4321 warm feature：
  `fea248efa7499d05b94d4b3b0b05ea687ff2a1d0346552d70125e2b24f59b8b7`
- Seed4321 K=5 graph：
  `577fd28f9398941738fca37c96f38ddd0946266ef27b06634d07747335b4db98`

参考图、query/gallery 边界、GPC 源文件和 EVA02 均未被本轮训练修改。

## 8. 最终回答与下一步

1. **epsilon 降低后激活率是否增加？** 是。Batch 激活约从 23%～29% 提升到
   89%～93%，负 PID 激活比例约从 1.5%～1.9% 提升到 15.6%～19.6%。
2. **两个 seed 的 Rank-1/mAP 是否更稳定？** 否。第 5 epoch 两个 seed 的
   Rank-1 都提升，但 Seed4321 mAP 仍下降 0.56 pp。
3. **是否改善 Seed4321 的 mAP 退化？** 没有。EPS005 的 -0.56 pp 比旧
   N5 的 -0.27 pp 更差。
4. **是否值得进一步验证？** EPS005 不值得直接延长训练。若保留研究价值，
   下一步只能做一次预注册的独立 seed/严格重复，以区分 seed/AMP 波动与真实
   过度激活；不继续无目的 epsilon 扫描。

结论：EPS005 证明“低 epsilon 能提高局部保护覆盖”，但没有证明“更高覆盖
能带来稳定检索收益”。当前 NDP 仍不能进入长程训练或作为已验证改进方案。
