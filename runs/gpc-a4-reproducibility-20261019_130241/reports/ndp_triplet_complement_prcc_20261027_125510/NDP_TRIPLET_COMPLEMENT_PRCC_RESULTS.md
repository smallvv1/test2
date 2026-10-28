# NDP-Triplet Complement PRCC 实验结果

日期：2026-10-28
实验目录：`/home/DIFFER/log_NDP/ndp_triplet_complement_prcc_20261027_125510/`

## 结论摘要

NCOMP 已完成 Seed 4321、3456 的 5 个 raw continuation epoch。它使用当前 Batch-Hard Triplet 的 raw `feat[0]`、欧氏距离和原始 `hard_example_mining` 得到 hard-negative PID；只有当 NDP 负 PID 不等于该 PID 时才保留 NFRESH 约束。

结果没有显示稳定收益：

- Seed 4321：NCOMP 在第 3、5 epoch 的 Rank-1 高于 B0（分别 `+1.242 pp`、`+0.480 pp`），但第 5 epoch mAP 为 `-0.001 pp`，不能视为完整指标改善。
- Seed 3456：NCOMP 在第 1、3、5 epoch 的 Rank-1/mAP 均低于 B0，第 5 epoch Rank-1 为 `-1.693 pp`。
- 相对于 NFRESH，NCOMP 没有把 Seed 3456 的退化修复；Seed 4321 第 5 epoch仅 Rank-1 略高、mAP 略低。

因此，本实验不支持“排除 Triplet hard-negative PID 后即可得到稳定增益”的结论。NCOMP 不值得直接扩展到 10/60 epoch，也不建议继续扫描该筛选规则的阈值。

## 1. 实现与固定项

NCOMP 的改动仅位于独立 NDP 路径：

1. `batch_hard_negative_pid()` 对 detached raw `feat[0]` 计算欧氏距离和原始 Batch-Hard 索引。
2. 将 hard-negative 索引映射为当前 anchor 的 PID。
3. 对每条历史/Top5 合法 NDP 关系，若其负 PID等于当前 hard-negative PID，则跳过该关系；否则使用 NFRESH 的 epoch-level fresh-bank Margin 约束。
4. hard-negative 筛选只使用 detached 张量，不改变原始 Triplet Loss，也不增加额外 Triplet 梯度。
5. B0 直接令 `total = main_total`，不计算 NDP 关系损失。

固定配置：

- 基础损失：`CE + Triplet + clipBio + 0.1 * GPC-v1`。
- NFRESH：epoch 级 train-only fresh bank，Top5 动态关系图，`lambda_ndp=0.05`。
- NCOMP：NFRESH + 当前 Triplet hard-negative PID 排除。
- PRCC C→A 官方换装评估；未使用 query/gallery 构建关系或选择参数。
- `run-id=ndp_prcc_baseline_fix_20261025_130457`。
- GPC phase 与 arm 无关，按 seed 固定为：
  `ndp_v5_prcc_gpc_pair_ndp_prcc_baseline_fix_20261025_130457_seed{seed}`。

## 2. 真实运行命令与校验

Smoke/单元测试：

```bash
/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_v5_b0_loss_contract.py
/usr/local/bin/python3.10 -m py_compile scripts/ndp_v5_prcc_runner.py
```

结果：`19 passed`，`py_compile` 通过。

正式训练使用以下命令结构（每个 seed/arm 各一组，双卡 DDP）：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train \
  --seed {4321|3456} --arm {B0|NFRESH|NCOMP} \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --run-dir /home/DIFFER/log_NDP/ndp_triplet_complement_prcc_20261027_125510/{arm}_seed{seed}_5raw \
  --resume-checkpoint {common_state} --epochs 5 --eval-epochs 1,3,5 \
  --save-updates 551,1653,2755 --diagnostic-steps 20
```

NCOMP Smoke Test 中，NDP Loss 有限、NDP feature gradient 非零、20/20 步执行成功；B0 的 NDP 统计与 NDP 梯度为零。构造性单元测试覆盖了“命中 hard-negative PID 时排除、命中其他 PID 时保留”。

## 3. PRCC C→A 结果

Rank-1 和 mAP 均为百分数；括号内为相对同 seed、同 epoch B0 的百分点变化。

| Seed | Arm | Epoch | Rank-1 | Δ Rank-1 | mAP | Δ mAP |
|---:|---|---:|---:|---:|---:|---:|
| 4321 | B0 | 1 | 61.614 | — | 56.742 | — |
| 4321 | NFRESH | 1 | 62.235 | +0.621 | 56.872 | +0.130 |
| 4321 | NCOMP | 1 | 62.320 | +0.706 | 56.834 | +0.092 |
| 4321 | B0 | 3 | 62.941 | — | 58.541 | — |
| 4321 | NFRESH | 3 | 62.743 | -0.198 | 58.902 | +0.361 |
| 4321 | NCOMP | 3 | 64.183 | +1.242 | 58.700 | +0.159 |
| 4321 | B0 | 5 | 64.042 | — | 60.519 | — |
| 4321 | NFRESH | 5 | 64.324 | +0.282 | 60.814 | +0.295 |
| 4321 | NCOMP | 5 | 64.522 | +0.480 | 60.518 | -0.001 |
| 3456 | B0 | 1 | 63.167 | — | 57.404 | — |
| 3456 | NFRESH | 1 | 62.828 | -0.339 | 56.979 | -0.425 |
| 3456 | NCOMP | 1 | 62.546 | -0.621 | 56.486 | -0.918 |
| 3456 | B0 | 3 | 64.098 | — | 59.344 | — |
| 3456 | NFRESH | 3 | 63.421 | -0.677 | 58.904 | -0.441 |
| 3456 | NCOMP | 3 | 63.251 | -0.847 | 58.702 | -0.642 |
| 3456 | B0 | 5 | 64.691 | — | 60.056 | — |
| 3456 | NFRESH | 5 | 63.957 | -0.734 | 60.057 | +0.002 |
| 3456 | NCOMP | 5 | 62.997 | -1.693 | 59.673 | -0.383 |

本 runner 的 PRCC C→A `R1_mAP_eval` 未暴露 mINP，因此本表不填 mINP；general 协议的内部字段不能与 C→A 指标混用。

## 4. NDP 激活、筛选与局部指标

### Seed 4321

| Arm | Epoch | Active Batch | Selected relation | Active relation | NDP loss | 平均当前 Margin | Purity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NFRESH | 1 | 9.89% | 188 | 125 | 0.000430 | 0.00729 | 0.30514 |
| NFRESH | 3 | 7.80% | 377 | 193 | 0.000384 | 0.00866 | 0.30400 |
| NFRESH | 5 | 6.62% | 190 | 87 | 0.000362 | 0.01001 | 0.30509 |
| NCOMP | 1 | 9.62% | 185 | 121 | 0.000428 | 0.00720 | 0.30514 |
| NCOMP | 3 | 7.58% | 373 | 184 | 0.000385 | 0.00869 | 0.30404 |
| NCOMP | 5 | 6.44% | 187 | 83 | 0.000387 | 0.00968 | 0.30510 |

NCOMP 排除关系：第 1/3/5 epoch 分别为 `3/188=1.60%`、`4/377=1.06%`、`3/190=1.58%`。

### Seed 3456

| Arm | Epoch | Active Batch | Selected relation | Active relation | NDP loss | 平均当前 Margin | Purity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NFRESH | 1 | 10.62% | 257 | 149 | 0.000521 | 0.00686 | 0.30676 |
| NFRESH | 3 | 9.12% | 516 | 242 | 0.000472 | 0.00778 | 0.30620 |
| NFRESH | 5 | 7.62% | 257 | 102 | 0.000579 | 0.00887 | 0.30848 |
| NCOMP | 1 | 9.98% | 252 | 143 | 0.000519 | 0.00683 | 0.30676 |
| NCOMP | 3 | 9.53% | 508 | 258 | 0.000539 | 0.00740 | 0.30618 |
| NCOMP | 5 | 7.99% | 253 | 107 | 0.000630 | 0.00840 | 0.30845 |

NCOMP 排除关系：第 1/3/5 epoch 分别为 `5/257=1.95%`、`8/516=1.55%`、`4/257=1.56%`。

排除比例较低，说明本实验改变的实际 NDP 预算有限；它没有形成足够大的监督结构差异来稳定改变结果。

## 5. 梯度与数值检查

前 20 个诊断 batch 的结果：

| Seed | Arm | 非零 NDP feature-grad batch | 最大 NDP feature-grad norm | 最大 NDP probe-param-grad norm | AMP found_inf |
|---:|---|---:|---:|---:|---:|
| 4321 | NFRESH | 3/20 | 0.010917 | 0.026951 | 0 |
| 4321 | NCOMP | 3/20 | 0.010917 | 0.038708 | 0 |
| 3456 | NFRESH | 1/20 | 0.010544 | 0.014261 | 0 |
| 3456 | NCOMP | 1/20 | 0.010544 | 0.014261 | 0 |

NDP 梯度确实回到当前模型参数；这不是“实现未激活”。同时，NDP 梯度在前 20 batch 中只出现在少数 batch，符合正式统计中的低 active rate。

AMP/有效更新：

- B0 Seed 4321：总 raw batch 2755，有效更新 2754；第 5 epoch 有一次 AMP skip。
- B0 Seed 3456：总 raw batch 2755，有效更新 2754；第 5 epoch 有一次 AMP skip。
- NFRESH Seed 4321、3456：总 raw batch/有效更新均为 2755/2755。
- NCOMP Seed 4321：2755/2755。
- NCOMP Seed 3456：2755/2754；第 1 个 segment 有一次 AMP skip。

所有正式运行均未出现 NaN/Inf；部分长任务遭遇系统 `SIGKILL(-9)` 后，从保存的第 3 epoch 完整节点恢复完成第 4、5 epoch。这是资源/进程中断，不是数值失败；恢复目录和原始目录均保留。

## 6. 配对与可复现性限制

两种子均使用同一个 t1 模型权重、同一 PRCC train-only Top5 图、相同学习率/采样协议和相同 arm-independent GPC phase。t1 模型 SHA256：

- Seed 4321：`3155a876f0b5b067ccf4c3f79a67ff31a93e81f0210ff693dd844f632956861b`
- Seed 3456：`44bed39e6659602354f40042840334cda19619d2628a4daaadd7ff6e2fa346d3`

但服务器在长任务期间清理了部分 common rank state：

- Seed 4321 的 B0/NFRESH 使用 `common_seed4321`；NCOMP 使用从同一 t1 模型重建的 `common_seed4321_rebuilt_for_ncomp`。
- Seed 3456 的 B0 使用 `common_seed3456`；NFRESH/NCOMP 使用从同一 t1 模型重建的 `common_seed3456_rebuilt_for_ndp`。

因此，模型起点相同且训练协议一致，但不能把本批结果表述为“所有 arm 具有可证明 bitwise-identical 的 optimizer/RNG 状态”。这降低了严格配对结论的强度，尤其是 Seed 3456 的 NCOMP 退化不能完全排除初始化状态差异的影响。后续若要做最终确认，应在同一作业中先生成一个只读 common rank state，再复制给三个 arm，避免任何中途状态清理。

## 7. 代码、数据与 checkpoint 指纹

- 上游参考提交：`f8d6ae4666f3f38f29f14d649bf40c020bca1615`。
- 当前本地 runner：`scripts/ndp_v5_prcc_runner.py`，SHA256 `ce2c39c18c3a95b1ed4d5dfded58d4e9fcf1ce03077f0790f2578b91139ce8d8`。
- 测试文件：`tests/test_ndp_v5_b0_loss_contract.py`，SHA256 `c4f3d6005fb39ac6c273e93982f1a4a912778dc9e18d0407788dcbc401682d40`。
- 固定 GPC 源：`/data/DIFFER/scripts/gpc_v2_train.py`，SHA256 `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`，本轮未修改。
- Seed 4321 t1 feature：`ea800bf4585243b54d218bd549a315d4402a4ef5f61f839ba74cebe175807517`。
- Seed 4321 Top5 graph：`6253226f373cfeac40346e4a3384d6656502c8b66b9fddcd50315af961cb673c`。
- Seed 3456 t1 feature：`e5a791b7aa7e2f3ab0b61d8f4468838044ca633b260e7ca8e6a24eabc635bff9`。
- Seed 3456 Top5 graph：`f8ca6fc91e04dddf736888c644e785e29ee6734cb12f309aee86d97721640455`。

最终 checkpoint SHA256：

| Seed | Arm | 终点模型 |
|---:|---|---|
| 4321 | B0 | `81ee732d13461be522e7131263555cd42255e391ee6506ab6a602fbb0630d669` |
| 4321 | NFRESH | `25a8b59f3ef3ed6531055c65aa0a281ed28caa0af29ce26cf42b589039027e43` |
| 4321 | NCOMP | `f567776ecced0b7f3fc72e1f2d9ad080bb6da220381c326d2934a293a80ab6ea` |
| 3456 | B0 | `0936c2b979cb3e6bf927cdbd8f7a3263532554477a6e811ce2d4ed79dc87de1b` |
| 3456 | NFRESH | `233d2e25cdd786e9f43fa3a82e11f03d9171ad04a0b35e87caf60d8cf704afab` |
| 3456 | NCOMP | `a670a394afdb34f06ca192e2136b71b3157c990a9fdd4224612834df3fae0750` |

## 8. 决策

### 是否验证了 Triplet PID 互补筛选？

验证了实现和训练链路：NCOMP 能计算当前 hard-negative PID、排除命中关系、保留其他 NDP 关系，并产生非零梯度。但性能结果不支持稳定互补收益。

### Seed 4321 的局部改善是否足够？

不够。NCOMP 第 3/5 epoch 的 Rank-1 改善没有同步转化为 mAP 改善，Seed 3456 还出现明显退化；不能据此宣称 NCOMP 有效。

### 是否继续 NCOMP、NFRESH 或增加训练轮数？

NO-GO：不建议直接扩展 NCOMP 到 10/60 epoch，也不建议继续堆叠相似的 PID 排除规则。NFRESH 在当前三臂实验中同样表现为 seed-dependent，不能作为稳定基线增量。

### 下一步建议

停止继续修改这一类 Margin 保护/Triplet PID 排除损失。若继续研究，应先做一项严格配对、只读的关系覆盖审计：量化 NDP 关系是否真正不在 Batch-Hard Triplet 的监督集合之外，再决定是否存在值得加入训练的新增监督；否则优先停止 NDP 训练分支，保留现有结果作为负面证据。

**本实验结论：NCOMP 实现 PASS；PRCC 稳定性能提升 NO-GO；不支持继续增加训练预算。**
