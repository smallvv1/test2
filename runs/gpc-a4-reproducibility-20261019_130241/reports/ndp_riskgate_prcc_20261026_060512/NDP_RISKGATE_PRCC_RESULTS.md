# NDP-RiskGate（NRG）PRCC 实验结果

## 结论

NRG 已完成 Seed1234、Seed4321 各 5 raw continuation epoch，并在第3、第5 continuation 节点完成 PRCC C→A 评估。NRG 确实筛选并激活了局部风险关系，但两个 seed 均没有在 Rank-1 和 mAP 上稳定优于上一轮 B0/N5/NF0 严格配对结果。

因此本轮不建议继续扫描阈值、lambda 或直接延长到60 epoch；RiskGate 证明了“只保护当前仍正确且接近边界的关系”可以降低保护覆盖，但尚未提供性能增量证据。

## 实验信息

- 远端 main：`a3c7d94659dc5c6d8acaf68498c91ccba16dd095`。
- 新日志：[ndp_riskgate_prcc_20261026_060512](/home/DIFFER/log_NDP/ndp_riskgate_prcc_20261026_060512)
- 训练数据：PRCC train-only，17,896 张图；评估为官方 C→A。
- `run-id`：`ndp_prcc_baseline_fix_20261025_130457`，未将 arm 放入 GPC phase。
- 固定 GPC SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`，未修改。
- 最终 runner SHA256：`f57ce0ed479d0d169ac9acaeaaa89d0e8ce58110d3d78109cc44cf6b3f5b43bd`。
- 最终测试文件 SHA256：`4b554896b1eac42e64168e6976b6b0c8625456006c8503ae7f9dfaa94733761c`。
- 单元测试：`7 passed`。
- 固定参数：`lambda_ndp=0.05`、`epsilon=0.02`、K=5、GPC 权重 0.1。

NRG 的逐 PID 筛选为：

```text
M_ref >= 0.02 且 0 < M_current < 0.02
M_target = max(M_ref - 0.02, 0.01)
L_NRG = ReLU(M_target - M_current)
```

筛选使用 detached 当前 Margin；每个 anchor 内对通过筛选的关系平均，再对有效 anchor 平均。B0 分支仍不读取或计算 NDP 关系。

## 起点限制

上轮严格实验目录中的 `common_seed1234/4321` 目前只保留 `model.pth` 和 manifest，完整 `rank*_state.pt` 已不存在；在 `/home/DIFFER` 与 `/data/DIFFER` 也未找到可恢复的对应 common rank state。因此本轮没有把 model-only 权重伪装成完整续训，而是从 manifest 对应的同一 B0 三 raw epoch 模型权重，重新创建了 seed-specific 的共享 optimizer/scheduler/GradScaler/RNG 状态：

- `common_seed1234_recreated/`
- `common_seed4321_recreated/`

六组 NRG 训练均使用各自 recreated common state，源模型权重与上轮一致；这是“共同模型权重 warm-start 的新训练片段”，不是上轮 optimizer state 的无缝续训。因此下表与旧 B0/N5/NF0 的比较保留该起点限制。

## Rank-1 / mAP 对照

数值为百分比；括号为相对同 seed、同节点 B0 的百分点差。B0/N5/NF0 数值直接复用上一轮严格配对日志，NRG 来自本轮正式日志。

| Seed | 节点 | B0 | N5 | NF0 | NRG |
|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 63.251 / 58.548 | 63.364 / 58.076 (+0.113 / -0.472) | 63.167 / 58.187 (-0.085 / -0.361) | 63.308 / 58.319 (+0.056 / -0.229) |
| 1234 | 5 | 63.195 / 59.581 | 63.251 / 59.162 (+0.056 / -0.419) | 63.675 / 59.558 (+0.480 / -0.022) | 62.772 / 58.926 (-0.423 / -0.655) |
| 4321 | 3 | 63.901 / 58.436 | 63.618 / 58.401 (-0.282 / -0.035) | 61.897 / 57.744 (-2.004 / -0.693) | 62.179 / 57.382 (-1.722 / -1.054) |
| 4321 | 5 | 64.409 / 60.155 | 64.409 / 60.005 (+0.000 / -0.150) | 64.183 / 59.665 (-0.226 / -0.490) | 63.844 / 59.770 (-0.564 / -0.385) |

仓库 PRCC evaluator 未暴露 mINP，本轮未用其他协议替代。

## NRG 激活与局部指标

| Seed | 节点 | 风险筛选关系 | 实际激活关系 | 激活/筛选 | 激活 Batch | NRG Loss | 当前 Margin | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 3,744 | 771 | 20.59% | 14.22% | 0.000289 | 0.01426 | 0.30655 |
| 1234 | 5 | 2,088 | 418 | 20.02% | 12.98% | 0.000291 | 0.01428 | 0.30665 |
| 4321 | 3 | 4,163 | 787 | 18.90% | 15.85% | 0.000306 | 0.01435 | 0.30514 |
| 4321 | 5 | 2,326 | 475 | 20.42% | 14.43% | 0.000314 | 0.01412 | 0.30520 |

两 seed 正式运行均为 2,755 raw batches、2,754 successful optimizer updates，各有1次 AMP skip，具体位置保存在对应 `amp_events.csv`。NRG 的 first-20-step `ndp_feature_grad_norm`：

| Seed | mean | max |
|---:|---:|---:|
| 1234 | 0.0002687 | 0.005373 |
| 4321 | 0.0007602 | 0.005249 |

这证明正式 NRG 中当前特征路径有非零梯度。最初正式训练的参数探针条件漏列 NRG，故 `ndp_probe_param_grad_norm` 未记录；随后已修复诊断条件并做了新的20步 Smoke。该 Smoke 两 seed 均 20/20、无 NaN，但当次20步未命中激活 hinge，所以参数探针为0；不能把这次 Smoke 的0写成正式训练的隔离参数梯度证据。

## 执行记录

1. NRG 初次 Smoke 因传入了不存在的参考图路径失败，未产生更新；失败目录保留。
2. 修正为上轮真实 seed-specific 参考图路径后，两个 seed 的 NRG Smoke 均 PASS。
3. 因上轮 common rank state 已清理，重新生成 recreated common state。
4. Seed1234、4321 正式 5 raw epoch 均 PASS，均完成第3/5评估。
5. 诊断条件修复后追加的参数探针 Smoke 不覆盖正式结果。

正式训练命令形式：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_v5_prcc_runner.py --mode train --arm NRG \
  --run-id ndp_prcc_baseline_fix_20261025_130457 \
  --resume-checkpoint common_seed{1234,4321}_recreated \
  --epochs 5 --eval-epochs 3,5 --ndp-epsilon 0.02
```

## 决策

NRG 相比 NF0 确实减少了保护覆盖，相比 N5 更集中地选择当前近边界且历史可靠关系；但这没有转化为稳定的 PRCC 性能提升。Seed4321 第3节点 Rank-1 为 `-1.722 pp`，Seed1234 第5节点 Rank-1/mAP 为 `-0.423/-0.655 pp`。

当前建议：停止继续增强或扫描 NDP 门控强度，不做60 epoch；若继续研究，应分析被 NRG 选中的关系是否真的会在未来翻转，以及 NRG 梯度是否与原 Triplet 方向一致，再决定是否进行新的、预注册的关系级实验。NRG 本轮结论为：实现和训练路径有效，但性能门禁未通过。
