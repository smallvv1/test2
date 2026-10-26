# NDP-Cap（NCAP）PRCC 实验结果

## 结论

NCAP 已完成 Seed1234、Seed4321 的 B0/NCAP 严格配对 5 raw continuation epoch 训练，并完成第3、第5节点的 PRCC C→A 评估。NCAP 确实降低了高历史 Margin 的目标上限，但没有在两个 seed 上同时带来稳定的 Rank-1/mAP 改善。

本轮结果支持停止继续扫描静态历史 Margin 截断阈值；如果继续 NDP，应转向基于动态关系变化的选择，而不是继续增加 N5/Floor0/RiskGate/Cap 类似目标函数。

## 实验信息

- 远端 main：`9047115af7d03002cb4467b63e7c57730aa5dd82`。
- 新日志：[ndp_cap_prcc_20261026_073551](/home/DIFFER/log_NDP/ndp_cap_prcc_20261026_073551)
- 数据：PRCC train-only，17,896 张训练图；官方 C→A 评估。
- `run-id`：`ndp_prcc_baseline_fix_20261025_130457`；arm 未进入 GPC hash phase。
- 两个 seed 的 `common_seed*_recreated` 完整包含 `rank0_state.pt`、`rank1_state.pt`、model、optimizer、scheduler、GradScaler 和 RNG 状态。
- 固定 GPC SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`，未修改。
- 最终 runner SHA256：`31a8d1f98f187d68677f3f95a71c7492a901c9596dccfa9cc766b390e986c027`。
- 单元测试：`9 passed`。
- 固定参数：`lambda_ndp=0.05`、`epsilon=0.02`、K=5、GPC 权重 0.1。

NCAP 实际实现为：

```python
target_margin = max(0.0, min(reference_margin - 0.02, 0.02))
hinge = F.relu(target_margin - current_margin)
```

NCAP 不使用 RiskGate 的 `0 < M_current < 0.02` 当前筛选；仅使用原有历史正确关系。B0 总损失严格为 `main_total`，不计算 NDP 关系损失。

## 20 步 Smoke Test

两个 NCAP Smoke 均完成 `20/20` 更新、无 NaN/Inf，并观察到非零 NDP loss、feature gradient 和参数探针梯度：

| Seed | 最后一步 NDP loss | NDP feature grad | NDP 参数探针 grad | NDP selected / active |
|---:|---:|---:|---:|---:|
| 1234 | 0.002717 | 0.002622 | 0.007072 | 18 / 2 |
| 4321 | 0.000071 | 0.001585 | 0.004156 | 19 / 1 |

## B0 / NCAP PRCC C→A 结果

数值为百分比；括号为同 seed、同节点 NCAP 相对 B0 的百分点差。

| Seed | continuation 节点 | B0 Rank-1 / mAP | NCAP Rank-1 / mAP |
|---:|---:|---:|---:|
| 1234 | 3 | 63.760 / 58.462 | 63.533 (-0.226) / 58.528 (+0.067) |
| 1234 | 5 | 63.308 / 59.105 | 63.562 (+0.254) / 59.256 (+0.151) |
| 4321 | 3 | 63.675 / 58.231 | 62.546 (-1.129) / 58.013 (-0.217) |
| 4321 | 5 | 63.760 / 59.724 | 63.280 (-0.480) / 59.323 (-0.402) |

仓库 PRCC evaluator 未暴露 mINP，本轮不使用其他协议结果替代。

## NCAP 激活、Loss 与局部指标

| Seed | 节点 | selected 关系 | active 关系 | active/selected | active Batch | NDP Loss | 当前 Margin | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 108,925 | 14,532 | 13.341% | 91.68% | 0.000981 | 0.01033 | 0.30653 |
| 1234 | 5 | 72,592 | 8,186 | 11.277% | 88.43% | 0.000852 | 0.01239 | 0.30660 |
| 4321 | 3 | 109,678 | 14,403 | 13.132% | 92.62% | 0.001037 | 0.01083 | 0.30516 |
| 4321 | 5 | 72,969 | 8,093 | 11.091% | 88.57% | 0.000892 | 0.01302 | 0.30522 |

正式训练前20步梯度统计：

| Seed | NDP feature grad mean / max | NDP 参数探针 mean / max |
|---:|---:|---:|
| 1234 | 0.001120 / 0.002622 | 0.003514 / 0.010807 |
| 4321 | 0.001317 / 0.005670 | 0.004344 / 0.025063 |

四组正式训练均为 2,755 raw batches、2,754 successful optimizer updates，各有1次 AMP skip；未补跑额外 batch。

## 与 V5、Floor0、RiskGate 的探索性比较

下表为各版本在其对应配对实验中相对 B0 的 Rank-1/mAP 差值，NCAP 为本轮差值；不同实验起点与数值轨迹不完全相同，不能当作同一条严格时间序列。

| 方法 | Seed1234 节点3 | Seed1234 节点5 | Seed4321 节点3 | Seed4321 节点5 |
|---|---:|---:|---:|---:|
| V5 | +0.113 / -0.472 | +0.056 / -0.419 | -0.282 / -0.035 | +0.000 / -0.150 |
| Floor0 | -0.085 / -0.361 | +0.480 / -0.022 | -2.004 / -0.693 | -0.226 / -0.490 |
| RiskGate | +0.056 / -0.229 | -0.423 / -0.655 | -1.722 / -1.054 | -0.564 / -0.385 |
| NCAP | -0.226 / +0.067 | +0.254 / +0.151 | -1.129 / -0.217 | -0.480 / -0.402 |

NCAP 的 active/selected 约 11.1%～13.3%，与上一轮 Floor0 约 11.1%～13.6% 接近，明显高于 V5 的约1.6%～1.9%。这说明 NCAP 的目标上限确实改变了高参考 Margin 的恢复目标，但对当前已经低于目标的关系，仍会产生大量 hinge；从实际激活量看，它没有实质性地把保护预算降到 V5 的弱约束水平。

## 训练与配对记录

1. 两个 recreated common state 的 rank state 均存在，四组均从对应 seed 的相同完整状态启动。
2. B0 两个 seed 的 NDP selected/active/loss 始终为0。
3. NCAP 两个 seed 均完成 Smoke 和正式训练，NDP 梯度真实到达当前特征和参数探针。
4. 参考图继续使用已有 PRCC train-only seed-specific K=5 图；未使用 query/gallery。
5. 没有修改 GPC、EVA02、CE、Triplet、Bio、sampler 或评估协议。

## 决策

NCAP 在 Seed1234 第5节点有小幅正向差值，但 Seed4321 第3/5节点均下降，不能通过双 seed 稳定提升门禁。它证明了截断历史目标可以工作并降低高 Margin 目标，但没有证明这种截断解决了性能不稳定问题。

下一步不建议继续修改静态历史 Margin 的 floor/cap/epsilon，也不建议直接扩展 LTCC 或60 epoch。若继续推进 NDP，应改为基于动态关系变化的筛选：只选择有可重复退化证据、且局部梯度方向与原 Triplet 不冲突的关系，并先进行小规模预注册验证。
