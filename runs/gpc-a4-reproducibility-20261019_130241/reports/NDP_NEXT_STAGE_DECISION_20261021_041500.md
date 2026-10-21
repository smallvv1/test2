# NDP 下一阶段决策

## 最终状态

- `A/A_REPRODUCIBILITY = FAIL`
- `LRD_C_EQUIVALENCE = CONFIRMED`
- `NDP_DBP_MECHANISM = INCONCLUSIVE`
- `CONTROLLED_SHORT_PILOT = NO_GO`

## 决策理由

1. LRD-C 已证明等价于聚合正/负 margin 上的 Bernoulli KL 双向约束，不是独立的局部边界目标。
2. LRD-A 会漏检 Q+ 上升但边界翻转的关系；LRD-C 会在 Q+ 上升时反向约束，存在错误方向。
3. DBP 能在离线数据中捕获大部分真实边界翻转，但其 min-positive/max-negative hinge 与 Hard Triplet 数学重叠，独立性尚未成立。
4. A/A 仍在 block22 反向内部出现非确定性，无法支持 optimizer-step 因果比较。

因此本轮不启动 NDP optimizer-step、30 epoch 或 60 epoch。下一步只能是：在 A/A 数值噪声可接受或根因明确后，使用固定关系和同预算对照验证 DBP 的可靠性门控是否提供超出 Hard Triplet 的信息。
