# 最小反应式 NDP 候选决策

候选保持为单侧关系保护：

`L = mean_r w_r ReLU(M_ref(r) - M_current(r) - epsilon)`。

它具有三项明确性质：参考分支 detach；关系改善时零惩罚；只有当前边界相对可靠参考下降超过容忍量才产生梯度。它与参考 Margin 自适应 Hinge 在单一 `(anchor, positive, negative)` 关系上是同一类约束，不能仅凭名字或 mask 差异称为独立创新。

目前唯一可审计的增量来自关系集合：固定多 PID/双侧关系可能不在 Batch-Hard 的直接选择范围内。但这一覆盖结论依赖重建 sampler，且 B6/B7 关系数过少，未证明可靠性、双侧性或多 PID 本身带来超出 Triplet 的优化信息。因 A/A 仍 FAIL，本轮不接入训练。

决策：保留为“关系诊断与反应式原型”，不晋级正式 NDP loss；预测/干预机制仍 `INCONCLUSIVE`。
