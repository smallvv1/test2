# NDP 第二创新点阶段性最终决策

运行：`20261021_082715`。

## 已验证事实

1. DIFFER 实际训练入口在 `/data/DIFFER/scripts/gpc_v2_train.py`，固定 `feat[0]`、CE、Batch-Hard Triplet、Bio 和 GPC-v1；本轮没有修改这些代码。
2. PPU A/A 的首个可观测反向差异位于 EVA02 block22 内部，精确底层 kernel 仍未确定，A/A 仍为 FAIL。
3. 在冻结 train-only 特征上，可以构建合法的 PID、衣物、相机对齐邻域图；但 Bio E60 与 GPC E60 是独立终点，不能作为时间稳定性证据。
4. 多 PID 软侵入指标已经实现并通过空集合、自引用、有限梯度、参考 detach 等单元测试，但两个 seed 的变化方向不一致。
5. P3 的所有数值均为独立终点 shift exploratory；没有自然单步风险预测验证。

## 尚未验证

- 稳定历史关系是否能预测同一训练轨迹的未来退化；
- R2–R5 是否比 Hard Triplet 提供稳定额外信息；
- 软侵入损失是否改善训练或检索性能；
- 双侧关系是否具有跨 seed 的增量价值。

## 门禁

- `P0_AA_NUMERICAL_AUDIT=FAIL`
- `P1_TEMPORAL_REFERENCE=INCONCLUSIVE`
- `P2_SOFT_INTRUSION=INCONCLUSIVE`
- `P3_ONLINE_RISK_PREDICTION=NOT_IDENTIFIABLE`
- `NDP_LOSS_UNIT_TEST=PASS`
- `NDP_OPTIMIZER_EFFECT=NOT_RUN`
- `FORMAL_30_60_EPOCH=NO_GO`

最终状态：`INCONCLUSIVE`。

当前最合理的下一步不是接入新损失，而是获得同一 Bio+GPC-v1 continuation 的完整 train-only 特征节点，预注册未来节点风险标签后再评估 R0–R5。A/A 未通过前，不允许用 optimizer-step 结果判断 NDP 增益。
