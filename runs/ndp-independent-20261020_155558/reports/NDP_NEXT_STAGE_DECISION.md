# NDP 下一阶段决策

## 结论

`NDP_PROTOTYPE_INCONCLUSIVE`

不允许进入 30 epoch 开发验证，也不启动 60 epoch 或官方 LTCC 测试。

## 逐项回答

1. **S1 是否停止？** 是，已封存为 `S1_FINAL_DECISION=NO_GO`。
2. **基础训练数值不确定性？** A/A 4/6 因 AMP `found_inf` 更新轨迹不一致；首个标量差异在第 2 步，算子级仍 `AMP_OPERATOR_ROOT_CAUSE_UNRESOLVED`。
3. **是否有局部异身份侵入？** 有 train-only 邻域漂移、最近异身份相似度上升和部分错误 PID 进入 Top-K；但局部 margin 两 seed 方向相反。
4. **哪些关系值得保护？** 仅考虑参考 `g_ref>0.05` 且当前出现超过 `epsilon=0.02` 的退化；不能保护参考中已错误的关系。
5. **参考模型可靠吗？** 足以做 train-only 诊断，不足以作为已验证的在线风险预测器。
6. **NDP 是否不同于 Triplet？** 关系选择和逐关系参考 margin 不同，但对相同激活三元组的梯度方向与 Triplet hinge 相同，必须做匹配 Triplet 控制。
7. **是否已证明减少退化？** 没有；未运行新的有效 PPU 更新，历史 4 步原型也为 inconclusive。
8. **是否损害跨衣一致性？** 本轮只做几何诊断，没有 NDP 更新，不能回答该因果问题；现有 GPC/Bio 几何的跨 seed 方向已是混合。
9. **是否有跨 seed 初步正向证据？** 没有。Promotion gate 的方向一致性为 false。
10. **是否允许 30 epoch？** 不允许。

当前保留固定 Bio+GPC-v1；NDP 仅保留为带明确控制组的后续原型研究。不得修改 GPC selector、lambda、matched-budget、Bio、Triplet、PRCC、NDP 历史探索或 checkpoint。

