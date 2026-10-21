# NDP 本轮最终研究决策

## 结论摘要

本轮完成了双侧时间对齐修复、新增错误 PID 可辨识性修复、稳健翻转敏感性分析、Hubness 去重、同轨迹 PPU 无梯度覆盖扩展和最小反应式原型单测。没有执行 NDP optimizer.step、30/60 epoch 或官方评估。

## 最终状态

```text
NDP_TIME_ALIGNMENT_FIX=PASS
NEW_NEGATIVE_PID_AUDIT=PASS_WITH_LIMITED_DECOMPOSABILITY
TRAJECTORY_COVERAGE_EXPANSION=PASS_PPU_EVAL_NO_GRAD_WITH_LIMITED_COVERAGE
REACTIVE_NDP_PROTOTYPE=UNIT_TEST_PASS_ONLY
NDP_MECHANISM=INCONCLUSIVE
A/A_REPRODUCIBILITY=FAIL
CONTROLLED_SHORT_PILOT=NO_GO
```

## 对用户问题的直接回答

1. 双侧时间错位已修复；35 行/seed 的同区间结果显示原先的 28 行确实漏掉区间，且不对称仍存在。
2. 新增错误 PID 已改为固定 PID 关系的正确→错误计数：可辨识事件为 seed1234=5、seed4321=2；其余聚合关系不可拆分。
3. 近零容差后，delta=1e-3 的稳健翻转为 6/1；delta=1e-2 为 4/0。seed4321 的 `-2.29e-5` 是 borderline。
4. 同轨迹节点均存在并已在 PPU eval/no_grad 重新编码；扩大固定覆盖为 159/137 anchor、30/31 PID，但仍不是全量，也没有多 PID 每关系相似度，因此不能完成真正多 PID 侵入验证。
5. 预测式 NDP 仍不可靠：上一段下降量对未来翻转 AUC 约 0.359/0.384；边界距离主要反映静态接近零，不证明预测更新造成的损害。
6. 反应式原型的数学和梯度接口合法，但单正/单负时与自适应 Margin Triplet 等价。独立价值目前只有“可靠历史关系/多 PID/双侧/单侧门控”等信息层面，尚无因果证据。
7. PPU A/A 首个可观测分叉仍在 block22 norm1/q_proj 反向链，底层 kernel 根因未定位。因此不允许短程训练。

## 保留与停止建议

保留：双侧有向关系审计、严格/稳健翻转区分、去重 Hubness、同轨迹 eval/no_grad 证据和反应式原型单测。

暂停：把 Hubness 直接作为权重、把聚合 advantage 拆成多 PID、预测式 NDP 门控，以及任何 optimizer-step 效果声明。

下一项必要工作是解决 PPU 反向 A/A，或预先建立可接受的数值噪声界；之后再用扩大后的固定多 PID关系进行无更新梯度对照。当前不申请正式短程训练。

## 版本与审计

远端 `main` 实际核对为 `6d88791f80c813fdb2426e5461d480c8e483a5a7`；`/data/DIFFER` 为 `573b050` 且 dirty，未用其未提交改动覆盖 `/home/DIFFER`。新增源码和单测快照位于：

`outputs/ndp_future_boundary_20261021_094250/source_snapshot_final/`

`outputs/ndp_future_boundary_20261021_094250/tests_snapshot_final/`

本轮关键离线测试为 8/8 PASS；模型计算仅使用 PPU，未使用 CPU 模型 fallback。
