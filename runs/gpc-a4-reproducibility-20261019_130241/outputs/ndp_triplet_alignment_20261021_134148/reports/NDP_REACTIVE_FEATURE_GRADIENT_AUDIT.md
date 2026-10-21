# NDP 反应式特征梯度审计

## 口径

参考为 update0，当前为 update200；可靠性/脆弱性关系选择只使用 update200 以前信息，update400 只作为事后翻转标签。所有梯度是冻结身份特征上的一阶 autograd，`network_parameter_gradient=false`，没有 optimizer.step。

| seed | mask | selected | active（mask内） | loss | feature grad norm | 200→400 flips |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | G2_reference_margin | 3 | 3 | 0.00983254 | 0.852667 | 0 |
| 1234 | G3_random_count_matched_G5 | 8 | 6 | 0.203398 | 0.445212 | 1 |
| 1234 | G5_reliable_vulnerable | 8 | 1 | 0.00208862 | 0.192916 | 0 |
| 1234 | G6_reciprocal_image_relation | 50 | 4 | 0.00313913 | 0.059587 | 2 |
| 1234 | G7_reliable_vulnerable_reciprocal | 0 | 0 | 0 | 0 | 0 |
| 4321 | G2_reference_margin | 8 | 8 | 0.00302546 | 0.588 | 0 |
| 4321 | G3_random_count_matched_G5 | 23 | 21 | 0.182142 | 0.275718 | 0 |
| 4321 | G5_reliable_vulnerable | 23 | 0 | 0 | 0 | 2 |
| 4321 | G6_reciprocal_image_relation | 32 | 2 | 0.000278468 | 0.0839364 | 2 |
| 4321 | G7_reliable_vulnerable_reciprocal | 5 | 0 | 0 | 0 | 1 |

修复后 `active_relations` 只统计被当前 mask 选中的正损失关系，`all_candidate_active_relations` 另行保留；因此不会把未选中的 raw active 关系误报成当前方法激活。`G7` 在 seed1234 为空，在 seed4321 只有 5 条且当前不激活；这说明严格双侧+可靠脆弱交集覆盖不足。

结果只能证明候选损失在 update200 特征空间中可计算、梯度有限、改善关系不惩罚；不能证明网络参数更新后的真实收益。特别是 G5 的 seed4321 选中 23 条但 loss/梯度为 0，说明“关系选择上看似危险”不等于当前参考单侧保护已经激活。

## 机制判断

固定关系上的 `ReLU(M_ref-M_cur-epsilon)` 仍与参考 Margin Hinge/自适应 Triplet 有直接数学重叠。当前新增信息主要来自关系选择（历史可靠性、双侧、跨 PID），而不是损失形式本身；在真实 miner 覆盖、A/A 和参数更新均未通过前，不能称为独立创新。

## 可审计命令

`/usr/local/bin/python3.10 -u scripts/ndp_reactive_feature_gradient_audit.py --seed 1234 --features outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed1234/trajectory_features_seed1234.npz --groups outputs/ndp_future_boundary_20261021_094250/coverage_expansion/seed1234_retry1/fixed_diagnostic_groups.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/reactive_feature_gradient/seed1234`
`/usr/local/bin/python3.10 -u scripts/ndp_reactive_feature_gradient_audit.py --seed 4321 --features outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed4321/trajectory_features_seed4321.npz --groups outputs/ndp_future_boundary_20261021_094250/coverage_expansion/seed4321/fixed_diagnostic_groups.json --output-dir outputs/ndp_triplet_alignment_20261021_134148/reactive_feature_gradient/seed4321`
