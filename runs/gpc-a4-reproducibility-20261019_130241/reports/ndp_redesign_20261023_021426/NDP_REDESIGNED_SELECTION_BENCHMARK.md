# P3 修复后预算匹配关系选择

候选池固定为每 seed 512 anchors × 5 negative PID = 2560 条。native 表保留方法本身的自然选中量；budget_matched 表以各 seed 非空 native 方法中的最小关系数为共同预算，并且 B0/B1/B2--B7 均不看 update400。B3--B7 的预算版是‘门控优先、当前 Margin 作 tie-break’的可比版本；原生 mask 仍单独报告，避免把补齐后的版本误称为纯交集。随机控制使用200个固定 RNG。

| seed | 版本 | 方法 | selected | active | flip | 覆盖全部flip | 当前错误 | anchor PID | negative PID | neg Gini |
|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | native | B0_random | 16 | 11 | 0 | 0.0 | 7 | 15 | 14 | 0.1071 |
| 1234 | native | B1_current_hard | 1760 | 1760 | 28 | 1.0 | 1368 | 73 | 75 | 0.3954 |
| 1234 | native | B2_reliability | 775 | 16 | 0 | 0.0 | 0 | 58 | 73 | 0.4015 |
| 1234 | native | B3_reliability_vulnerability | 637 | 16 | 0 | 0.0 | 0 | 58 | 70 | 0.4085 |
| 1234 | native | B4_reliability_degradation | 23 | 10 | 0 | 0.0 | 0 | 5 | 14 | 0.2640 |
| 1234 | native | B5_safe_boundary_activation | 16 | 16 | 0 | 0.0 | 0 | 10 | 11 | 0.2159 |
| 1234 | native | B6_reliability_multi_pid | 442 | 16 | 0 | 0.0 | 0 | 53 | 67 | 0.4261 |
| 1234 | native | B7_reliability_bidirectional | 438 | 6 | 0 | 0.0 | 0 | 56 | 67 | 0.4041 |
| 1234 | budget_matched | B0_random | 16 | 11 | 0 | 0.0 | 10 | 14 | 13 | 0.1442 |
| 1234 | budget_matched | B1_current_hard | 16 | 16 | 0 | 0.0 | 16 | 2 | 14 | 0.1071 |
| 1234 | budget_matched | B2_reliability | 16 | 12 | 0 | 0.0 | 0 | 13 | 12 | 0.1667 |
| 1234 | budget_matched | B3_reliability_vulnerability | 16 | 12 | 0 | 0.0 | 0 | 13 | 12 | 0.1667 |
| 1234 | budget_matched | B4_reliability_degradation | 16 | 8 | 0 | 0.0 | 0 | 3 | 11 | 0.2159 |
| 1234 | budget_matched | B5_safe_boundary_activation | 16 | 16 | 0 | 0.0 | 0 | 10 | 11 | 0.2159 |
| 1234 | budget_matched | B6_reliability_multi_pid | 16 | 12 | 0 | 0.0 | 0 | 13 | 12 | 0.1667 |
| 1234 | budget_matched | B7_reliability_bidirectional | 16 | 6 | 0 | 0.0 | 0 | 13 | 15 | 0.0583 |
| 4321 | native | B0_random | 4 | 3 | 0 | 0.0 | 2 | 4 | 4 | 0.0000 |
| 4321 | native | B1_current_hard | 1788 | 1788 | 18 | 1.0 | 1411 | 73 | 77 | 0.4112 |
| 4321 | native | B2_reliability | 761 | 8 | 0 | 0.0 | 0 | 56 | 73 | 0.3736 |
| 4321 | native | B3_reliability_vulnerability | 618 | 8 | 0 | 0.0 | 0 | 56 | 72 | 0.4028 |
| 4321 | native | B4_reliability_degradation | 4 | 2 | 0 | 0.0 | 0 | 2 | 3 | 0.1667 |
| 4321 | native | B5_safe_boundary_activation | 8 | 8 | 0 | 0.0 | 0 | 6 | 8 | 0.0000 |
| 4321 | native | B6_reliability_multi_pid | 415 | 8 | 0 | 0.0 | 0 | 46 | 65 | 0.4263 |
| 4321 | native | B7_reliability_bidirectional | 449 | 4 | 0 | 0.0 | 0 | 52 | 68 | 0.4157 |
| 4321 | budget_matched | B0_random | 4 | 3 | 0 | 0.0 | 3 | 3 | 4 | 0.0000 |
| 4321 | budget_matched | B1_current_hard | 4 | 4 | 0 | 0.0 | 4 | 2 | 4 | 0.0000 |
| 4321 | budget_matched | B2_reliability | 4 | 4 | 0 | 0.0 | 0 | 3 | 4 | 0.0000 |
| 4321 | budget_matched | B3_reliability_vulnerability | 4 | 4 | 0 | 0.0 | 0 | 3 | 4 | 0.0000 |
| 4321 | budget_matched | B4_reliability_degradation | 4 | 2 | 0 | 0.0 | 0 | 2 | 3 | 0.1667 |
| 4321 | budget_matched | B5_safe_boundary_activation | 4 | 4 | 0 | 0.0 | 0 | 3 | 4 | 0.0000 |
| 4321 | budget_matched | B6_reliability_multi_pid | 4 | 4 | 0 | 0.0 | 0 | 3 | 4 | 0.0000 |
| 4321 | budget_matched | B7_reliability_bidirectional | 4 | 4 | 0 | 0.0 | 0 | 4 | 4 | 0.0000 |

梯度范数只在同一 safe-boundary 关系代理上测量；未把少量 active 的 native mask 与大 active 集合直接做 raw norm 结论。原始和预算匹配统计的 bootstrap 95% CI 保存在 `selection_summary_seed*.json`。

- 运行命令：`/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_021426 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_021426 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_021426 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，因此不伪造本地 commit。
- 独立研究代码 SHA256：`25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`。
- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。
- 明确 checkpoint SHA256：seed1234=`e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`；seed4321=`d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。
