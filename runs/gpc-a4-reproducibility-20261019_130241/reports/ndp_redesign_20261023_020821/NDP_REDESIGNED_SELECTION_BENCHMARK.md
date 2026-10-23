# P3 修复后预算匹配关系选择

候选池固定为每 seed 512 anchors × 5 negative PID = 2560 条。native 表保留方法本身的自然选中量；budget_matched 表以各 seed 非空 native 方法中的最小关系数为共同预算，并且 B0/B1/B2--B7 均不看 update400。B3--B7 的预算版是‘门控优先、当前 Margin 作 tie-break’的可比版本；原生 mask 仍单独报告，避免把补齐后的版本误称为纯交集。随机控制使用200个固定 RNG。

| seed | 版本 | 方法 | selected | active | flip | 覆盖全部flip | 当前错误 | anchor PID | negative PID | neg Gini |
|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | native | B0_random | 2560 | 1900 | 31 | 1.0 | 1540 | 77 | 77 | 0.3737 |
| 1234 | native | B1_current_hard | 1896 | 1896 | 31 | 1.0 | 1540 | 74 | 77 | 0.4077 |
| 1234 | native | B2_reliability | 643 | 17 | 0 | 0.0 | 0 | 52 | 72 | 0.4109 |
| 1234 | native | B3_reliability_vulnerability | 528 | 17 | 0 | 0.0 | 0 | 52 | 70 | 0.4273 |
| 1234 | native | B4_reliability_degradation | 13 | 7 | 0 | 0.0 | 0 | 4 | 9 | 0.2222 |
| 1234 | native | B5_safe_boundary_activation | 17 | 17 | 0 | 0.0 | 0 | 10 | 13 | 0.1900 |
| 1234 | native | B6_reliability_multi_pid | 378 | 16 | 0 | 0.0 | 0 | 47 | 67 | 0.4773 |
| 1234 | native | B7_reliability_bidirectional | 359 | 7 | 0 | 0.0 | 0 | 50 | 63 | 0.4128 |
| 1234 | budget_matched | B0_random | 13 | 12 | 0 | 0.0 | 12 | 10 | 13 | 0.0000 |
| 1234 | budget_matched | B1_current_hard | 13 | 13 | 0 | 0.0 | 13 | 2 | 11 | 0.1259 |
| 1234 | budget_matched | B2_reliability | 13 | 13 | 0 | 0.0 | 0 | 10 | 12 | 0.0705 |
| 1234 | budget_matched | B3_reliability_vulnerability | 13 | 13 | 0 | 0.0 | 0 | 10 | 12 | 0.0705 |
| 1234 | budget_matched | B4_reliability_degradation | 13 | 7 | 0 | 0.0 | 0 | 4 | 9 | 0.2222 |
| 1234 | budget_matched | B5_safe_boundary_activation | 13 | 13 | 0 | 0.0 | 0 | 10 | 12 | 0.0705 |
| 1234 | budget_matched | B6_reliability_multi_pid | 13 | 12 | 0 | 0.0 | 0 | 10 | 12 | 0.0705 |
| 1234 | budget_matched | B7_reliability_bidirectional | 13 | 7 | 0 | 0.0 | 0 | 13 | 11 | 0.1259 |
| 4321 | native | B0_random | 2560 | 1904 | 9 | 1.0 | 1583 | 77 | 77 | 0.3638 |
| 4321 | native | B1_current_hard | 1903 | 1903 | 9 | 1.0 | 1583 | 75 | 77 | 0.4027 |
| 4321 | native | B2_reliability | 628 | 4 | 0 | 0.0 | 0 | 51 | 71 | 0.3788 |
| 4321 | native | B3_reliability_vulnerability | 518 | 4 | 0 | 0.0 | 0 | 51 | 71 | 0.4103 |
| 4321 | native | B4_reliability_degradation | 3 | 1 | 0 | 0.0 | 0 | 2 | 2 | 0.1667 |
| 4321 | native | B5_safe_boundary_activation | 4 | 4 | 0 | 0.0 | 0 | 4 | 4 | 0.0000 |
| 4321 | native | B6_reliability_multi_pid | 358 | 4 | 0 | 0.0 | 0 | 45 | 68 | 0.4737 |
| 4321 | native | B7_reliability_bidirectional | 362 | 0 | 0 | 0.0 | 0 | 47 | 68 | 0.4309 |
| 4321 | budget_matched | B0_random | 3 | 2 | 0 | 0.0 | 2 | 3 | 3 | 0.0000 |
| 4321 | budget_matched | B1_current_hard | 3 | 3 | 0 | 0.0 | 3 | 2 | 3 | 0.0000 |
| 4321 | budget_matched | B2_reliability | 3 | 3 | 0 | 0.0 | 0 | 3 | 3 | 0.0000 |
| 4321 | budget_matched | B3_reliability_vulnerability | 3 | 3 | 0 | 0.0 | 0 | 3 | 3 | 0.0000 |
| 4321 | budget_matched | B4_reliability_degradation | 3 | 1 | 0 | 0.0 | 0 | 2 | 2 | 0.1667 |
| 4321 | budget_matched | B5_safe_boundary_activation | 3 | 3 | 0 | 0.0 | 0 | 3 | 3 | 0.0000 |
| 4321 | budget_matched | B6_reliability_multi_pid | 3 | 3 | 0 | 0.0 | 0 | 3 | 3 | 0.0000 |
| 4321 | budget_matched | B7_reliability_bidirectional | 3 | 0 | 0 | 0.0 | 0 | 3 | 3 | 0.0000 |

梯度范数只在同一 safe-boundary 关系代理上测量；未把少量 active 的 native mask 与大 active 集合直接做 raw norm 结论。原始和预算匹配统计的 bootstrap 95% CI 保存在 `selection_summary_seed*.json`。

- 运行命令：`/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_020821 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_020821 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_020821 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，因此不伪造本地 commit。
- 独立研究代码 SHA256：`f410368ae4a4fb2ed3fe4ce2a891eee4b3516c40ef40c66df8cc96550c55f463`。
- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。
- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。
