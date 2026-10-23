# Historical Safe Boundary 候选审计

预注册候选：`M_safe=max(0.05,Q25(M0,M20,M50,M100)-0.02)`；C 只对历史可靠关系计算。A 使用固定 update0 Margin，B 使用 update100，D 使用固定当前阈值0.05，E 使用 `softplus(-M_current)` 的 Batch-Hard 关系代理。A/B/C 的 reference 均停止梯度；当前改善（当前 Margin 高于目标）对应零 hinge。

| seed | 候选 | R内选择 | R内激活 | 平均原始值 | 原始值L2 |
|---|---|---|---|---|---|
| 1234 | A 固定 update0 | 643 | 35 | 0.000561081 | 0.095758 |
| 1234 | B 历史最后可信节点 | 643 | 12 | 0.000338647 | 0.0730875 |
| 1234 | C 历史 Q25 安全边界 | 643 | 17 | 7.70022e-05 | 0.016478 |
| 1234 | D 当前固定0.05 | 643 | 13 | 5.72557e-05 | 0.0146319 |
| 1234 | E Batch-Hard SoftMargin代理 | 643 | 643 | 0.633922 | 16.0873 |
| 4321 | A 固定 update0 | 628 | 9 | 6.39936e-05 | 0.0186015 |
| 4321 | B 历史最后可信节点 | 628 | 0 | 0 | 0 |
| 4321 | C 历史 Q25 安全边界 | 628 | 4 | 2.05511e-05 | 0.00763252 |
| 4321 | D 当前固定0.05 | 628 | 3 | 1.74138e-05 | 0.00737383 |
| 4321 | E Batch-Hard SoftMargin代理 | 628 | 628 | 0.633662 | 15.8908 |

C 若与 A/B 只有激活边界不同，不能称为独立机制；必须同时查看目标激活集合、梯度范数和 P4 局部影响。结果未执行更新，因此不构成训练收益证据。

- 运行命令：`/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_020821 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_020821 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_020821 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，因此不伪造本地 commit。
- 独立研究代码 SHA256：`f410368ae4a4fb2ed3fe4ce2a891eee4b3516c40ef40c66df8cc96550c55f463`。
- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。
- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。
