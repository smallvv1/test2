# NDP 首次风险触发分析

运行批次：`20261021_094250`

## 证据与执行

本报告只复用了同一条 Bio+GPC-v1 continuation 的阶段3结果，没有重新训练、没有 optimizer.step，也没有读取官方 query/gallery/test。输入为两个 seed 各 64 组固定关系、节点 `0/20/50/100/200/400`，每个 seed 共 320 条相邻节点转移。实际脚本：

- `/home/DIFFER/scripts/ndp_future_boundary_risk.py`
- `/home/DIFFER/scripts/ndp_risk_benchmark.py`
- 输入：`/home/DIFFER/outputs/ndp_overnight_recovery_20261021_024248/`
- 结果：`/home/DIFFER/outputs/ndp_future_boundary_20261021_094250/future_boundary_recomputed/`

`future_boundary_risk.py` SHA256：`15c1c41e2056e750b68fc97e5f5c0558036f5c18c6a45eace9f20feb810e75a1`。

## 标签定义

- 持续风险：`D_target > 0.02`，研究已经发生的风险是否继续存在。
- 首次触发：`D_prev <= 0.02 且 D_target > 0.02`，只在前一节点尚未触发的关系上评估。
- 新增退化：`D_target-D_prev > 0.02`。
- `D` 为固定关系上的 `reference_advantage-current_advantage`。

风险分数只使用上一节点；目标节点仅生成标签。

## 结果

| seed | 持续事件率 | 持续 AUC/AP | 首次触发样本/正例 | 首次触发 AUC/AP（上一节点 drop） |
|---:|---:|---:|---:|---:|
| 1234 | 114/320=35.625% | 0.7263/0.6697 | 243/58 | 0.5445/0.2841 |
| 4321 | 91/320=28.438% | 0.7880/0.6596 | 255/47 | 0.6724/0.3061 |

首次触发的结果明显低于持续风险。高累计 AUC 主要包含“已经处于高 drop 的关系继续处于高 drop”的持续性，不能解释为新风险的可靠预测。

按区间的首次触发率：

| 区间 | seed1234 | seed4321 |
|---|---:|---:|
| 0→20 | 34.375% | 17.188% |
| 20→50 | 3.125% | 6.250% |
| 50→100 | 3.125% | 18.750% |
| 100→200 | 6.250% | 17.188% |
| 200→400 | 43.750% | 14.063% |

两 seed 的高风险区间不同，支持“训练阶段相关”而不是稳定、可直接在线使用的单一风险规则。

## 结论

`NEW_ONSET_PREDICTION=INCONCLUSIVE`。上一节点的 drop 对 seed1234 仅略高于随机，seed4321 有一定信号，但跨 seed 和区间不稳定；不能申请预测式 NDP 训练。

