# GPC-S1 研究封存报告

本报告对应 S1 审计运行 `20261020_153410`，不覆盖历史结果。

## 封存结论

`S1_FINAL_DECISION=NO_GO`

S1 不再作为独立正样本选择方法继续训练或推广。历史 S1 的 20 步、100 步、30 epoch 和 60 epoch 结果均保留为只读负结果；本轮没有重新执行 S1。

## 证据

- A/A 配对只有 2/6 正常完成；4/6 因 PPU AMP `found_inf` 导致两条轨迹有效更新数不一致，算子级根因仍为 `AMP_OPERATOR_ROOT_CAUSE_UNRESOLVED`。
- 严格 S1-Pair 只能在 `55/301 = 18.27%` 的冻结关系上保持 PID、同衣/跨衣类型并替换具体图像对；其余关系没有合法替代。
- 原始 S1-Exposure 的 selector 从全部 same-ID 候选中选择，改变了 PID 曝光和同衣/跨衣比例，不能解释为纯正样本对选择收益。
- 现有短程 S1 没有稳定、可信的训练效用证据。

## 保护声明

- GPC-v1、Phase 7I/7O、Bio、CE、Triplet、历史 checkpoint、PRCC、NDP 均未修改。
- 没有读取官方 LTCC query/gallery 进行 S1 选择。
- 没有因 S1 负结果删除或覆盖任何日志、JSON、checkpoint。

## 证据路径

- [S1 Go/No-Go](/home/DIFFER/reports/GPC_S1_FINAL_GO_NO_GO_20261020_153410.md)
- [S1-Pair 可行性](/home/DIFFER/reports/GPC_S1_PAIR_FEASIBILITY_AUDIT_20261020_153410.md)
- [S1 Exposure 混杂](/home/DIFFER/reports/GPC_S1_EXPOSURE_CONFOUNDING_20261020_153410.md)
- [S1 AMP 有效干预](/home/DIFFER/reports/GPC_EFFECTIVE_AMP_INTERVENTION_20261020_153410.md)
- [A/A 根因](/home/DIFFER/reports/GPC_AA_FAILURE_ROOT_CAUSE_20261020_153410.md)

因此，当前有效基线仍为固定的 Bio+GPC-v1；后续 NDP 诊断与 GPC 正样本关系选择完全解耦。
