# NDP P3 指标修复与重算

运行：`20261021_085235`。数据仍是独立 Bio E60 / GPC E60 终点之间的回顾性 endpoint shift；`online_predictive_valid=false`，不能解释为在线风险预测。

## 修复

旧 `auc_rank()` 按降序排序，却套用了升序 Mann–Whitney 公式，导致 ROC-AUC 反向。新实现使用升序分数、并列平均秩，并与 `sklearn.metrics.roc_auc_score` 交叉核对。PR 指标明确命名为 Average Precision，不再称为积分型 PR-AUC。非有限分数被显式计数，不能静默进入排序。

`active_count` 也修复：旧代码对已经执行 `relu(cur-ref-epsilon)` 的结果再次比较 epsilon，实际门槛变成 `cur-ref>2epsilon`；现改为 `violation>0`。

## 重算结果

| seed | 方法 | ROC-AUC | AP | Top10 P | Top10 R |
|---:|---|---:|---:|---:|---:|
|1234|R0 Hard margin|0.893835|0.369044|0.223301|0.489362|
|1234|R1 DBP|0.893835|0.369044|0.223301|0.489362|
|1234|R2 Soft intrusion|0.512620|0.199112|0.135922|0.297872|
|1234|R3 + history|0.511956|0.211146|0.126214|0.276596|
|1234|R4 seed-local hubness|0.511618|0.179905|0.135922|0.297872|
|1234|R5 bilateral|0.565191|0.120861|0.116505|0.255319|
|4321|R0 Hard margin|0.853815|0.171007|0.145631|0.483871|
|4321|R1 DBP|0.853815|0.171007|0.145631|0.483871|
|4321|R2 Soft intrusion|0.715038|0.295409|0.155340|0.516129|
|4321|R3 + history|0.690966|0.225624|0.145631|0.483871|
|4321|R4 seed-local hubness|0.718806|0.302429|0.145631|0.483871|
|4321|R5 bilateral|0.685281|0.202586|0.106796|0.354839|

全部12个方法/seed的 ROC-AUC 与 sklearn 差异为 0 或 `1.11e-16`，AP 差异不超过 `1.11e-16`；本次无非有限风险分数。R0/R1 完全相同，说明 DBP 在这个边界翻转标签上没有额外信息。R2–R5 的方向和幅度跨 seed 不稳定，不能支持复杂化损失。

## 产物与源码

- 修复脚本：`/home/DIFFER/scripts/ndp_risk_benchmark.py`
- 单测：`/home/DIFFER/scripts/tests/test_ndp_risk_modules.py`
- 结构化结果：`/home/DIFFER/outputs/ndp_metric_audit_20261021_085235/p3_corrected/`
- 日志：`/home/DIFFER/log_NDP/ndp_metric_audit_20261021_085235/metric_and_consistency.log`

结论：指标实现已修复；结果仍为历史终点诊断，不构成 NDP 在线预测或 optimizer-step 证据。
