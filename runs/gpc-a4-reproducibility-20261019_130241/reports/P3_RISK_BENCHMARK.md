# P3：风险排序与消融基准

本阶段使用 Bio E60→GPC E60 的固定终点差异作为探索性标签：参考选定边界大于 0、当前固定关系边界不大于 0，定义为 endpoint-shift boundary flip。由于两个输入不是同一轨迹的未来 checkpoint，`online_predictive_valid=false`，所有 ROC/PR 数值不能解释为在线风险预测性能。

| seed | 方法 | ROC-AUC | PR-AUC | Top10 Precision | Top10 Recall |
|---:|---|---:|---:|---:|---:|
|1234|R0 Hard-margin drop|0.1062|0.3690|0.2233|0.4894|
|1234|R1 DBP|0.1062|0.3690|0.2233|0.4894|
|1234|R2 Soft intrusion|0.4874|0.1991|0.1359|0.2979|
|1234|R3 + history stability|0.4887|0.2113|0.1262|0.2766|
|1234|R4 + hubness control|0.4889|0.1915|0.1359|0.2979|
|1234|R5 + bidirectional|0.4348|0.1209|0.1165|0.2553|
|4321|R0 Hard-margin drop|0.1462|0.1710|0.1456|0.4839|
|4321|R1 DBP|0.1462|0.1710|0.1456|0.4839|
|4321|R2 Soft intrusion|0.2850|0.2954|0.1553|0.5161|
|4321|R3 + history stability|0.3090|0.2256|0.1456|0.4839|
|4321|R4 + hubness control|0.2816|0.3024|0.1456|0.4839|
|4321|R5 + bidirectional|0.3147|0.2026|0.1068|0.3548|

R0 与 R1 完全相同，说明当前 DBP 在这个固定边界标签上没有超出 hard-margin 的额外检测信息。R2–R5 的方向跨 seed 不稳定，不能支持复杂化风险模型。由于缺少同轨迹未来特征，不能把这些数值当作 P3 在线预测通过。

结果：`P3_ENDPOINT_SHIFT_ONLY`，不进入 NDP optimizer-step 试验。

产物：`/home/DIFFER/outputs/ndp_second_innovation_20261021_082715/p3_benchmark/`。
