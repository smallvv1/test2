# P2：多身份软侵入风险审计

运行：`20261021_082715`。仅使用冻结的 LTCC train-only Bio/GPC 特征。

实现：`/home/DIFFER/scripts/ndp_intrusion_risk_audit.py`。

规则已冻结为：同 PID 正样本最多 4 个；每个异 PID只选一个 reference 代表；选择最多 20 个异 PID；正身份阈值为正样本相似度的 0.25 分位数；`tau=0.07`；current 使用相同图像索引，不重新挑选邻居。

定义：

`u_i = Quantile_0.25(sim(i,P_i))`；
`v_ic = sim(i, representative(c))`；
`I_i = mean(sigmoid((v_ic-u_i)/tau))`。

| seed | 有效 anchor | `Delta_I>0` 比例 | 参考边界正确数 | 边界翻转数 | 双向关系可审计行数 |
|---:|---:|---:|---:|---:|---:|
|1234|1024|7.52%|985|47|910|
|4321|1024|42.97%|990|31|940|

`Delta_I` 均值分别为 `-0.02270` 和 `-0.00139`，方向和分布明显跨 seed 不一致。说明多 PID soft intrusion 捕获的信息不稳定，不能直接作为新损失的在线选择器。

Hubness 统计显示 sampled anchor 中最高代表出现次数为 61/45；因此若未来加入 hubness 修正，必须单独设置匹配控制，不能先验认为高频邻居都是错误样本。

P2 仍属于跨独立终点的静态诊断，不是同一训练更新的风险因果证据。

产物：

- `/home/DIFFER/outputs/ndp_second_innovation_20261021_082715/p2_seed1234/`
- `/home/DIFFER/outputs/ndp_second_innovation_20261021_082715/p2_seed4321/`
