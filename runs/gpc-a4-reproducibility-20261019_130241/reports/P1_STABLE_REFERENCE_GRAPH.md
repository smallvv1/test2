# P1：稳定历史身份邻域图

运行：`20261021_082715`。数据仅来自 LTCC train split，未使用官方 query/gallery。

实际脚本：`/home/DIFFER/scripts/ndp_stable_reference_graph.py`。

输入为既有 9576 张、77 PID、1024 维 `feat[0]` 特征库中的 Bio E60 与 GPC E60。两者是独立训练终点，不是同一训练轨迹的相邻 checkpoint，因此本报告只支持“跨终点关系变化审计”，不支持时间稳定性结论。

固定规则：每个 seed 固定抽取 1024 个 anchor；异 PID Top-20；同 PID 参考正样本最多 4 个；自引用排除；图像路径、PID、衣物和相机字段逐项对齐。

| seed | 负邻域 Jaccard 均值/中位数 | 正邻域 Jaccard 均值 | Bio 最大 hubness | GPC 最大 hubness |
|---:|---:|---:|---:|---:|
|1234|0.2473 / 0.2500|0.5137|61|58|
|4321|0.4365 / 0.4286|0.5470|45|56|

这说明负邻域相对不稳定，且不同 seed 的稳定程度不同；不能把一次终点的稳定邻居直接视为可靠保护关系。历史 continuation 节点虽有 update 0/20/50/100/200/400 的完整状态，但当前没有对应的全量特征库，不能伪造同轨迹 Top-K Jaccard。

产物：

- `/home/DIFFER/outputs/ndp_second_innovation_20261021_082715/p1_seed1234/`
- `/home/DIFFER/outputs/ndp_second_innovation_20261021_082715/p1_seed4321/`

判定：`P1_STATIC_GRAPH_BUILT_BUT_TEMPORAL_STABILITY_INCONCLUSIVE`。
