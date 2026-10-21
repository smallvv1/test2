# LRD-A/B/C float64 梯度矩阵审计

此前梯度余弦出现 `1.0026`，原因是诊断向量、范数和余弦均用 float32，导致大向量点积的舍入超出数学上限；不是通过输出 clamp 修复。

本轮改为 float64 计算原始 dot、L2 norm 和 cosine，并重新生成两个 seed 的候选矩阵。

最大观测余弦超界为 `5.35e-14`（seed1234）和 `3.64e-14`（seed4321），属于 float64 舍入；单位向量、正交向量、零向量测试均通过。计算使用显式 float64 点积和 L2 范数，未用 clamp 隐藏超界，也没有发现向量拼接错位或参数集合混入。

新结果：

- `/home/DIFFER/outputs/ndp_lrd_gradient_audit_20261021_040000/seed1234.json`
- `/home/DIFFER/outputs/ndp_lrd_gradient_audit_20261021_040000/seed4321.json`
- 审计脚本 SHA256：`ca1bb7d96b7e92a2b8f1f44b606acbc0c9c6fd156df1291e75044ba3fdbf8ada`。

代表性与自适应 Margin Triplet 的特征梯度余弦：

- LRD-A：0.8162 / 0.8058；
- LRD-B：0.1305 / 0.2570；
- LRD-C：0.1326 / 0.4573。

这些数值说明候选梯度可区分，但仍只是冻结特征梯度，不是完整 EVA02 参数更新效果。
