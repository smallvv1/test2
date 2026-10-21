# 反向 KNN 验证

图构建脚本已从“仅计算选定 anchor 的 Top-K”改为对所有候选负图像执行真实 Top-K 查询；anchor 或反向查询缺失时抛出异常，不再默认 False。

| Seed | 反向查询数 | 反向 Top-K 是否实际计算 | 未来节点参与选择 | 自引用 |
|---:|---:|---|---|---|
|1234|2153|是|否|排除|
|4321|2112|是|否|排除|

图关系源自 update0 train-only 特征，使用分块相似度，不执行 optimizer.step，也未使用官方 query/gallery/test。这个修正确保互为近邻标记有明确分母，但不证明互反关系本身值得保护。

证据：`/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/reference_graph/seed1234/multi_pid_reference_summary.json`、Seed4321 对应 JSON。
