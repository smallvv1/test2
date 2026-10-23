# 原始 Triplet 与反应式 NDP 冻结特征互补性

同一 update200 raw `feat[0]` 上，T0 使用原始欧氏距离、hard positive=max distance、hard negative=min distance、`SoftMarginLoss(dist_an-dist_ap)=softplus(dist_ap-dist_an)`；NDP 使用同一 K=5 关系池，先 FP32 L2 normalize，再使用 mean-positive/max-negative cosine Margin。没有把 normalized 检索库冒充原 Triplet 输入。

| seed | T0 grad norm | T2 reactive reliable | T3 reliable+observed | T0/T2 cosine | T0/T3 cosine | T2/T3 cosine |
|---|---:|---:|---:|---:|---:|---:|
|1234|0.064783|0.000338|0.007821|0.0486|0.0486|1.0000|
|4321|0.068664|0.000167|0.014005|0.0230|0.0230|1.0000|

T2/T3 等价是因为 reliability 内真正 active 的行就是 observed degradation 行；不是两个机制。低 cosine 仅说明在固定关系池的冻结特征空间方向不同，不能证明网络参数更新或检索收益。该实验是关系级 SoftMargin 代理，不是历史真实 Batch-Hard miner 记录。
