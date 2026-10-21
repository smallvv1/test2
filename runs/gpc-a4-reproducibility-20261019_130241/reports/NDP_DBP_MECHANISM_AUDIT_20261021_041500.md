# NDP-DBP 双尺度候选机制审计

## 候选定义

跨身份质量尺度：`Q_positive`。

局部边界尺度：

`g_edge = min(pos_similarity) - max(neg_similarity)`。

探索性损失：

`L_edge = relu(g_edge_ref - g_edge_cur - epsilon_edge)`，只对 `g_edge_ref>0` 的参考邻域使用。

本轮未接入正式 processor、未更新参数。

## 离线结果

- seed1234：55 个真实边界翻转，52 个被 `epsilon_edge=0.02` 的 DBP 探针激活；其中 Q+ 已提升的 25 个里有 24 个仍被 DBP 捕获。
- seed4321：41 个真实边界翻转，38 个被捕获；其中 Q+ 已提升的 14 个里有 12 个被捕获。

DBP 比 LRD-A 多使用了局部最弱正/最强负边界信息，因此能识别 LRD-A 漏检的 Anchor 10 类型事件。

## 与 Hard Triplet 的重叠

在固定正负集合上，`min(pos)-max(neg)` 正是 hard-positive/hard-negative 边界。因此 DBP 的 hinge 本身等价于固定关系集合上的自适应 Hard Triplet；它的潜在新增信息仅来自：

1. 参考边界可靠性掩码；
2. 与 Q+ 的双尺度联合门控；
3. 允许区分“Q+ 改善但局部边界恶化”。

这不足以单凭公式宣称独立创新。必须用同预算 Hard Triplet、LRD-A、LRD-C 控制组验证这些门控是否带来额外训练信息。

## 单元测试

已通过：同状态零损失、边界翻转激活、当前分支梯度有限、参考分支停止梯度、单/多邻居时与 hard margin 的重叠关系。

脚本：`/home/DIFFER/scripts/ndp_dbp_candidate.py`，SHA256 `4b4f7a2f533d31c2a87a3af57b37262cee0207818204cac2e339824bf8e2ee5d`。

测试：`/home/DIFFER/scripts/test_ndp_dbp_candidate.py`，SHA256 `5768b4c0959953f163033f02c4aa69caa6674ea4c5b556f0b417e23c71cf64a0`。

结论：DBP 是有用的风险识别/门控候选，但目前只能标记为机制可检验，不能标记为已验证独立方法。
