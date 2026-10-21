# LRD-C 数学等价性与可靠性审计

运行标识：`20261021_041500`。固定 Bio+GPC-v1，全部数据来自 LTCC train-only。

## 数学结论

令

`g = logsumexp(pos_logits) - logsumexp(neg_logits)`，则

`Q_positive = exp(g)/(1+exp(g)) = sigmoid(g)`。

因此

`KL(Bern(Q_ref) || Bern(Q_cur))`

对 `g_cur` 的导数为：

`dL/dg_cur = Q_cur - Q_ref`。

float64 单元测试已通过。由此确认 LRD-C 不是新的局部排序目标，而是对聚合正负身份质量 / 聚合 margin 的双向回归约束。

## LRD-A 与 LRD-C 的差异

- LRD-A：`relu(Q_ref-Q_cur-epsilon)`，只在正身份质量下降超过阈值时激活；Q+ 改善时梯度为 0。
- LRD-C：Bernoulli KL，没有单侧门控；Q+ 下降和 Q+ 上升都会产生梯度，Q+ 改善时会把当前质量拉回参考值。

因此两者有效门控不一致。旧实现中的 LRD-C 会错误约束有益的正身份质量提升。

## 真实危险状态复核

| seed | 参考可靠且边界正确→当前边界翻转 | 其中 Q+ 提升 | LRD-A 激活 | LRD-C 非零 | LRD-C 反向拉回 | DBP 边界激活 |
|---|---:|---:|---:|---:|---:|---:|
|1234|55|25|23|55|25|52|
|4321|41|14|24|41|14|38|

seed1234 Anchor 10：

- anchor：`/home/data/LTCC_ReID/train/000_1_c10_001016.png`，PID 0；
- 参考 Q+ `0.5487288`，当前 Q+ `0.6346858`；
- 参考最弱正 `0.8153957`，最强负 `0.7886940`，边界 `+0.0267017`；
- 当前最弱正 `0.8394769`，最强负 `0.8437031`，边界 `-0.0042261`；
- LRD-A：不激活；
- LRD-C：KL `0.6467379`，且 `dL/dg=+0.0859570`，会反向压低当前已改善的 Q+；
- DBP：激活，边界下降 `0.0309278`。

这确认了：LRD-A 会漏掉“聚合 Q+ 上升但最弱正/最强负边界翻转”的危险关系；LRD-C 能看到质量变化，但方向可能错误。

完整逐 Anchor 图片路径、PID、相似度和风险状态位于：
`/home/DIFFER/outputs/ndp_dbp_reliability_20261021_040000/seed1234.json` 与 `seed4321.json`。

## 参考可靠性

在 `g_ref>0.05` 下，参考边界正确→当前翻转分别涉及 29、28 个 PID。若进一步要求 `g_edge_ref>0.05`，只保留 19、9 个翻转，说明 `Q+` 可靠性和边界可靠性不是同一条件；阈值不能只靠 Q+ 定义。

## 文件

- 数学测试同时对实际 `lrd_c_cross_identity_kl` 实现做了一正一负关系的 autograd 回归，SHA256 `30f71018c68cfdac04dfd024e1ce06cf967deb779cfa5e0029f9b23cb5539fa2`。
- 审计脚本：`/home/DIFFER/scripts/ndp_dbp_reliability_audit.py`，SHA256 `b7279030918ec88f6f0bcf40b839b3d05aa2ea808de8ab7cde8fa37605526301`。
