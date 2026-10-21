# NDP 真实轨迹自然激活报告

## 训练状态合法性

两个 seed 使用的 B0 manifest 都明确写有：`arm=B0`、`GPC-v1 原始 I3 matched-budget selector`、`Phase7I I3 deterministic SHA-256 ID-only selector`、`lambda=0.1`、`same I3 per-batch minimum count`，并标记 `gpc_v1_selector_preserved=true`。开发训练 PID 为60，验证 PID与训练 PID分离，训练期间未使用官方 query/gallery。

E60 状态文件完整，阶段2从该状态建立独立 continuation；两个 seed 各完成400个有效 optimizer update，AMP skip=0，学习率为 `2e-7`。这是一条新的受控 continuation，不把它解释为原60 epoch轨迹的中间点。

## 激活定义

阶段3固定64个 train-only anchor及相同正负关系，`tau=0.2`。宽松诊断定义为：

`advantage_drop = A_reference - A_current > 0.02`。

本报告的“可靠激活”进一步要求参考优势 `A_reference>0.05`。这一步只用于审计参考结构可靠性，不改变任何训练关系。

| seed | 固定组 | 可靠组 | update20 | update50 | update100 | update200 | update400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1234 宽松激活 | 64 | 38 | 22 | 22 | 20 | 13 | 37 |
| 1234 可靠激活 | 64 | 38 | 15 | 13 | 10 | 3 | 26 |
| 4321 宽松激活 | 64 | 36 | 11 | 9 | 18 | 27 | 26 |
| 4321 可靠激活 | 64 | 36 | 9 | 6 | 9 | 16 | 17 |

初始节点 update=0 两种子均为0，这是同一参考状态与自身比较时的预期结果。此后两个 seed 都出现可靠自然激活，因此 `REAL_TRAJECTORY_ACTIVATION` 已得到支持。

## 限制

激活来自固定 B0 continuation 的状态差异，不能证明 NDP 造成了这些局部退化，也不能证明加入 NDP 后会改善检索。阶段3使用 eval-mode、no-grad 重编码；后续梯度探针在真实节点重新前向，但仍不执行 optimizer.step。
