# NDP 最小反应式原型设计

## 数学定义

对固定、有向、可靠的身份关系 `r=(i,p,n)`：

```text
h_r = ReLU(M_ref(r) - M_cur(r) - epsilon)
L_reactive = sum_r w_r h_r / sum_r w_r
```

参考 margin、可靠性 mask 和权重均 stop-gradient。关系改善或下降不超过 epsilon 时不产生保护梯度；没有合法关系时返回与当前张量连接的零损失，不产生 NaN。权重可设上限，避免 Hub 或重复关系支配。

实现：`scripts/ndp_reactive_prototype.py`；单测：`tests/test_ndp_reactive_prototype.py`。

## 已验证性质

执行命令：`PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER/tests /usr/local/bin/python3.10 -m pytest -q tests/test_ndp_reactive_prototype.py`。

- 改善关系零惩罚；
- 真实退化产生正损失，当前 margin 梯度方向为增大 margin；
- 参考分支没有梯度；
- 空关系、mask、有限性、权重上限安全；
- A→B 与 B→A 保留独立梯度；
- 本轮全测试 8/8 通过。

## 与 Hard Triplet 的关系

在单一正样本、单一负样本上，该函数的梯度形式与带参考 margin 的自适应 Triplet hinge 相同；因此不能仅凭“历史参考”或“单侧 ReLU”宣称独立创新。可能的额外信息只来自：可靠历史关系、固定多 PID 关系集合、双侧有向记录、Hub 上限和“只保护已退化关系”的门控。上述信息价值尚未通过更新实验验证。

因此当前状态是 `REACTIVE_PROTOTYPE_ONLY`，不是正式 NDP loss。不得接入 optimizer。

## 推荐的可推翻验证

若未来 A/A 通过，应固定历史参考与多 PID 关系，比较基础 Hard Triplet、自适应 margin、该反应式原型和匹配预算的多负样本控制；只在同一真实更新状态观察参数梯度与双侧退化。若梯度仅等价于自适应 Triplet，停止该候选。
