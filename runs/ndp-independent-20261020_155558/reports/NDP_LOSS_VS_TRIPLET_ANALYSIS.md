# NDP 与普通 Triplet 的损失/梯度关系分析

候选 NDP：

```text
g_current = sim(i,p)-sim(i,n)
L_NDP = mean(relu(g_ref - g_current - epsilon))
```

对每个固定的 `(i,p,n)`，令 `m_i=g_ref-epsilon`，则：

```text
relu(g_ref - g_current - epsilon) = relu(m_i - g_current)
```

所以它在同一三元组、同一激活区域内与 Triplet hinge 具有完全相同的当前特征梯度方向；区别在于 NDP 使用冻结参考模型产生逐关系自适应 margin，并只保护参考状态下可靠的关系。它不是一个与 Triplet 梯度正交的新信息源。

本轮运行 `/data/DIFFER/scripts/test_ndp_structure_losses.py`，结果为 `NDP_STRUCTURE_UNIT_TEST_PASS`：参考分支停止梯度，当前分支可产生有限梯度，空关系和可靠 mask 接口均可处理。该单元测试不是模型训练结果。

因此控制设计必须至少包括：

- B0：固定 Bio+GPC-v1；
- C1：B0 + 参考自适应 margin NDP；
- C2：B0 + 相同关系/相同预算的普通固定 margin Triplet 控制；
- C3：B0 + 不使用参考排序的局部一致性控制。

在没有跨 seed 的稳定真实退化信号、且 PPU AMP A/A 噪声仍未完全解决前，不执行这些短程更新比较。不能把“loss 能反向”解释成“比 Triplet 独立且有效”。

