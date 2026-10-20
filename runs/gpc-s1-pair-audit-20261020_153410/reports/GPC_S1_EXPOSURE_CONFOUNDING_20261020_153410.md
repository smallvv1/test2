# GPC S1-Exposure 混杂审计

## 为什么现有 S1 改变了 PID 和衣服关系

实际 `choose_s1` 在 `gpc_strict_budget_plan_audit.py:150-153` 使用：

```python
pool = unique_ordered(candidates[iteration]["id"], ...)
chosen = pool[:need]
```

它从当前 batch 的全部 same-ID 候选 `id` 中按 hash 选前 `need` 条，没有约束候选必须与对应 S0 关系拥有相同 PID，也没有约束同衣/跨衣类型。因此它保持 active batch 和数量，却同时改变了 PID 曝光与衣服关系构成。

## 20 步直接证据

| seed | S0 | S1 | 发生的混杂 |
|---:|---|---|---|
| 1234 | PID 58、71；2 条同衣 | PID 35、67；1 条同衣、1 条跨衣 | PID 与衣服类型均变化 |
| 4321 | PID 49；1 条同衣 | PID 6；1 条跨衣 | PID 与衣服类型均变化 |

因此现有 20 步 A/B 的 relation hash 改变并不等于“只改变正样本图像对”。

## 完整 epoch Exposure 统计

既有 `gpc_strict_budget_plan_20261020_094059` 显示 S1-Exposure 保持每个 rank 的 active batch 与关系总数，但不保持 PID 或同衣/跨衣比例：

| seed/rank | S0 关系 | S1 关系 | S0 同/跨衣 | S1 同/跨衣 | S0 PID数 | S1 PID数 |
|---|---:|---:|---|---|---:|---:|
| 1234/0 | 75 | 75 | 23/52 | 25/50 | 33 | 32 |
| 1234/1 | 78 | 78 | 23/55 | 24/54 | 30 | 31 |
| 4321/0 | 62 | 62 | 26/36 | 23/39 | 27 | 32 |
| 4321/1 | 86 | 86 | 31/55 | 37/49 | 37 | 39 |

这证明 Exposure 是一个不同机制：它不是纯图像对替换，而是“在相同数量约束下改变身份与衣服关系曝光”。其任何性能变化都不能单独归因于正样本对本身。

## 决策含义

现有 S1 结果只能作为 Exposure 探索记录，不能命名为纯正样本选择实验。若继续研究，必须预注册 S1-Pair 与 S1-Exposure 两个独立问题；本轮不启动任一训练。

