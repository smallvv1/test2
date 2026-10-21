# NDP 严格翻转与稳健翻转审计

## 定义

保留历史严格翻转：`M_prev>0 且 M_target<=0`。另做预注册敏感性分析：

```text
M_prev > delta 且 M_target < -delta
delta ∈ {1e-4, 1e-3, 1e-2}
```

阈值不是根据测试结果选择的主结论。

## 结果

| Seed | 严格翻转 | delta=1e-4 | delta=1e-3 | delta=1e-2 |
|---|---:|---:|---:|---:|
|1234|6|6|6|4|
|4321|2|1|1|0|

seed4321 有一条目标 margin 约 `-2.2888e-5` 的近零事件；在 1e-4/1e-3 下已被稳健标签排除，属于 `BORDERLINE`，不能视为强身份混淆证据。seed1234 在 1e-2 下有 2 条严格但不稳健事件。

delta=1e-3 时，翻转发生在 seed1234 的 20→50、50→100、200→400，seed4321 的 200→400。样本仍少，不能仅用 ROC-AUC 证明动态预测机制。

## 预测对照

执行命令：`/usr/local/bin/python3.10 -u scripts/ndp_robust_boundary_flip_audit.py --rows future_boundary_fixed/future_boundary_seed1234.csv future_boundary_fixed/future_boundary_seed4321.csv --output-dir robust_flip --deltas 1e-4 1e-3 1e-2`。

在前一节点正确的关系上，距离零边界的分数对严格/稳健翻转的 AUC：

- seed1234：约 `0.9683`，AP `0.4233`（delta≤1e-3）；上一段下降量 AUC `0.3593`。
- seed4321：约 `1.0000`，AP `1.0000`，但仅 1 个稳健正例；上一段下降量 AUC `0.0000`。

这主要说明边界邻近本身是强静态风险信号，不等于预测某次训练更新造成的损害。

证据：`outputs/ndp_future_boundary_20261021_094250/robust_flip/robust_boundary_flip_summary.json`。

## 结论

严格翻转不应全部等权解释；真实强翻转数量在小样本下约为 seed1234=4、seed4321=0（delta=1e-2）。预测式 NDP 仍不能晋级。
