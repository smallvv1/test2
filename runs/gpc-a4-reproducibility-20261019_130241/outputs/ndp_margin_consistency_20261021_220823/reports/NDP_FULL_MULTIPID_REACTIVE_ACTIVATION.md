# K=5 多 PID 反应式 NDP 激活审计

关系冻结于 update0 的 `difficult_k5` 图，后续节点只评估。每个 seed 有 512 anchors、2560 条关系、77 个 anchor PID，每个 anchor 5 个不同负 PID；节点为 0/20/50/100/200/400，PPU `eval/no_grad` 导出 raw 与 normalized `feat[0]`。

| seed | mode | A类可靠 | 当前脆弱 | 0→200退化 | active | 200→400翻转 |
|---|---|---:|---:|---:|---:|---:|
|1234|q25|630|1896|207|33|31|
|1234|mean|763|1760|183|33|28|
|1234|single|852|1669|214|59|29|
|4321|q25|625|1903|80|9|9|
|4321|mean|755|1788|68|9|18|
|4321|single|860|1670|90|10|12|

mean 主定义的冻结特征梯度：seed1234 reliability `selected=763, active=33, loss=0.00042184, grad_norm=0.0106678`；seed4321 `755, 9, 0.00007743, 0.0052709`。均有限。

`reliability+vulnerability` 两个 seed 都为 0，不是漏激活：A类可靠性包含历史最小 Margin≥0.05 且历史窗口含 update200，而 vulnerability 使用当前 Margin≤0.05/波动条件，二者逻辑互斥。该耦合不应被包装为“可靠且脆弱关系不存在”。

K=5 关系与特征 CSV/NPZ 位于 `/home/DIFFER/outputs/ndp_margin_consistency_20261021_220823/`；全程无更新、无测试集。
