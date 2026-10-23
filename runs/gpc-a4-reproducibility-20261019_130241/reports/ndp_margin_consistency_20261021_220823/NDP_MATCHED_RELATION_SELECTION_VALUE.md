# 预算匹配关系选择价值审计

候选池固定为每 seed 2560 条 K=5 关系；update400 只作事后标签，选择仅使用 update0→200。随机基线在同一候选池中匹配 selected 数量，重复 200 次，完整结果在 `outputs/.../matched_selection/`。

| seed | 方法 | selected | active | 200→400翻转 | flip rate | anchor PID | negative PID |
|---|---|---:|---:|---:|---:|---:|---:|
|1234|current hard|1760|150|28|0.0159|73|75|
|1234|reliability|763|33|0|0|58|73|
|1234|reliability+observed|33|33|0|0|16|21|
|1234|observed|183|183|0|0|36|53|
|4321|current hard|1788|59|18|0.0101|73|77|
|4321|reliability|755|9|0|0|56|73|
|4321|reliability+observed|9|9|0|0|6|6|
|4321|observed|68|68|0|0|14|33|

reliability+observed 与反应式 active 是同一关系集，不能宣称独立。current hard 更容易包含未来翻转，但它是边界邻近的回顾性关联，不是因果预测。当前没有证据表明历史门控超过预算匹配随机/当前 Margin 基线。
