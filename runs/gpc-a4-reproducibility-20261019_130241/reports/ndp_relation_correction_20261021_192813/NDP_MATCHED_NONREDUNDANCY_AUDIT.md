# 预算匹配离线非冗余性审计

该审计固定 2560 条 update0 difficult-K5 关系，比较实际 Batch-Hard 重建覆盖、margin 代理、可靠性/脆弱性 mask、严格图像互反 mask 及数量匹配随机 mask。它没有执行网络 forward 或 optimizer.step；冻结 feature autograd 只验证局部梯度接口，因此不能宣称训练收益。

## Seed1234

| mask | 关系数 | 200→400 事件 | 组内事件率 | PID数 |
|---|---:|---:|---:|---:|
|B0_reconstructed_batch_hard|0|0|NA|0|
|B1_fixed_margin_proxy|2560|31|1.211%|77|
|B2_reference_margin_proxy|2498|31|1.241%|77|
|B3_random_count_matched_to_B5|310|7|2.258%|77|
|B3_random_count_matched_to_B6|32|0|0.000%|25|
|B3_random_count_matched_to_B7|4|0|0.000%|4|
|B4_current_hard_margin_proxy|1896|31|1.635%|74|
|B5_reliability_vulnerability|310|8|2.581%|59|
|B6_bidirectional_image_relation|32|0|0.000%|21|
|B7_reliability_plus_bidirectional|4|0|0.000%|4|

## Seed4321

| mask | 关系数 | 200→400 事件 | 组内事件率 | PID数 |
|---|---:|---:|---:|---:|
|B0_reconstructed_batch_hard|0|0|NA|0|
|B1_fixed_margin_proxy|2560|9|0.352%|77|
|B2_reference_margin_proxy|2508|9|0.359%|77|
|B3_random_count_matched_to_B5|298|0|0.000%|73|
|B3_random_count_matched_to_B6|28|1|3.571%|21|
|B3_random_count_matched_to_B7|2|0|0.000%|2|
|B4_current_hard_margin_proxy|1903|9|0.473%|75|
|B5_reliability_vulnerability|298|2|0.671%|59|
|B6_bidirectional_image_relation|28|0|0.000%|23|
|B7_reliability_plus_bidirectional|2|0|0.000%|2|

B0 直接关系覆盖为 0；B6/B7 样本数非常小（32/28 及 4/2），不能用其零事件支持机制优越性。B5 组虽有较高描述性事件率，但它的 vulnerability 定义含 margin 规则，且与随机数量匹配结果并未提供可确认的跨 seed 增量证据。梯度 JSON 报告 update0 同状态时 reactive hinge 不应激活；其“feature_space_only=true”，不等价网络参数梯度。

证据 JSON/CSV：`/home/DIFFER/outputs/ndp_relation_correction_20261021_192813/matched_nonredundancy/seed1234/`、Seed4321 对应目录。
