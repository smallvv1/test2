# NDP Hubness 去重审计

## 修复口径

同一固定负关系在五个时间区间重复出现，旧实现将其计数五次。新实现同时保留三种口径：

1. `time_repeated`：历史错误口径，仅用于对照；
2. `unique_image`：每个唯一 anchor→negative 图像关系只计一次；
3. `unique_pid`：每个 anchor→negative PID 只计一次。

统计按 seed 隔离，不混合两个 seed。

## 结果

执行命令：`/usr/local/bin/python3.10 -u scripts/ndp_risk_predictor_comparison.py --future-rows future_boundary_fixed/future_boundary_seed1234.csv future_boundary_fixed/future_boundary_seed4321.csv --bidirectional-rows bidirectional_fixed/bidirectional_boundary_seed1234.csv bidirectional_fixed/bidirectional_boundary_seed4321.csv --output-dir predictors_fixed`。

重复计数只把计数尺度放大约 5 倍，因此排序和 AUC 未改变，但语义错误。唯一图像 Hubness 的 AUC/AP 为：

- seed1234：`0.5150 / 0.0294`；
- seed4321：`0.3492 / 0.00995`。

唯一 PID Hubness 为：

- seed1234：`0.4699 / 0.0404`；
- seed4321：`0.5364 / 0.0189`。

方向不一致，不能作为稳定的 NDP 权重依据。seed1234 唯一图像中心性最大为 2、PID 最大为 7；seed4321 图像最大为 4、PID 最大为 11。由于关系覆盖很小，不能把这些中心性数字外推到全训练集。

证据：`outputs/ndp_future_boundary_20261021_094250/predictors_fixed/risk_predictor_summary.json`；源码 SHA256：`ndp_risk_predictor_comparison.py` = `700884d5984eb906b637faa9bc4395237f90b026b4ecf1a866a4dfde3063f6ec`。

结论：Hubness 去重已修复，但当前没有跨 seed 的独立增量价值证据，暂停将其纳入 NDP 权重。
