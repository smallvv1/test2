# NDP 独立机制研究归档

运行时间：`20261020_155558`

本归档封存 S1 `NO_GO`，并记录固定 Bio+GPC-v1 基线上的 LTCC train-only NDP 局部邻域诊断。没有启动 NDP 训练、30 epoch、60 epoch 或官方 LTCC 测试。

## 结论

- `S1_FINAL_DECISION=NO_GO`
- `NDP_PROTOTYPE_INCONCLUSIVE`
- NDP-A 离线 promotion gate：`MIXED/STOP`
- PPU AMP 算子级技术债务：`AMP_OPERATOR_ROOT_CAUSE_UNRESOLVED`

## 安全边界

归档只包含源码、中文报告、日志和小型统计 JSON/CSV。未包含 checkpoint、特征库、Caption 特征、数据集、密钥或 Token。

## 核心输入

- LTCC train-only：9576 图、77 PID、256 衣物状态。
- 诊断 K：5、10、20。
- 参考阈值：`g_ref > 0.05`。
- 容忍量：`epsilon=0.02`。

