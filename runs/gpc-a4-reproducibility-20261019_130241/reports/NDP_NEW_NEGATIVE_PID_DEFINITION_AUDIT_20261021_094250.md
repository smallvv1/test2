# NDP 新增错误 PID 定义审计

## 定义

对固定的 anchor、正样本和异身份 PID，只有在同一固定关系满足：

```text
M_prev > 0 且 M_target <= 0
```

才记为一个新增错误 PID。此前已经错误的 PID、恢复正确的 PID 和始终正确的 PID 分开统计。多个同 PID 图像不能重复计数。

## 结果

| Seed | 固定组 | 可分解单图像关系组 | 不可分解组 | 可分解 transition 行 | 新增错误 PID事件 | 恢复事件 | 已有错误事件 |
|---|---:|---:|---:|---:|---:|---:|---:|
|1234|64|56|8|280|5|6|72|
|4321|64|50|14|250|2|1|98|

此前严格 aggregate flip 为 6/2；seed1234 有 1 条翻转来自不可分解的聚合关系，因此可辨识的单一负 PID 新增事件只有 5 条。seed4321 的 2 条均在可分解关系中。

## 多 PID 限制

seed4321 有 5 行 `negative_pid_count>1`；但阶段3 CSV 只保存聚合 advantage，没有每个负 PID 的独立相似度。因此无法判断哪个 PID 翻转，不能声称已完成真正多 PID 侵入统计。seed1234 的不可分解组含多图像/聚合关系，不能按图像拆成 PID 事件。

状态：`MULTI_PID_DECOMPOSITION=NOT_AVAILABLE_WITH_CURRENT_STAGE3_CSV`。

## 证据

执行命令：`/usr/local/bin/python3.10 -u scripts/ndp_new_negative_pid_audit.py --rows future_boundary_fixed/future_boundary_seed1234.csv future_boundary_fixed/future_boundary_seed4321.csv --output-dir new_negative_pid`。

结果：`outputs/ndp_future_boundary_20261021_094250/new_negative_pid/new_negative_pid_summary.json`；逐行结果：`new_negative_pid_rows.csv`。

源码 SHA256：`ndp_new_negative_pid_audit.py` = `37b2334ae32fe7c27385f94a9445e65e1a43a60d5e216235ce2450530fe918d3`。

本阶段没有使用官方 query/gallery/test，也没有 optimizer.step。
