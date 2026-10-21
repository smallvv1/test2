# NDP 夜间实验取证审计

## 结论

昨晚实验确实启动并执行：阶段0、阶段1通过，阶段2两种子各完成400个有效更新；阶段3失败，阶段4/5未启动。没有活动训练进程，也没有超时失控任务。

原运行：`20261020_201841`，记录时间约13分04秒。恢复审计：`20261021_013814`，运行约16秒后在阶段3失败。

## 阶段状态

| 阶段 | 原运行状态 | 恢复结论 |
|---|---|---|
| 0 PPU/环境探针 | COMPLETED/PASS | 可复用 |
| 1 完整状态恢复 | COMPLETED/PASS | 两 seed E60 rank 状态完整 |
| 2 Bio+GPC continuation | COMPLETED/PASS | seed1234、4321均400有效更新 |
| 3 NDP激活率 | FAILED/BLOCKED | 原运行因二次 loader 包装失败；恢复运行又因变量名错误失败 |
| 4 NDP梯度增量 | NOT_STARTED | 未执行 |
| 5 B0/NDP短程干预 | NOT_STARTED | 未执行 |

## 关键证据

- 原运行状态：[stage_status.json](/home/DIFFER/outputs/ndp_overnight_20261020_201841/stage_status.json)
- 原运行故障：[failure_events.json](/home/DIFFER/outputs/ndp_overnight_20261020_201841/failure_events.json)
- 恢复状态：[stage_status.json](/home/DIFFER/outputs/ndp_overnight_recovery_20261021_013814/stage_status.json)
- 恢复主日志：`/home/DIFFER/logs/NDP夜间恢复阶段3_20261021_013814.log`

没有重新训练，没有使用官方 LTCC 测试，没有修改 GPC-v1。
