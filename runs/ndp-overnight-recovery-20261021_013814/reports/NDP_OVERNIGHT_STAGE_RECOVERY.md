# NDP 夜间阶段恢复报告

## 已恢复成果

原运行的 E60 完整状态可以复用：seed1234 和 seed4321 都有模型、optimizer、scheduler、GradScaler、RNG 和 rank0/rank1 状态。阶段2输出的 continuation 节点完整保留，未覆盖。

| seed | 有效更新 | GPC激活batch | 激活率 | AMP skip |
|---:|---:|---:|---:|---:|
| 1234 | 400 | 46 | 11.50% | 0 |
| 4321 | 400 | 47 | 11.75% | 0 |

## 恢复尝试

新恢复目录：`/home/DIFFER/outputs/ndp_overnight_recovery_20261021_013814/`

恢复阶段3只建立了5个旧 continuation 模型节点的符号链接，没有复制或修改 checkpoint。seed1234 在启动激活脚本后立即失败，尚未进入 EVA02 forward；seed4321 未启动。

失败异常：

```text
NameError: name 'protocol' is not defined
```

原因是修复 `choose_groups()` 后，`main()` 的局部变量仍命名为 `_protocol`，调用处却引用 `protocol`。该恢复运行标记为 `FAILED`，不是科学实验结果。

## 后续恢复条件

只需修复变量名并增加静态测试，再从已有 continuation 节点重跑阶段3；不需要重做阶段2。当前不允许进入阶段4/5。
