# NDP 阶段3故障根因与修复报告

## 根因

旧恢复运行 `20261021_013814` 的 seed1234 子进程在模型前向之前失败。真实 traceback 位于：

`/home/DIFFER/scripts/ndp_real_trajectory_activation.py:181`

调用使用了未定义的 `protocol`，而上一行只保存了 `_protocol`。实际错误为：

`NameError: name 'protocol' is not defined`

因此该失败不是 PPU、checkpoint、邻域图或 NDP 损失失败，也没有执行模型计算。

监督器另有记录缺陷：`emit(stream, message, **failure)` 在 `failure` 也含 `message` 时会在进入函数前触发重复关键字错误，可能遮蔽原始异常。

## 修复

1. `ndp_real_trajectory_activation.py` 接收 `protocol = v2.install_protocol(...)` 并将其传入 `choose_groups()`。
2. `ndp_overnight_recovery_stage3.py` 将事件参数改为 `event_message`，并把冲突字段规范化为 `error_message`。
3. 每个子进程保存完整 command、PPU 环境白名单、returncode、日志字节数、日志 SHA256 和 `child_result.json`。
4. 子进程安装 `sys.excepthook`，异常时保存 `activation_error.json`；阶段状态使用原子 JSON 写入。
5. 本轮日志改存到 `/home/DIFFER/log_NDP/阶段3恢复_20261021_024248/`。

## 回归与重跑

修复后的静态编译和 `emit(..., message=...)` 冲突回归测试均通过。新阶段3运行：

`/home/DIFFER/outputs/ndp_overnight_recovery_20261021_024248/`

两个 seed 均 returncode=0，均生成 `activation_summary.json`，总耗时约80.82秒；阶段3只做前向审计，不执行 optimizer.step。

原始失败记录保留，没有覆盖。
