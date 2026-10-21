# NDP 日志归档根因

## 空日志的真实原因

GitHub 中以下四个0字节文件：

- `NDP夜间自动实验_20261020_201841.log`
- `NDP夜间自动实验_20261020_201841_恢复1.log`
- `NDP夜间阶段3激活审计_20261020_201841.log`
- `NDP夜间阶段3激活审计_20261020_201841_恢复1.log`

均是外层 shell/supervisor 输出文件。监督器本身没有向 stdout 打印阶段事件；真正的子进程 stdout/stderr 被重定向到了运行目录：

`/home/DIFFER/outputs/ndp_overnight_20261020_201841/logs/`

其中基础轨迹日志分别为3529和31339字节，激活审计失败日志为814字节，说明实验确实执行过。不是 Python 缓冲或“文件尚未写完”的竞态主因，也不是 PPU 没有启动。

## 已验证的修复方向

恢复监督器使用 `/usr/local/bin/python3.10 -u`、`PYTHONUNBUFFERED=1`，主日志采用逐事件 flush；状态 JSON 使用临时文件后 `os.replace` 原子替换；子进程日志与主日志分开保存，并在完成时记录 returncode、异常和 SHA256。

恢复日志本身已非空：[NDP夜间恢复阶段3_20261021_013814.log](/home/DIFFER/logs/NDP夜间恢复阶段3_20261021_013814.log)。

GitHub 旧归档应标记为 `ARCHIVE_INCOMPLETE`，不能据空文件判断实验未执行。
