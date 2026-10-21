# NDP 真实训练轨迹结果

## 基础轨迹

| seed | 来源 | 有效更新 | GPC激活batch | GPC激活率 | found_inf | 参数有限 |
|---:|---|---:|---:|---:|---:|---|
| 1234 | 历史B0 E60→新 continuation | 400 | 46 | 11.50% | 0 | 是 |
| 4321 | 历史B0 E60→新 continuation | 400 | 47 | 11.75% | 0 | 是 |

该 continuation 使用固定 Bio+GPC-v1、lambda=0.1 和原始 I3 selector 命名空间；它是从 E60 开始的新训练阶段，不是原始60 epoch轨迹的中间状态。

## NDP激活

没有生成 `activation_summary.json`。原阶段3因双重 `install_protocol()` 导致 `loader_v2` 二次 remap；新恢复阶段3修复后又因 `protocol`/`_protocol` 变量名错误退出。两个错误都发生在模型前向之前。

因此：真实 NDP 激活率、首次激活步、激活 PID 和局部身份指标均为 **未测得**，不能写成0，也不能写成自然激活。
