# NDP 同轨迹关系覆盖扩展

## 方法

使用已存在的 Bio+GPC continuation 节点 0/20/50/100/200/400，固定同一批 train-only 关系；在 PPU 上执行 `eval + no_grad` 重编码，不执行 optimizer.step。两个 continuation manifest 均记录 `gpc_v1_preserved=true`、双 PPU、无 CPU fallback。代码文件名含 `gpc_v2_train.py`，但 B0 标签明确为“GPC-v1 原始 I3 matched-budget selector”；本报告以 manifest 与 B0 标签为版本证据，不把脚本文件名误认为 GPC-v2 方法。

## 覆盖结果

执行命令（两个 seed 分别运行，PPU 环境变量已按已验证配置导出）：`PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK PYTHONUNBUFFERED=1 /usr/local/bin/python3.10 -u scripts/ndp_trajectory_coverage_expansion.py --seed {1234,4321} --trajectory-dir .../trajectory_seed{1234,4321} --resume-dir .../B0_seed{1234,4321} --output-dir coverage_expansion/seed{1234,4321}`。

| Seed | 固定 anchor | PID 数 | 唯一图像数 | update20 | update50 | update100 | update200 | update400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|159|30|382|62/159=38.99%|60/159=37.74%|53/159=33.33%|34/159=21.38%|82/159=51.57%|
|4321|137|31|340|22/137=16.06%|18/137=13.14%|37/137=27.01%|50/137=36.50%|47/137=34.31%|

扩大关系集合中，seed1234 有 9 个多正、6 个多负 PID 组；seed4321 有 15 个多正、11 个多负 PID 组。它比旧的 64 组诊断包含更多 PID 和多邻居结构，但仍是固定采样子集，不是 9576 张图像全量覆盖。

## 环境异常

seed1234 第一次尝试因未导出 `PPU_SDK/PPU_HOME` 失败，日志显示 `Both PPU_SDK and PPU_HOME are not exist`，不是模型 forward 结论。补齐已验证环境后重跑成功，输出状态 `PASS_PPU_EVAL_NO_GRAD`。失败日志保留在 `coverage_expansion/seed1234_retry1/coverage_seed1234_failed_env.log`。

## 证据与限制

结果：

- `coverage_expansion/seed1234_retry1/expanded_activation_summary.json`；
- `coverage_expansion/seed4321/expanded_activation_summary.json`。

源码 SHA256：`ndp_trajectory_coverage_expansion.py` = `028f730e4066918894ac3dbc51303949371954abb2586ee15244c2a5dd77c99a`。

这次扩展证明真实 PPU 节点可被无梯度读取并产生自然激活变化，但尚未建立逐负 PID 的完整 margin/翻转标签，也没有做任何 NDP 更新干预。
