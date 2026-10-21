# NDP PPU 同精度 A/A 算子隔离

探针：`scripts/ndp_ppu_same_precision_kernel_probe.py`，PPU `cuda:0`、torch 2.6.0、float32、固定 CPU 生成输入与固定上游梯度、每个算子重复3次、无 optimizer.step。

LayerNorm backward 的相邻重复相对 L2 为 1.337083～1.413252，余弦为 0.000220～0.050444，且有限但输出哈希不同。Q projection、attention matmul+softmax、独立 softmax、output projection 的相同探针均为 max_abs=0 或接近机器精度、余弦≈1。

因此，LayerNorm backward 是当前首要候选根因，与完整模型中 block22/norm1 附近的历史观测一致；但该探针不是 EVA02 fused block22 的最小复现，不能宣称已经定位到唯一底层 kernel。A/A 状态仍为 `FAIL`，不能进行 optimizer-step 因果比较。

另有一次轨迹评估子进程因未继承 `PPU_SDK/PPU_HOME` 退出码134，原始日志保留；使用同一 PPU 环境修复后两 seed 均成功，说明该异常是运行环境传递错误，不是模型结论。


## 可审计执行信息

- 运行目录：`/home/DIFFER/outputs/ndp_true_multipid_20261021_103952`；报告目录：`/home/DIFFER/reports/ndp_true_multipid_20261021_103952`。
- 远端 main 检查到：`dfb93095a9c5ac9eee36a6b0c58c5d4b9b86f55e`；服务器工作树 HEAD：`573b0502a231568d509fe3a463664c1a3c42e258`，未推送、未覆盖 main。
- 主要命令：`/usr/local/bin/python3.10 -u scripts/ndp_full_update0_feature_extract.py --seed 1234/4321 ...`；`ndp_multi_pid_reference_graph.py --max-anchors 512 --k-pid 1 3 5 10`；`ndp_multi_pid_trajectory_eval.py --graph-type difficult --k-pid 5 --max-anchors 512`；`ndp_ppu_same_precision_kernel_probe.py --repeats 3`；`PYTHONPATH=/home/DIFFER/scripts:/home/DIFFER/tests /usr/local/bin/python3.10 -m pytest ...`。
- PPU：`/usr/local/bin/python3.10`、torch 2.6.0、PPU-ZW810E、model 计算仅使用 `cuda:0`，`world_size=2`；本轮无 optimizer.step。
- 数据：LTCC train-only；官方 query/gallery/test 未读取；CPU 仅用于离线图构建和统计。
- 固定基线文件指纹：`gpc_v2_train.py=d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`；`make_loss.py=9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043`；`eva02_l_bio.yml=c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97`。
- 新增源码 SHA256：
  - `ndp_historical_relation_reliability.py`: `45106dd15e67e69dd5864d18a24758a1e49aec0b5720c1ea205ab8aab3b812f1`
  - `ndp_full_update0_feature_extract.py`: `0101905c5c0ee97cf4ffaebbf8b4ca1f9a6335e2f938e12788bc2a94ae15a2d5`
  - `ndp_multi_pid_reference_graph.py`: `ff41caf58952f71ebe52e9b6ea5ada2293be489cf77a7ae3b9ddedd3fbd7b0d7`
  - `ndp_multi_pid_trajectory_eval.py`: `6a4cd151587372bf754f5eadf1eeb7b288fd2b204d58b781ba6bc02bed838101`
  - `ndp_reliability_vulnerability_graph_audit.py`: `4513a8e3e132a886114e27e0b69a7b39c14a7fe9447e9f9c6bffb530bd0ea9be`
  - `ndp_bidirectional_nonredundancy.py`: `4f507fa90f98b7843b445df21bfb2d3ec5debf58bea6777731d39303c23d9108`
  - `ndp_true_nonredundancy_experiment.py`: `53cf66515066819ebd5fc51f8565a1af3f2bdd6b3d0b0bd12bc1e51834848a68`
  - `ndp_reactive_protection_safety.py`: `5e6f3f8de5ceeab95a3baf0893ac3015ee68d1e198297260b6c6186916097b54`
  - `ndp_ppu_same_precision_kernel_probe.py`: `4483a6f5221fde70f0d21290f447a8c456b75cb9c55932c309aec246fa2f416a`

- 两种子 update0 checkpoint/特征指纹见 `full_features/seed1234/feature_manifest.json` 与 `full_features/seed4321/feature_manifest.json`；关系图 SHA256 见 `reference_graph/seed*/multi_pid_reference_summary.json`。
- 失败/限制：第一次轨迹评估因未传 `PPU_SDK/PPU_HOME` rc=134，原始日志保留；修复环境后两 seed rc=0。A/A 同精度仍 FAIL。
