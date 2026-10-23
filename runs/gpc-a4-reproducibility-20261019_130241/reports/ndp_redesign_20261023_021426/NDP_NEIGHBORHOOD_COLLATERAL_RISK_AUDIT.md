# P4 邻域附带损害与双侧风险审计

目标关系为新定义的 R∩V；每个 anchor 同时展开五个不同负 PID，方向 A→B 与 B→A 使用各自合法正样本。`predicted_delta_margin=-eta<grad(M),grad(L)>`，eta=1e-3；每个候选 loss 的梯度先归一到相同 L2 budget=1，因此这是统一预算的一阶方向诊断，不是网络更新。

| seed | method | rows | target ΔM均值 | 其他 ΔM均值 | 其他关系恶化数 | 预测跨零数 | finite |
|---|---|---|---|---|---|---|---|
| 1234 | bidirectional_safe | 3688 | 1.2536758732198495e-05 | 9.154432862622618e-06 | 32 | 0 | True |
| 1234 | margin_hinge_target | 3688 | 7.164483741265714e-06 | 2.466080495844317e-06 | 9 | 0 | True |
| 1234 | multi_pid_safe | 3688 | 1.9829636035505604e-05 | 1.808340565467975e-05 | 22 | 0 | True |
| 1234 | safe_boundary_target | 3688 | 1.6160412565934775e-05 | 5.176950579162796e-06 | 15 | 0 | True |
| 1234 | triplet_batchhard | 3688 | 9.367943905725219e-06 | 1.0371090687302915e-05 | 16 | 0 | True |
| 4321 | bidirectional_safe | 3593 | 1.1609841606477768e-05 | 8.905198195438665e-06 | 26 | 0 | True |
| 4321 | margin_hinge_target | 3593 | 6.622850257657697e-06 | 2.138438777611562e-06 | 7 | 0 | True |
| 4321 | multi_pid_safe | 3593 | 1.8329611864675568e-05 | 1.659690945849647e-05 | 11 | 0 | True |
| 4321 | safe_boundary_target | 3593 | 1.4307861261152283e-05 | 4.397066417635499e-06 | 9 | 0 | True |
| 4321 | triplet_batchhard | 3593 | 9.483384381976436e-06 | 1.001261536869543e-05 | 15 | 0 | True |

`triplet_batchhard` 使用原始 raw feat[0] 的欧氏 Batch-Hard 局部代理；`margin_hinge_target` 是单侧固定阈值；`safe_boundary_target` 是历史边界；`multi_pid_safe` 同 anchor 五个 PID；`bidirectional_safe` 加入可匹配的反向身份关系。若目标改善同时伴随其他正确关系负 ΔM 或跨零，说明单侧保护存在 collateral risk。有限差分复核保存在每个 seed 的 JSON。

统计不确定性：P4 以固定 R∩V 目标和固定邻域全集计算，伤害计数是全集计数；eta=1e-3 仅为统一诊断步长，没有把它外推为训练分布置信区间。

- 运行命令：`/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_021426 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_021426 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_021426 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，因此不伪造本地 commit。
- 独立研究代码 SHA256：`25ab1256933cf6cb6c0037834da08915b805aba68125b6f75147d106aadea6b8`。
- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。
- 明确 checkpoint SHA256：seed1234=`e11c90b63d7252f0fde99439bb1d5a6f1d72e757d43143fe9ac92061e2411177`；seed4321=`d69859975ae386f82bebd05d944633e962cc377f73f55599eb685128a09637a7`。
- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。
