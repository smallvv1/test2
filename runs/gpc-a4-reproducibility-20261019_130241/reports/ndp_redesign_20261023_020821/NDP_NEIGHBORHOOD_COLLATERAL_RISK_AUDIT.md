# P4 邻域附带损害与双侧风险审计

目标关系为新定义的 R∩V；每个 anchor 同时展开五个不同负 PID，方向 A→B 与 B→A 使用各自合法正样本。`predicted_delta_margin=-eta<grad(M),grad(L)>`，eta=1e-3；每个候选 loss 的梯度先归一到相同 L2 budget=1，因此这是统一预算的一阶方向诊断，不是网络更新。

| seed | method | rows | target ΔM均值 | 其他 ΔM均值 | 其他关系恶化数 | 预测跨零数 | finite |
|---|---|---|---|---|---|---|---|
| 1234 | bidirectional_safe | 3050 | 1.3571777867598718e-06 | 5.525269462120617e-06 | 17 | 0 | True |
| 1234 | margin_hinge_target | 3050 | 9.892894330264288e-07 | 3.4523341842453126e-07 | 1 | 0 | True |
| 1234 | multi_pid_safe | 3050 | 1.260710759297443e-05 | 1.36890720432146e-05 | 8 | 0 | True |
| 1234 | safe_boundary_target | 3050 | 1.328544075559793e-06 | 4.495167897595851e-07 | 2 | 0 | True |
| 1234 | triplet_batchhard | 3050 | 9.238740467711354e-06 | 1.0252359923436595e-05 | 9 | 0 | True |
| 4321 | bidirectional_safe | 2996 | 7.336026474133911e-07 | 5.600352599990377e-06 | 34 | 0 | True |
| 4321 | margin_hinge_target | 2996 | 2.4296896720734786e-07 | 8.153515325087221e-08 | 0 | 0 | True |
| 4321 | multi_pid_safe | 2996 | 1.1301184862616232e-05 | 1.2435087168680618e-05 | 5 | 0 | True |
| 4321 | safe_boundary_target | 2996 | 3.1358703097119984e-07 | 9.676603857255233e-08 | 0 | 0 | True |
| 4321 | triplet_batchhard | 2996 | 9.486931569974949e-06 | 1.0149189883870633e-05 | 11 | 0 | True |

`triplet_batchhard` 使用原始 raw feat[0] 的欧氏 Batch-Hard 局部代理；`margin_hinge_target` 是单侧固定阈值；`safe_boundary_target` 是历史边界；`multi_pid_safe` 同 anchor 五个 PID；`bidirectional_safe` 加入可匹配的反向身份关系。若目标改善同时伴随其他正确关系负 ΔM 或跨零，说明单侧保护存在 collateral risk。有限差分复核保存在每个 seed 的 JSON。

- 运行命令：`/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_reliability_safe_boundary_audit.py --run-id 20261023_020821 --output-root /home/DIFFER/outputs/ndp_redesign_20261023_020821 --report-dir /home/DIFFER/reports/ndp_redesign_20261023_020821 --remote-main d6e24867a9bf33423d3342682c96d053d37ff190`
- 远端 main：`d6e24867a9bf33423d3342682c96d053d37ff190`；本地工作树无 `.git`，因此不伪造本地 commit。
- 独立研究代码 SHA256：`f410368ae4a4fb2ed3fe4ce2a891eee4b3516c40ef40c66df8cc96550c55f463`。
- 数据来源和 SHA256：见 `RUN_MANIFEST.json`；固定 checkpoint SHA256：见已有 `outputs/ndp_triplet_alignment_20261021_134148/RUN_MANIFEST.json`。
- 协议：只读 train-only 特征；update400 只作事后标签；无 optimizer.step、无官方 query/gallery。
