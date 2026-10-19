# GPC-A4 代码与可复现性报告

审计 RUN_TIME：`20261019_130241`；GitHub 参考提交：`3dd98604a38075060d96aeb70506a977dd123be4`。

## 结论

真实执行源码与 GitHub 参考提交已逐文件对照，机器可读结果见 `/home/DIFFER/outputs/gpc_a4_reproducibility_20261019_130241/gpc_a4_code_file_comparison.json`。当前 data 工作树是 dirty，不能把 GitHub main 当作实际训练源码。修复后的评估源码已保存并通过单元测试：`5/5 PASS`。

| 文件 | 对照状态 |
|---|---|
| `scripts/gpc_v2_expanded_dev_eval.py` | `DIFF`：当前版本加入显式名单、资产/配置/checkpoint 指纹、复用门禁、严格 Query 配对；检索和指标计算规则保持 |
| `scripts/gpc_a4_recheck_train.py` | `LOCAL_ONLY`：参考提交缺失 |
| `data/DIFFER/scripts/gpc_v2_train.py` | `LOCAL_ONLY`：旧运行实际源码，参考提交缺失 |
| Phase 7I / 6B / 5B | `MATCH` |
| sampler / loss | `MATCH` |
| `model/build.py` | `DIFF` |
| `processor/processor.py` | `LOCAL_ONLY` |
| scheduler / scheduler_factory | `MATCH` |

## 实际执行链

`gpc_a4_recheck_train.py → Phase 7I → Phase 6B → Phase 5B → loss/model/processor/solver`。基础身份训练是 CE + Triplet + Bio；总损失为 `base_total + lambda * relation`，本轮 `lambda=0.1`。基础 sampler 是 `DistributedRandomIdentitySampler`；A4 的 PID 选择发生在 wrapper 的全局 epoch 计划，不在 sampler 文件内。

当前 A4 使用 `cumulative_pid_balanced_selection`，B0 使用 Phase7I 的确定性 I3 selector，R2 使用随机 matched-budget selector。旧 A4 使用的是 batch-local `balanced_selection`，所以旧 A4 与新 A4 不是同一实现。

## 评估修复边界

对当前源码与参考提交的 diff 逐段检查后确认：原始模型 forward、L2 归一化、LTCC 同 PID 同 camera / 同 clothing 排除、Euclidean 检索排序、逐 Query AP/Rank-1、PID 聚合均保留；修复只增加协议/Query/Gallery/配置/权重指纹、显式 B0/A4/R2×seed 名单、缺失文件报错、严格配对清单检查和安全复用门禁。

## 归档

已归档修复源码、旧/新训练 wrapper、Phase 7I/6B/5B、sampler、loss、测试和 JSON/CSV 小型报告；归档不包含密钥、模型权重或数据集。六个新 checkpoint 及其 SHA256 只写入本地审计元数据，不上传权重。
