# GPC-A4 reproducibility audit archive

审计 RUN_TIME：`20261019_130241`

参考仓库提交：`3dd98604a38075060d96aeb70506a977dd123be4`

本归档只包含可复现性诊断所需的源码、测试、报告、统计元数据和实验日志；不包含模型权重、Query/Gallery 数据集、数据图像或密钥。

## 运行范围

- 新结果：`20261016_161631`，B0/A4/R2 × seed1234/4321，30 epoch。
- 结果汇总修复：显式动态名单、checkpoint/config/protocol/query/gallery 指纹、严格 Query 配对和安全复用门禁。
- 离线统计：693 Query、1256 Gallery、11 个有效 PID；每个配对 2000 次固定种子 PID Bootstrap。
- 本归档未启动新的大规模训练，未运行官方 LTCC 测试。

## 目录

- `code/`：实际执行源码和评估修复源码。
- `reports/`：五份最终报告。
- `metadata/`：六个 checkpoint/config/manifest SHA256、原始指标、逐 PID 配对统计、旧新审计、源码逐文件对照。
- `logs/`：本轮实验与审计日志。

## 复核

使用 `/usr/local/bin/python3.10` 执行：

```bash
PYTHONPATH=/home/DIFFER/scripts:/data/DIFFER/scripts:/data/DIFFER \
/usr/local/bin/python3.10 code/home_DIFFER/scripts/test_gpc_v2_expanded_dev_eval.py
```

结果：`5/5 PASS`。评估和报告脚本只读取已有结果，不训练、不上传权重。

最终状态：`A4_MIXED`、`REPRODUCIBILITY_UNRESOLVED`、`MECHANISM_INCONCLUSIVE`；当前不允许启动下一轮训练。
