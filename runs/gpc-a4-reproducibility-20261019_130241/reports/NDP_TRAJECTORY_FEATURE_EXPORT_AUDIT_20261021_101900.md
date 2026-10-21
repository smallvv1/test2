# NDP 同轨迹逐节点特征导出审计

## 结论

两个 seed 的固定诊断集合均已从同一条 Bio+GPC continuation 的 update 0/20/50/100/200/400 节点导出逐图像 FP32 归一化 `feat[0]`。这是“扩大固定集合”的真实逐样本特征，不是全量 9576 张图像特征库。

| seed | 固定 anchor | 唯一图像 | 节点 | NPZ | 旧 advantage 行数 | 最大/均值/P95复算误差 |
|---:|---:|---:|---|---|---:|---:|
|1234|159|382|[0, 20, 50, 100, 200, 400]|`/home/DIFFER/outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed1234/trajectory_features_seed1234.npz`|954|0.0/0.0/0.0|
|4321|137|340|[0, 20, 50, 100, 200, 400]|`/home/DIFFER/outputs/ndp_multipid_reconstruction_20261021_101900/feature_export/seed4321/trajectory_features_seed4321.npz`|822|0.0/0.0/0.0|

两组复算均 `rows_unmatched=0`，误差全为 0，说明旧的 LogMeanExp advantage 公式和新导出特征可逐条对齐。update0 重复编码误差也为 0。首轮 seed1234 曾在复算函数漏调用 `act.advantage` 时退出码 1，真实 traceback 保存在 `log_NDP/ndp_multipid_reconstruction_20261021_101900/feature_export_seed1234_retry0_traceback.log`；修复后只重跑该导出任务。

## 资产与边界

节点参数 SHA256、groups SHA256、图像索引 SHA256 写入各 seed 的 `feature_export_manifest.json`。来源 continuation manifest 标记 `gpc_v1_preserved=true`、`optimizer_step_count=400`、`cpu_model_fallback=false`。文件名 `gpc_v2_train.py` 只是真实训练引擎文件名，不能单独作为方法版本判断；本轮沿用 B0/GPC-v1 continuation 的 manifest 证据。

没有导出全量训练集，因此不能把本结果外推为全量局部邻域覆盖；但已满足本轮固定 159/137 anchor 集合的逐节点分解要求。

本报告属于 RUN_TIME `20261021_101900`，所有模型计算仅使用 PPU-ZW810E 的 `eval()+no_grad` 或无 optimizer.step 反向探针；未读取官方 query/gallery/test，未修改 GPC-v1。

实际远端 main：`ef18089486f340e11dfc52949e5bc8b6bc8d58d2`；服务器 `/data/DIFFER` 本地 HEAD：`573b0502a231568d509fe3a463664c1a3c42e258`，且工作树原有 dirty，故本轮新增代码只在 `/home/DIFFER` 保存，未将 dirty 工作树误当作远端版本。

固定基线文件与新增代码 SHA256：

```json
{
  "baseline": {
    "/data/DIFFER/scripts/gpc_v2_train.py": "d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5",
    "/data/DIFFER/loss/make_loss.py": "9be543c08b50553a95839351929fa9341edf900947593fbd675c0e4847889043",
    "/data/DIFFER/configs/ltcc/eva02_l_bio.yml": "c0589950b23885c334c64c85bd6db9803935765d0647e02c1bb74b5b5ae00b97"
  },
  "new_code": {
    "scripts/ndp_trajectory_feature_export.py": "f9eb73a76e860e2f30ec7f627d77f3ad3452eaf11602a8533686cefa53b525db",
    "scripts/ndp_pid_level_intrusion_audit.py": "8cee06251722c1aeacbce3721e4f010a09e69452441f231550b656e749e5450d",
    "scripts/ndp_historical_relation_reliability.py": "84efba18e431abe12b823ce6aaad5fe1bd8b9947bdb548bd138a30c12fb9ba31",
    "scripts/ndp_reactive_nonredundancy_audit.py": "1f7117408e77560d6d328b05b29c343556daec10473ec58c7c6f6d795a9608ac",
    "scripts/ndp_relation_coverage_bias_audit.py": "52223f353c59bcb2b16b3896ab0ab8ed8cd497b64fcfb77f7fddce0f1de04f14",
    "scripts/ndp_reactive_prototype.py": "0374cd3e35df73e3b1326375d7f90cee0d883b515b7c3f43cbc7742edbafdf32",
    "scripts/ndp_aa_operator_probe.py": "7e4496cfbea93468094c13b91896bd1eb14c8748f79c573da2cc2d92f023183d",
    "scripts/ndp_aa_operator_analysis.py": "f64410985f164c8465bed896730fa08c77ca682f73d1f970e339e51587117e73",
    "tests/test_ndp_reactive_prototype.py": "4011f878aaaeb9cadeaee26cdd3cb77585bf4fa2a4ffc9935dfa3e29b49593d6",
    "tests/test_ndp_aa_operator_analysis.py": "33d5822b9868759da1550ec38d19ad4993d5c6cb7282f4a83c20c525d9837c6d"
  }
}
```
