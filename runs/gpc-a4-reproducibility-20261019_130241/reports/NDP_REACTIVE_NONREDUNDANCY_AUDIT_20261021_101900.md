# NDP 反应式候选非冗余性审计

本审计使用固定特征边界的 200→400 转移，不执行网络 optimizer.step。T0/T1 是固定关系 proxy：没有逐步真实 Batch-Hard miner 记录，不能把它们写成完整 Batch-Hard 训练梯度。

| seed | 候选 | 关系数 | 激活数 | loss | 与T0梯度余弦 |
|---:|---|---:|---:|---:|---:|
|1234|T0_HARD_TRIPLET_FIXED_RELATION_PROXY|159|131|0.212272|1.0000000000000002|
|1234|T1_FIXED_MARGIN_HINGE|159|131|0.212272|1.0000000000000002|
|1234|T2_REFERENCE_ADAPTIVE_HINGE|159|116|0.0186043|0.7138681891806675|
|1234|T3_REACTIVE_STABLE_HISTORY|92|78|0.0267312|0.49463751899813097|
|1234|T4_DIRECTED_ALL_RELIABLE|92|78|0.0267312|0.49463751899813097|
|1234|T5_MULTI_PID_ALL_RELATIONS|159|116|0.0186043|0.7138681891806675|
|4321|T0_HARD_TRIPLET_FIXED_RELATION_PROXY|138|134|0.264505|0.9999999999999989|
|4321|T1_FIXED_MARGIN_HINGE|138|134|0.264505|0.9999999999999989|
|4321|T2_REFERENCE_ADAPTIVE_HINGE|138|64|0.00324151|0.669498029825554|
|4321|T3_REACTIVE_STABLE_HISTORY|59|35|0.00446493|0.48186736976776917|
|4321|T4_DIRECTED_ALL_RELIABLE|59|35|0.00446493|0.48186736976776917|
|4321|T5_MULTI_PID_ALL_RELATIONS|138|64|0.00324151|0.669498029825554|

T0 与 T1 在固定关系集合上完全相同，证明固定-margin hinge 与该离线 proxy 不提供额外信息。T2 是参考 margin 自适应 hinge，T3 是历史 A 类关系的单侧版本；两者相对 T0 的梯度余弦约为 seed1234 的 0.714/0.495、seed4321 的 0.669/0.482，但这只是冻结边界张量梯度，不是 EVA02 参数更新因果证据。当前实现的 T4 在该数据中与 T3 关系集合相同，T5 仅因纳入所有固定负 PID关系而扩大集合；不能据此宣布双侧或多 PID的独立算法增益。

`reactive_boundary_loss` 已修复全 NaN/Inf 行通过 `sum()*0` 产生 NaN 的问题；单元测试验证参考 detach、改善不惩罚、恶化有负方向梯度、空关系和非有限输入均安全。

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
