# NDP 关系覆盖与抽样偏差审计

| seed | anchor | PID | image | 平均负PID/anchor | 最大负PID | 多负PID anchor | 有跨衣正样本 | 有跨相机正样本 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|159|30|382|1.000|1|0|159|122|
|4321|137|31|340|1.007|2|1|137|101|

seed1234 的固定集合每个 anchor 只有 1 个负 PID；seed4321 只有 1 个 anchor 有 2 个负 PID。因此本轮虽然已正确保存逐样本特征并可按 PID 分解，但“多身份”覆盖仍严重不足。当前集合是历史困难/固定关系集合，代表性按 PID×衣服×相机的独立分层集合不存在；结果不能外推全量 LTCC，也不能把困难集合的事件率当作总体事件率。

下一次如继续，必须在不使用未来节点的前提下预注册一个代表性集合，并与困难集合分开报告。

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
