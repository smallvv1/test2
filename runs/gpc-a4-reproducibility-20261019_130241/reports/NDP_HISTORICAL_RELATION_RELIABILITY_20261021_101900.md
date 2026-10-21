# NDP 历史关系可靠性审计

可靠性分层只使用节点 `0..t` 的历史前缀，未来节点只用于离线标签。预注册阈值为：A 类所有历史 margin>0、最小 margin≥0.05 且历史标准差≤0.05；B 类历史均正确但接近边界或波动较大；C 类历史已出现错误。该分层不修改 GPC。

| seed | A 稳定正确行 | B 正确但近/波动行 | C 历史不可靠行 |
|---:|---:|---:|---:|
|1234|552|88|314|
|4321|354|148|326|

在 200→400 的新增错误事件中，seed1234 的 A/B/C 分别为 0/0/5；seed4321 分别为 0/2/1。这个结果支持“单次当前正确”不等于稳定可靠，也没有支持 A 类稳定关系在本小样本中发生后续新增侵入。它只是观察性分层，样本仍是固定困难集合，不足以成为训练门控。

逐行结果：`historical_relation_rows.csv`；摘要：`historical_relation_summary.json`。

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
