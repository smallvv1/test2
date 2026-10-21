# NDP 多 PID 局部身份侵入审计

## 结果

脚本 `ndp_pid_level_intrusion_audit.py` 固定使用启动前 groups；同一负 PID 的多张图片先按 `max similarity` 得到一个 `v_ic`，不会把多张图片重复计成多个身份。`M_ic=Q25(P_i)-max_c(sim(i,N_c))`，新增错误 PID 严格定义为 `M_ic(t)>0` 且 `M_ic(t+1)<=0`。

| seed | anchor | 固定负 PID关系 | 关系行 | 正确→错误 | 错误→正确 | 正确→正确 | 错误→错误 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|159|38|795|9|10|544|232|
|4321|137|37|690|3|2|420|265|

| seed | 区间 | 关系数 | 新增错误 PID | 恢复正确 PID |
|---:|---|---:|---:|---:|
|1234|0->20|159|2|3|
|1234|100->200|159|0|3|
|1234|20->50|159|1|2|
|1234|200->400|159|5|2|
|1234|50->100|159|1|0|
|4321|0->20|138|0|0|
|4321|100->200|138|0|1|
|4321|20->50|138|0|0|
|4321|200->400|138|3|0|
|4321|50->100|138|0|1|

seed1234 的 159 anchor 实际每个只有 1 个负 PID，不能声称该 seed 已完成多身份邻域风险的充分检验；seed4321 只有 1 个 anchor 含 2 个负 PID，其余也基本是单负 PID。因此当前结果能证明“按 PID 去重的固定关系统计实现正确”，不能证明多 PID 软侵入在大覆盖上成立。

跨衣/摄像头字段写入逐关系 CSV：`positive_cross_clothes`、`negative_cross_clothes_count`、`positive_cross_camera_count`、`negative_cross_camera_count`。本轮不把这些分组差异解释为因果训练效果。

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
