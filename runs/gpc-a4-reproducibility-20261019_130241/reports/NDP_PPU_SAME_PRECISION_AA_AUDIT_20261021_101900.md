# PPU 同精度 A/A 无 optimizer.step 审计

同一物化 batch、同一关系 hash、同一 checkpoint、AMP、同一 PPU 环境下，每个 seed 独立进程重复两次 forward/backward；四次探针均 `PASS`、`found_inf=0`、`optimizer_step_executed=false`，loss 和输入/关系 hash一致。

| seed | 最差抽样参数 | 重复反向相对L2差 | 余弦（float64） | 最大绝对差 |
|---:|---|---:|---:|---:|
|1234|blocks.0.attn.proj.weight|0.0041019|0.999991594012|0.0332031|
|4321|blocks.0.attn.proj.weight|0.00471182|0.999989054850|0.00561523|

差异不是 AMP skip 造成：同一 AMP 精度、无 step、found_inf=0 仍有梯度张量哈希差异。差异最大出现在本探针保存的早期 `blocks.0.attn.proj.weight`/`q_proj` 和 patch embedding，block22 q_proj 差异较小但仍非零。由于只保存了主要模块和输出梯度，没有对每个底层 kernel 建立同上游梯度的最小复现，因此根因仍不能定位到某一个 norm、Q/K/V、matmul 或 softmax kernel。

余弦实现已改为 float64 点积/范数并同时保存 `cosine_raw` 与有界 `cosine`；本轮超界问题不再由未经检查的 float32 统计产生。该结果仍不构成 A/A PASS：同精度反向不完全重复，尚未建立可接受的预注册噪声界。

旧的 block22 观察与本轮全模型重复探针并不矛盾：前者是另一套更窄的算子/模块观察；本轮证据只支持“全模型反向存在可重复的微小非确定性”，不支持精确根因定位。

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
