# NDP 下一阶段研究决策

## 状态

```text
FEATURE_EXPORT = PASS_FIXED_SUBSET_ONLY
PID_LEVEL_DECOMPOSITION = PASS_WITH_SEVERE_MULTI_PID_COVERAGE_LIMIT
HISTORICAL_RELIABILITY = PASS_PREFIX_ONLY
REACTIVE_NONREDUNDANCY = UNIT_TEST_AND_FROZEN_BOUNDARY_ONLY
PPU_SAME_PRECISION_AA = FAIL
NDP_MECHANISM = INCONCLUSIVE
CONTROLLED_SHORT_PILOT = NO_GO
```

## 已验证事实

1. 两个 seed 的 0/20/50/100/200/400 节点均可对固定 382/340 图像集合导出逐样本特征，且旧 LogMeanExp advantage 954/822 行逐条零误差复现。
2. 固定集合中 seed1234 没有真正的多负 PID anchor，seed4321 只有 1 个；因此本轮不能把多 PID 侵入机制当作已验证。
3. 历史前缀可靠性可以无未来泄漏地分层；200→400 的新增错误事件并未出现在 A 稳定类中，但样本量和抽样偏差不足以推广。
4. 反应式 NDP 与固定关系 margin hinge 的差异主要来自历史可靠性掩码、参考 margin 和关系集合，不是已证明的独立网络监督；T0/T1 完全相同，T4/T3 在当前集合也相同。
5. 同精度 PPU 反向重复仍有相对 L2 约 0.3%–0.47% 的主要参数梯度差异，虽余弦接近 1，但足以维持 A/A FAIL。没有进行 optimizer-step。

## 决策

当前不申请 NDP 短程训练，不启动 30/60 epoch。最小可保留方向是“历史前缀可靠性 + 固定多 PID/双侧关系的反应式诊断”，但它还没有超出 Hard Triplet 的因果独立价值证据。下一步优先级不是再加 Loss，而是：扩充预注册代表性多 PID 关系、获取逐步真实 Batch-Hard miner 覆盖记录、并用同上游梯度的 PPU 最小算子复现定位 A/A。

不要把本轮冻结特征梯度、自然轨迹关系退化或固定集合统计解释为 Rank-1/mAP收益。

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
