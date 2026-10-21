# NDP 下一阶段决策

## 本轮实际完成

- 新增并运行 `ndp_future_boundary_risk.py`：重构持续风险、首次触发、新增退化、真实翻转和四状态标签。
- 新增并运行 `ndp_bidirectional_boundary_audit.py`：保留双侧有向 identity boundary。
- 新增并运行 `ndp_risk_predictor_comparison.py`：比较当前 margin、历史 drop、历史斜率、静态 Hubness 和双侧分数。
- 新增并运行 `ndp_future_boundary_cluster_bootstrap.py`：2000次 PID cluster bootstrap。
- 新增 `tests/test_ndp_future_boundary.py`；已有风险模块测试继续 PASS。
- 没有执行模型前向、反向、参数更新或正式训练；GPC-v1未修改。

源码快照：`/home/DIFFER/outputs/ndp_future_boundary_20261021_094250/source_snapshot/`。结构化结果：`/home/DIFFER/outputs/ndp_future_boundary_20261021_094250/artifacts/`。

## 科学结论

1. 之前较高的累计风险 AUC（0.7263/0.7880）明显包含风险持续性；首次触发仅0.5445/0.6724，新增退化0.3667/0.4693。
2. 真正的正确→错误翻转只有6/211与2/201，且跨 seed 区间模式不同。`-M_prev` 的高 AUC是边界邻近性，不是未来更新风险的因果预测。
3. 双侧身份边界存在一侧改善、另一侧恶化，但仅覆盖7个 image-pair/seed；这支持双侧诊断必要性，不支持双侧损失已有效。
4. 当前历史斜率、固定样本 Hubness没有稳定增量；同轨迹完整邻域稳定性和高分冒名者尾部因缺少节点级特征未运行。
5. 反应式局部保护比预测式门控更符合现有证据，但其损失仍需解决与 Hard Triplet/DBP 的重叠、参考正确性和 A/A 数值门禁，暂不接入训练。

## 门禁

| 项目 | 状态 |
|---|---|
| A/A 数值可重复性 | `FAIL` |
| 新增风险预测 | `INCONCLUSIVE / NOT_SUPPORTED_BY_CURRENT_DATA` |
| 真实身份翻转跨 seed 稳定性 | `NOT_ESTABLISHED` |
| 双侧边界诊断 | `PASS_WITH_LIMITED_COVERAGE` |
| 反应式 NDP 训练 | `NO_GO` |
| 30/60 epoch | `NO_GO` |

## 下一步

不启动任何 NDP optimizer-step、30 epoch 或60 epoch。若继续投入，最小且可证伪的顺序是：先完成 PPU block22 反向 kernel 隔离；再扩大同轨迹固定关系覆盖；预注册反应式单侧保护与普通 Hard Triplet 的相同关系对照。预测式风险加权本轮停止，不再堆叠 B3–B7。

最终状态：`INCONCLUSIVE`；`CONTROLLED_SHORT_PILOT=NO_GO`。

