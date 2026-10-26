# NDP-Floor0 PRCC 对照实验结果

## 结论

NF0 已完成两个 seed 的 5 raw continuation epoch 训练和 PRCC C→A 固定节点评估。Floor0 显著提高了 NDP 触发率，但没有在两个 seed、两个节点上同时改善 Rank-1 和 mAP：

- Seed1234：第3节点相对 B0 提升， 第5节点基本回到 B0 以下。
- Seed4321：Rank-1 两个节点均低于 B0，mAP 第3节点略高、第5节点明显低于 B0。

因此 **NDP-Floor0 暂不具备稳定性能增量，不建议继续长程训练或继续无目的调 epsilon**。

## 实验身份

- 远端 `test2/main` 核对：`f244fb92c3be8746bd5460d936f8ad0bfaf00e94`。
- 固定 GPC pair run-id：`ndp_prcc_baseline_fix_20261025_130457`。
- 新日志：`/home/DIFFER/log_NDP/ndp_floor0_prcc_20261026_013302/`。
- 当前 runner SHA256：`c8f3a67139a83d93f7f185ded0d23b738e3e055a30f274a2ca5cb70b2980dc93`。
- GPC 源 SHA256：`d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5`，未修改。
- 单元测试 SHA256：`76622cd4b89a9b43a56630a337381e40c4f1776905f351a6d76feb97cad91eb4`。

GPC、EVA02、CE、Triplet、clipBio、sampler、PRCC 评估协议和 `lambda_ndp=0.05` 均未改动。NF0 只把 NDP-V5 的目标 margin 改为：

```text
M_target = max(M_ref - 0.02, 0)
L = ReLU(M_target - M_current)
```

仍只使用历史 `M_ref > 0` 的关系，参考特征停止梯度，当前 anchor 保留反向路径。

## 实现与测试

新增独立 arm `NF0`：

- `N5`：`M_ref - epsilon`。
- `NF0`：`max(M_ref - epsilon, 0)`。
- `B0`：完全跳过参考图和 NDP 关系计算，`total == main_total`。
- NF0/N5 均使用 `main_total + 0.05 * ndp_loss`。

验证命令：

```bash
python3 -m py_compile scripts/ndp_v5_prcc_runner.py
PYTHONPATH=/home/DIFFER/scripts:/data/DIFFER/scripts:/data/DIFFER \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/test_ndp_v5_b0_loss_contract.py
```

结果：`4 passed`，编译通过。

两个 seed 的 20 步 NF0 smoke 均 PASS：

- NDP Loss 有限。
- NDP feature gradient 非零。
- NDP probe parameter gradient 非零。
- 20/20 optimizer steps 成功。
- 无 NaN/Inf、无 AMP skip。

Seed1234 smoke：selected 1271，active 约 27，active batch rate 25%。Seed4321 smoke 也产生非零 NDP Loss 和梯度。

## 恢复状态限制

指定的 B0 节点实际只有 `model.pth`，没有 `rank0_state.pt`/`rank1_state.pt`，因此不存在可验证的 optimizer、scheduler、GradScaler、RNG 完整恢复状态。第一次按完整 rank state 恢复的 smoke 因此被正确拒绝并保留在 `nf0_seed1234_smoke/`，没有伪造成功。

正式 NF0 使用：

- 相同修复后 B0 的模型权重作为共同初始模型。
- `--resume-epoch 3 --resume-global-step 1653`。
- fresh optimizer/scheduler/GradScaler/RNG。
- 每个 seed 的训练特征和 K=5 图保持 seed-specific。

该限制由 `model_only_warm_start.json` 记录。因此 NF0 与此前拥有完整 continuation state 的 B0/N5 结果不是严格的 optimizer/RNG 无缝配对，只能作为“共同 B0 模型权重、相同 raw continuation 预算”的探索性比较。

## B0 / N5 / NF0 PRCC C→A 结果

精确值来自既有正确 B0/N5 日志和本轮 NF0 `metrics_per_epoch.csv`。mINP 未被仓库评估器暴露，本轮不填充伪造值。

### Seed 1234

| continuation raw epoch | 模型 | Rank-1 | mAP | 相对 B0 Rank-1 | 相对 B0 mAP |
|---:|---|---:|---:|---:|---:|
| 3 | B0 | 63.3644% | 58.3306% | — | — |
| 3 | N5 | 62.6588% | 57.7206% | -0.7056 pp | -0.6100 pp |
| 3 | NF0 | 63.6466% | 58.4480% | +0.2822 pp | +0.1173 pp |
| 5 | B0 | 62.9410% | 59.1704% | — | — |
| 5 | N5 | 61.9249% | 58.6693% | -1.0161 pp | -0.5011 pp |
| 5 | NF0 | 62.8846% | 59.0592% | -0.0564 pp | -0.1112 pp |

### Seed 4321

| continuation raw epoch | 模型 | Rank-1 | mAP | 相对 B0 Rank-1 | 相对 B0 mAP |
|---:|---|---:|---:|---:|---:|
| 3 | B0 | 63.2797% | 57.8626% | — | — |
| 3 | N5 | 63.5620% | 58.0502% | +0.2822 pp | +0.1877 pp |
| 3 | NF0 | 62.6305% | 58.0465% | -0.6492 pp | +0.1840 pp |
| 5 | B0 | 63.7313% | 59.6883% | — | — |
| 5 | N5 | 63.8442% | 59.6111% | +0.1129 pp | -0.0772 pp |
| 5 | NF0 | 63.4208% | 58.9600% | -0.3105 pp | -0.7283 pp |

## NF0 训练侧统计

“当前错误关系”是历史正确关系中 `M_current <= 0` 的数量，可作为 margin 过零关系计数；它不是测试集错误 PID。

| Seed | 节点 | active batch rate | selected relation | active relation | active/selected | 当前错误关系 | NDP mean loss | mean margin | Purity@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 3 | 91.50% | 108925 | 14554 | 13.36% | 14287 | 9.8405e-4 | 0.010362 | 0.306549 |
| 1234 | 5 | 88.11% | 72592 | 8122 | 11.19% | 7974 | 8.5540e-4 | 0.012480 | 0.306624 |
| 4321 | 3 | 92.50% | 109678 | 14430 | 13.16% | 14208 | 1.0443e-3 | 0.010821 | 0.305169 |
| 4321 | 5 | 88.57% | 72969 | 8021 | 10.99% | 7865 | 8.8722e-4 | 0.013090 | 0.305198 |

相较此前 N5 约 1.6%～1.9% 的 active/selected，NF0 提升到约 11.0%～13.4%，说明 Floor0 的确增加了对接近或已经越过零边界的历史正确关系的约束。但更高激活率没有转化为双 seed 稳定收益。

前 20 个 diagnostic steps 的梯度：

- Seed1234：NDP feature gradient max `2.6223e-3`，mean `1.1159e-3`；NDP probe parameter gradient max `1.4607e-2`。
- Seed4321：NDP feature gradient max `5.6696e-3`，mean `1.3110e-3`；NDP probe parameter gradient max `1.9932e-2`。
- 两 seed 正式训练均无 NaN；Seed1234 有 1 个 AMP skip（2754/2755 effective updates），Seed4321 无 skip。

## Checkpoint

- Seed1234 第3节点：`nf0_seed1234_5raw/node_rawepoch03_update1653/model.pth`，SHA256 `9e1b281627e63c50073a69cd4f6f0190b23457e821b60af631a069f312e64c5c`。
- Seed1234 第5节点：`nf0_seed1234_5raw/node_rawepoch05_update2754/model.pth`，SHA256 `1faded45e89bbfdbd6d75a2abaf102d6b94eebb75a3f594cfdc45ce2ebef6951`。
- Seed4321 第3节点：`nf0_seed4321_5raw/node_rawepoch03_update1653/model.pth`，SHA256 `2b4794114ebfe3b37186fc802914eed79a729548393433700826fea9f76319af`。
- Seed4321 第5节点：`nf0_seed4321_5raw/node_rawepoch05_update2755/model.pth`，SHA256 `062210c2ff848aaf89628fe9369d8c8d57b6dbe2a3d633b4919a7eeb59882c6d`。

两个正式 run summary 均为 `PASS`；Seed1234 有 1 个 AMP skip，Seed4321 为 0 个。

## 决策

**NO-GO for longer Floor0 training.**

Floor0 的唯一明确正向证据是激活覆盖显著增加，且 NDP 梯度确实参与训练；性能证据不满足继续扩展门槛。下一步不应继续堆叠 NDP loss 或扫描 epsilon。若仍要推进，优先补做严格保存的 B0/NF0 完整 optimizer-state 配对和前100个 batch 的逐关系 GPC ID 校验，确认模型权重 warm-start 与 fresh optimizer 造成的混杂后，再决定是否保留 NF0。

本轮不进入 LTCC、独立 seed 或 60 epoch；PPU 已知 A/A 非确定性也使零点几 pp 的变化不能视为可靠收益。
