# P0：A/A 数值重复性审计

运行：`20261021_082715`。本阶段没有执行 optimizer.step，也没有启动 NDP 训练。

## 实际证据

复核了：

- `/home/DIFFER/reports/NDP_AA_OPERATOR_LEVEL_AUDIT_20261021_034200.md`
- `/home/DIFFER/reports/NDP_AA_OPERATOR_RECHECK_20261021_041500.md`
- `/home/DIFFER/reports/NDP_AA_REPRODUCIBILITY_20261021_031415.md`
- `/home/DIFFER/outputs/ndp_aa_operator_audit_20261021_034000/`
- `/home/DIFFER/outputs/ndp_aa_operator_audit_20261021_041000/`

本轮重新运行了无更新的梯度张量比较：

```text
/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_aa_operator_analysis.py \
  --root /home/DIFFER/outputs/ndp_aa_operator_audit_20261021_034000 \
  --seed 1234 --out outputs/ndp_second_innovation_20261021_082715/p0_aa/seed1234.json
/usr/local/bin/python3.10 -u /home/DIFFER/scripts/ndp_aa_operator_analysis.py \
  --root /home/DIFFER/outputs/ndp_aa_operator_audit_20261021_041000 \
  --seed 4321 --out outputs/ndp_second_innovation_20261021_082715/p0_aa/seed4321.json
```

## 结果

- 两个 seed 第 1 步 forward/loss 一致，`found_inf=0`，没有 AMP skip。
- 第 1 步未缩放梯度已经不同，第 2 步 forward 开始分叉。
- 首个可观测差异位于 `blocks.22.norm1` 和 `blocks.22.attn.q_proj` 反向输出；block22 顶层梯度仍相同。
- 差异继续向早期 block 放大；尚未定位到 norm、Q/K/V、attention matmul 或归约中的单个底层 PPU kernel。
- DDP all-reduce 不是主要方向性根因，rank-local 梯度在归约前已经不同。

因此：`A/A_REPRODUCIBILITY=FAIL`。现有证据不支持把 NDP 小幅差异解释为 optimizer-step 因果效果。

模型相关历史探针使用 `/usr/local/bin/python3.10`、torch 2.6.0 和 PPU-ZW810E；本轮 P1–P3 只处理已落盘的 train-only 特征，不存在 CPU 模型 fallback。

证据 JSON/CSV：

`/home/DIFFER/outputs/ndp_second_innovation_20261021_082715/p0_aa/`
