# NDP-V6 阶段性训练实验结果

实验目录：`/home/DIFFER/log_NDP/ndp_v6_20261023_131720/`
用户指定 GitHub 提交：`6d1852537e130a3109fc83f05ebfb6ac7dd2802a`
固定 DIFFER 工程：`/data/DIFFER` HEAD = `573b0502a231568d509fe3a463664c1a3c42e258`

## 结论

NDP-V6 的阶段策略已真实完成双 seed、5960 有效更新：前 3 epoch 复用 N5，后 7 epoch 使用 `lambda_ndp=0` 的 DIFFER+GPC。

- N6 在第 5 epoch 两个 seed 的 Rank-1、mAP、mINP 都不低于 B0，短程结果正向。
- N6 到第 10 epoch 时，Seed1234 仍优于 B0，但 Seed4321 Rank-1 从 52.81% 降至 49.74%，低于 B0 的 51.53%。
- 关闭 NDP 后不能保证长程稳定性；两 seed 的第 10 epoch Rank-1 并未同时优于 B0。
- 不继续尝试更多 NDP 开关时间，也不跑 60 epoch。当前候选仍保留 N5 的 3 epoch 节点；N6 的 5 epoch 作为短程观察结果，需独立 seed/PRCC 才能确认。

结论标签：`NDP-V6_10EPOCH_NO_GO`；`NDP-V6_5EPOCH_PROMISING_BUT_UNCONFIRMED`；`NDP_MECHANISM = INCONCLUSIVE`。

## 固定策略与实现

基础损失始终为：

`L_base = CE + Triplet + clipBio + 0.1 * GPC`

N6 使用原有 NDP-V5 逐负 PID Margin，但预注册阶段调度为：

`effective_updates < 1788: lambda_ndp=0.05`
`effective_updates >= 1788: lambda_ndp=0`

本轮从 N5 的 update1788 checkpoint 继续，因此续训的每一步均为 `lambda_ndp=0`；前三 epoch 不重复训练，而是直接复用 N5 已完成状态。GPC 正样本选择、GPC Loss、系数 0.1、EVA02、CE、Triplet、Bio、sampler 和评估协议均未修改。

最小代码变更：

- runner 新增 `N6` arm 和固定 update1788 阶段切换。
- train-only 特征导出脚本仅扩展 `--arm N6` 元数据枚举，特征计算逻辑不变。
- 单元测试：`2 passed`；N6 schedule test：`N6_SCHEDULE_TEST_PASS`。

代码 SHA256：

| 文件 | SHA256 |
|---|---|
| `scripts/ndp_e60_minimal_pair_runner.py` | `e37986bcd11339ada55cbb31a6c086aeec32cbf11d9650ecd72e4121526d0810` |
| `scripts/ndp_v2_feature_export.py` | `afa6f5b8150859b2f8a9ea2d35e7910dabed2a7029b3a6ceb982bed9f7171100` |
| `scripts/ndp_v5_local_metrics.py` | `8341795a7a42bd0f9ad948c010d0aee69ec63b68cd10739c41c10662b6cb7b46` |
| 固定 GPC `/data/DIFFER/scripts/gpc_v2_train.py` | `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5` |
| LTCC config | `c7c9e967a5ae824d9bfad9fb27ea77bd1af645e4de009c18111609523d8d33e6` |

## 真实执行命令

恢复状态检查使用：

```bash
/usr/local/bin/python3.10 - <<'PY'
import torch
from pathlib import Path
for p in [
  Path('/home/DIFFER/log_NDP/ndp_v5_20261023_100048/n5_seed1234/node_update1788'),
  Path('/home/DIFFER/log_NDP/ndp_v5_20261023_100048/n5_seed4321/node_update1788'),
]:
  for rank in (0, 1):
    state = torch.load(p / f'rank{rank}_state.pt', map_location='cpu', weights_only=False)
    print(p, rank, sorted(state))
PY
```

实际训练使用：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode full --arm N6 \
  --seed 1234|4321 --max-effective-updates 5960 \
  --save-updates 2980,5960 --run-id <seed-specific-run-id> \
  --run-dir /home/DIFFER/log_NDP/ndp_v6_20261023_131720/n6_seed<seed> \
  --reference-feature /home/DIFFER/outputs/ndp_true_multipid_20261021_103952/full_features/seed<seed>/update0_full_features_seed<seed>.npz \
  --graph /home/DIFFER/outputs/ndp_v2_20261023_061358/reference_graph/seed<seed>/multi_pid_reference_graph_v2.json \
  --resume-dir /home/DIFFER/log_NDP/ndp_v5_20261023_100048/n5_seed<seed>/node_update1788 \
  --resume-effective-updates 1788 --resume-batch-index 1788|1789 \
  --diagnostic-steps 20
```

实际 run-id：

| Seed | run-id | resume batch index | failure |
|---:|---|---:|---|
| 1234 | `ndp_v6_20261023_131720_n6_s1234` | 1788 | `null` |
| 4321 | `ndp_v6_20261023_131720_n6_s4321` | 1789 | `null` |

固定 LTCC 评估输出：

- `ltcc_seed1234_n6.json`
- `ltcc_seed4321_n6.json`

第一次 train-only 导出因旧入口不接受 `--arm N6` 被安全拒绝，未写入错误特征；随后只扩展枚举并重新执行，四份 N6 特征均导出成功。

## LTCC 结果

3 epoch 的 N6 直接复用 N5 update1788，因为 N6 不重复前三 epoch。百分比为 Rank-1、mAP、mINP。

| Seed | 模型 | continuation epoch | update | Rank-1 | mAP | mINP |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 3 | 1788 | 50.51% | 30.38% | 8.33% |
| 1234 | N5 | 3 | 1788 | 54.34% | 32.27% | 8.66% |
| 1234 | N6 | 3 | 1788 | 54.34% | 32.27% | 8.66% |
| 1234 | B0 | 5 | 2980 | 50.51% | 30.25% | 8.26% |
| 1234 | N5 | 5 | 2980 | 52.81% | 32.02% | 8.53% |
| 1234 | N6 | 5 | 2980 | 52.30% | 31.64% | 8.79% |
| 1234 | B0 | 10 | 5960 | 51.28% | 30.18% | 7.69% |
| 1234 | N5 | 10 | 5960 | 50.77% | 30.97% | 7.65% |
| 1234 | N6 | 10 | 5960 | 52.04% | 31.62% | 8.36% |
| 4321 | B0 | 3 | 1788 | 50.77% | 30.74% | 8.55% |
| 4321 | N5 | 3 | 1788 | 51.28% | 31.36% | 8.85% |
| 4321 | N6 | 3 | 1788 | 51.28% | 31.36% | 8.85% |
| 4321 | B0 | 5 | 2980 | 52.04% | 30.82% | 8.02% |
| 4321 | N5 | 5 | 2980 | 52.04% | 30.74% | 7.83% |
| 4321 | N6 | 5 | 2980 | 52.81% | 31.34% | 8.04% |
| 4321 | B0 | 10 | 5960 | 51.53% | 30.72% | 8.07% |
| 4321 | N5 | 10 | 5960 | 51.02% | 31.05% | 8.25% |
| 4321 | N6 | 10 | 5960 | 49.74% | 31.65% | 8.67% |

相对 B0：

| Seed | N6 节点 | ΔRank-1 | ΔmAP | ΔmINP |
|---:|---:|---:|---:|---:|
| 1234 | 5 epoch | +1.79 pp | +1.39 pp | +0.53 pp |
| 1234 | 10 epoch | +0.77 pp | +1.44 pp | +0.68 pp |
| 4321 | 5 epoch | +0.77 pp | +0.52 pp | +0.02 pp |
| 4321 | 10 epoch | -1.79 pp | +0.94 pp | +0.60 pp |

N6 在第 5 epoch 两 seed 三项指标均高于 B0；第 10 epoch 只有 Seed1234 保持 Rank-1 高于 B0，Seed4321 的 Rank-1 明显退化。

## 训练状态与 NDP 激活

| Seed | effective updates | final batch index | NDP lambda | selected | active | active rate | AMP found_inf | 非有限参数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 5960 | 5962 | 0.0 | 0 | 0 | 0.0000 | 2 | 0 |
| 4321 | 5960 | 5962 | 0.0 | 0 | 0 | 0.0000 | 1 | 0 |

两 seed 的最终 checkpoint 均包含模型、optimizer、scheduler、GradScaler、RNG 和 protocol 状态，保存恢复检查通过。AMP skip 没有导致参数非有限或训练中止。

## Train-only 局部指标

指标使用同一 2048-anchor、K=5 图和 9576 张训练图像；不使用 query/gallery。

| Seed | 模型 | update | 正确关系 | 平均 Margin | 新增错误 PID | Purity@10 |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | B0 | 2980 | 1610 | 0.05518 | 259 | 0.96445 |
| 1234 | N5 | 2980 | 1703 | 0.06581 | 85 | 0.97266 |
| 1234 | N6 | 2980 | 1674 | 0.06535 | 119 | 0.97319 |
| 1234 | B0 | 5960 | 1695 | 0.06797 | 152 | 0.97168 |
| 1234 | N5 | 5960 | 1717 | 0.07308 | 107 | 0.97231 |
| 1234 | N6 | 5960 | 1719 | 0.07363 | 92 | 0.97451 |
| 4321 | B0 | 2980 | 1679 | 0.05776 | 96 | 0.97051 |
| 4321 | N5 | 2980 | 1691 | 0.06172 | 89 | 0.96797 |
| 4321 | N6 | 2980 | 1687 | 0.05930 | 88 | 0.97070 |
| 4321 | B0 | 5960 | 1722 | 0.06578 | 57 | 0.97227 |
| 4321 | N5 | 5960 | 1683 | 0.06302 | 95 | 0.96899 |
| 4321 | N6 | 5960 | 1732 | 0.06527 | 82 | 0.97090 |

局部指标在第 10 epoch 对 N6 两 seed 均比持续 N5 更好；但这不能消除 Seed4321 的 LTCC Rank-1 下降，说明局部关系改善没有直接转化为全局检索收益。

## Checkpoint SHA256

| Seed | update2980 | update5960 |
|---:|---|---|
| 1234 | `9b6565d83b6126d0a5685060d005d04a4cb5351118f5aa9d6171271b4ecb339c` | `28ab817a2a28c9f11f658f8af8ed74c35bc9eea37ba69f046ca8abaee808e1f9` |
| 4321 | `f0b6515de3faffb8b75d03ca5313f5e664b02eb477a73e10f684a448eda96c85` | `d02505cb3c10e0c028cbfbe105335ce960b5c4c07ebb0511ef47b637fcdcc91e` |

## 最终决策

1. **N6 是否比持续 N5 更稳定？** 短程更稳定：第 5 epoch 两 seed 均优于持续 N5/B0；长程不稳定，Seed4321 第 10 epoch Rank-1 明显下降。
2. **N6 是否在 10 epoch 两 seed 同时优于 B0？** 否。只有 Seed1234 的 Rank-1 保持提升，Seed4321 为 -1.79 pp。
3. **是否值得继续更多开关时间或 60 epoch？** 不值得。当前证据不足以支持更多阶段切换搜索或 60 epoch。
4. **当前候选和下一步？** 保留 N5 3 epoch 作为主候选；N6 5 epoch 作为待复核短程候选。下一步只做预注册的独立 seed 与 PRCC 验证，不修改 Loss、不增加复杂模块。

此前完整 EVA02 PPU A/A 约有 0.4% relative L2 差异，本轮保留该数值限制；所有微小收益不视为已证实机制收益。
