# NDP 训练实验报告（2026-10-23）

代码版本：`164d7b1a3242157c2e898caffd03d65445f3ab35`（远端 HEAD 与 main 核对一致）。本轮报告目录：`/home/DIFFER/reports/ndp_training_20261023_033405`。

固定约束：EVA02、feat[0] 1024 维、原始 CE/Triplet/clipBio、GPC-v1 matched-budget 选择及 `0.1` 系数均由现有训练引擎提供；本轮只新增最小 NDP。NDP 使用 update0/E60 的 train-only K=5 固定图，历史正确条件为 `M_ref(update0)>0`，当前项为 live anchor 与冻结端点的同一 `mean-positive cosine - max-negative cosine` Margin，`epsilon=0.02`、`lambda_ndp=0.05`。训练关系选择没有读取未来节点、query/gallery 或测试信息。

核心命令：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode smoke|full --seed 1234|4321 \
  --run-id <timestamp> --run-dir <new-output-dir> --diagnostic-steps 20
```

新代码和 checkpoint 的 SHA256 见 [`NDP_CHECKPOINT_AND_CODE_SHA256.csv`](NDP_CHECKPOINT_AND_CODE_SHA256.csv)；逐关系 train-only 结果见两个 `local_relation_metrics_seed*.csv`，节点汇总见 `local_node_summary_seed*.csv`。这些 CSV 未使用官方 query/gallery。

## 1. 配对协议

B0 复用了已完成且核对为 400/400 有效更新的 `outputs/ndp_overnight_20261020_201841/trajectory_seed{1234,4321}`；其来源 phase、E60 节点、epoch tag=61、global step=35760→36160、LR=2e-7 与 N1 一致。N1 的 E60 source 分别为两个 B0 seed 的 `rank{0,1}_state_epoch060.pt`，新输出位于 `outputs/ndp_training_20261023_032622`（seed1234）和 `outputs/ndp_training_20261023_033008`（seed4321）。B0 没有重跑，也没有覆盖历史结果。

训练统计：

| seed | arm | mean loss | final loss | mean CE | mean Triplet | mean Bio | mean GPC | GPC 非零 batch | NDP active | finite |
|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | B0 | 4.246458 | 3.938562 | 2.026099 | 0.149250 | 2.067852 | 0.032572 | 46 | 0 | True |
| 1234 | N1 | 4.248059 | 3.932709 | 2.026881 | 0.149894 | 2.067896 | 0.032420 | 46 | 135 | True |
| 4321 | B0 | 4.303120 | 4.157325 | 2.074933 | 0.156627 | 2.068125 | 0.034354 | 47 | 0 | True |
| 4321 | N1 | 4.300611 | 4.137074 | 2.074567 | 0.154305 | 2.068213 | 0.034305 | 47 | 98 | True |

## 2. 固定节点结果（N1-B0）

| seed | update | Δ mean Margin | Δ 正确关系 | Δ 新错误 PID | Δ Purity@10 | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---|---|---|---|---|---|---|---|
| 1234 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0.00% | 0.0000000 | 0.0000000 |
| 1234 | 20 | 0.0001174 | 2 | -22 | 0.0025391 | -0.77% | 0.0005523 | -0.0006068 |
| 1234 | 50 | -0.0000684 | 4 | -24 | 0.0011719 | -1.02% | 0.0000556 | -0.0006295 |
| 1234 | 100 | -0.0000672 | 8 | -17 | 0.0009766 | -0.26% | -0.0003695 | -0.0003836 |
| 1234 | 200 | -0.0001515 | 1 | -12 | -0.0001953 | 0.51% | 0.0007528 | -0.0002859 |
| 1234 | 400 | 0.0011737 | 4 | -4 | -0.0009766 | -0.26% | 0.0000723 | 0.0010169 |
| 4321 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0.00% | 0.0000000 | 0.0000000 |
| 4321 | 20 | 0.0002601 | 7 | -5 | -0.0007812 | 0.00% | 0.0000185 | 0.0000562 |
| 4321 | 50 | 0.0000022 | 4 | -3 | 0.0003906 | 0.51% | 0.0001388 | 0.0001496 |
| 4321 | 100 | 0.0002096 | 2 | -14 | 0.0007812 | -0.26% | 0.0005111 | -0.0004506 |
| 4321 | 200 | -0.0002127 | 4 | -7 | -0.0005859 | -0.77% | 0.0002231 | 0.0001307 |
| 4321 | 400 | -0.0000590 | 10 | -5 | -0.0011719 | -0.77% | 0.0014861 | 0.0004203 |

`Δ` 均为 N1−B0；Purity@10 是固定 K=5 endpoint train-only bank 内的 top-10 同 PID 比例。完整 B0/N1 数值及逐关系 CSV 保存在本报告目录的 `NDP_TRAINING_PAIRED_SUMMARY.csv`、`../../outputs/ndp_evaluation_20261023_033405/local_metrics_seed{1234,4321}/`。官方 LTCC 使用固定 query/gallery、clothes-changing 过滤、归一化 feat[0]、欧氏距离、无 reranking；mINP 是对每个有效 query 最后一个正样本位置的 precision 取平均。

## 3. 400 步重点

| seed | NDP active batch/400 | NDP active relation | Δ mean Margin | Δ 正确关系 | Δ 新错误 PID | Δ Purity@10 | Δ Rank-1 | Δ mAP | Δ mINP |
|---|---|---|---|---|---|---|---|---|---|
| 1234 | 54/400 (13.50%) | 135 | 0.0011737 | 4 | -4 | -0.0009766 | -0.26% | 0.0000723 | 0.0010169 |
| 4321 | 42/400 (10.50%) | 98 | -0.0000590 | 10 | -5 | -0.0011719 | -0.77% | 0.0014861 | 0.0004203 |

seed1234 的 400 步局部关系正确数 +4、新错误 PID −4、平均 Margin +0.001174；seed4321 分别为 +10、−5、平均 Margin −0.000059。官方 Rank-1 在两个 seed 都下降（约 −0.26 和 −0.77 个百分点），而 mAP/mINP 都只有很小正变化；因此不存在两个 seed 一致的全面检索改善。

## 4. 同 anchor 其他负 PID 的局部影响（观察性）

以下只比较 B0/N1 的固定 update400 特征，不是一次网络 step 的反事实因果估计。`reliable` 指 update0 Margin>0；`other-side` 对同一 anchor 的其它历史可靠负 PID 做汇总。285/276 个 anchor 至少有两个历史可靠关系。

| seed | 可靠关系 ΔMargin | 可靠关系变差比例 | 其它可靠 PID ΔMargin | 其它可靠 PID 变差比例 |
|---|---|---|---|---|
| 1234 | 0.0009360 | 48.87% | 0.0007894 | 49.45% |
| 4321 | -0.0003275 | 51.81% | -0.0005905 | 52.42% |

seed1234 的其它可靠 PID 平均 ΔMargin 为 +0.000789，变差比例 49.45%；seed4321 为 −0.000591，变差比例 52.42%。这支持“没有可证明的普遍邻域保护”，也提示 seed4321 存在轻微附带退化信号；由于 B0/N1 是整段 continuation 配对，不能把它等同于 NDP 单条关系的独立因果损害。

## 5. 统计限制

这里只有两个 seed、单个 400-step continuation 和固定节点配对差值，没有把两个 seed 当作充分样本做显著性检验，也没有用未来节点选 mask。PPU 完整模型 A/A 约 0.4% 的既有差异仍限制小幅结果的因果解释；本轮 N1/B0 是探索性对照，不能声称 NDP 已改善 LTCC。
