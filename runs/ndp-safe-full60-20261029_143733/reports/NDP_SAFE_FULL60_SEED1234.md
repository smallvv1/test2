# NDP_SAFE LTCC/PRCC Seed1234 Full60 实验报告

本报告只汇总真实生成的 `RUN_COMPLETE.json` 与固定协议评估结果；缺失项明确标记为未完成。

- 输出根目录：`/home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733`
- seed：1234
- 基础损失：CE + Triplet + clipBio + 0.1 × GPC-v1
- NDP_SAFE：margin=0.05，lambda=0.005，前10 epoch线性warmup，每5 epoch更新一次 train-only 邻域，最多3个负PID。
- 测试集未参与训练、邻域构建或参数选择。

## 实际命令与版本

正式训练使用 PPU 环境和双进程 torchrun，四组命令形如：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_ndp_listwise_full60.py --mode train \
  --dataset {ltcc|prcc} --seed 1234 --arm {B0|NDP_SAFE} \
  --epochs 60 --output-dir /home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733/...
python3.10 scripts/train_ndp_listwise_full60.py --mode eval \
  --dataset {ltcc|prcc} --seed 1234 --arm {B0|NDP_SAFE} \
  --output-dir /home/DIFFER/log_NDP/NDP_SAFE_FULL60_SEED1234_20261029_143733/...
```

- 上游仓库检查到的 commit：`14ccb751c10d49e4b76451da26bc37bc99dcd4d7`；`/home/DIFFER` 是独立工作目录，NDP 修改未推送。
- 本次实际训练代码 SHA256：`train_ndp_listwise_full60.py` = `46353ddd7ec829826edcc83b5c1bfc0a7efd04a6b58c2520f0c7b4197ea1a89d`；`ndp_v5_prcc_runner.py` = `4df4bb1a2a67bb5bbee0016086f7a94e30c8ac2f09587f5f5128b4cbe035b1f3`；`ndp_safe.py` = `3cccd4f5b6e26a26f657e0537e169619c74d9ca6bd5c9ba9a8047b1f8e05807b`。
- LTCC 正式初始化 SHA256：`72892762c37044b85dc9ed9999c833da574026c3715e760885363201a8a40ed0`。
- PRCC 正式初始化 SHA256：`db5c71389b37a859ec82b3e529c1cd7149c21a85df4a1b34f9650c9f6bd9378f`。

## 完成状态与最终指标

| 数据集 | Arm | 完成 epoch | General/SC R1 | CC/C→A R1 | General/SC mAP | CC/C→A mAP | 模型 SHA256 |
|---|---|---:|---:|---:|---:|---:|---|
| LTCC | B0 | 60 | 85.3955% | 57.1429% | 53.6745% | 33.4150% | `55a46707b51b7c5b1c0b9da2de0a5cd33b9e8fd07f725e73f49eb1c3d6d91526` |
| LTCC | NDP_SAFE | 60 | 84.7870% | 50.7653% | 52.4340% | 28.9737% | `2b54cf4a0facc8fbed8675c088104fce7c9ee1835efe4e01e292fe528ad53a95` |
| PRCC | B0 | 60 | 100.0000% | 64.3522% | 99.4811% | 63.9120% | `a85715ed909b9899a18596b8ce7cd10328cde17022a5a1e473a2ba298f512415` |
| PRCC | NDP_SAFE | 60 | 100.0000% | 64.6063% | 99.4277% | 63.6400% | `1a024bfd11a7cc8f51f596d059efa92a0ca00982d1f8ea5ed3e96a5a6111aff1` |

## 训练侧摘要

| 数据集 | Arm | 最后 epoch loss | NDP loss | NDP 激活 batch | 激活关系 | NDP/主梯度比 | AMP skip | 有效更新 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LTCC | B0 | 4.964011 | 0.000000 | 0.000000 | 0 | Unavailable | 0 | 596 |
| LTCC | NDP_SAFE | 5.338776 | 0.110648 | 1.000000 | 11853 | Unavailable | 0 | 596 |
| PRCC | B0 | 7.177611 | 0.000000 | 0.000000 | 0 | Unavailable | 0 | 551 |
| PRCC | NDP_SAFE | 7.182942 | 0.026202 | 0.998185 | 9833 | Unavailable | 0 | 551 |

## NDP_SAFE 相对同批次 B0 的差值

差值为 NDP_SAFE − B0；正值代表 NDP_SAFE 更高。

| 数据集 | 协议 | Rank-1 差值 | mAP 差值 |
|---|---|---:|---:|
| LTCC | General | -0.6085 pp | -1.2406 pp |
| LTCC | Clothes-Changing | -6.3776 pp | -4.4414 pp |
| PRCC | Same-Clothes | +0.0000 pp | -0.0534 pp |
| PRCC | Clothes-Changing C→A | +0.2540 pp | -0.2720 pp |

## 与历史结果的可比性

本批次使用 `/data/DIFFER` 中已核验的正式 Bio common init，并将 GPC phase 固定为 `phase7i_relation_specificity`；这修正了旧 NLIST 批次的初始化与 GPC namespace 混杂。历史用户报告的 LTCC Bio+GPC 数值仍需以相同评估器、相同初始化和相同 GPC phase复核，不能仅凭旧数字宣称严格复现。

## 文件安全检查

- 本批次禁止逐 rank 状态文件；下面列出扫描结果。
- `rank*state.pt/pth` 与 `checkpoint_e*_rank*`：未发现

## 决策

四个 arm 均实际完成60 epoch且固定协议评估成功。LTCC 上 NDP_SAFE 的 General 与 Clothes-Changing 均低于同批次 B0；PRCC 上 C→A Rank-1 小幅上升但 mAP 下降，不能称为稳定增益。NDP_SAFE 的梯度范数比在本批次 CSV 中为 nan，故不据此声称低干扰；后续若继续研究，先修复梯度统计而不是延长当前配置。

与用户此前报告的 LTCC Bio+GPC（CC Rank-1=55.6122%、mAP=33.2649%；General Rank-1=83.1643%、mAP=53.5246%）相比，本批次同一运行的 B0 实际为 CC 57.1429%/33.4150%、General 85.3955%/53.6745%，因此本次 B0 并未低于该历史结果；降低的是 NDP_SAFE：CC 50.7653%/28.9737%、General 84.7870%/52.4340%。历史结果与本批次若非完全相同初始化、GPC namespace、代码和评估器，仍不能作为严格配对。

PRCC 并非未完成：B0 与 NDP_SAFE 均完成 60 epoch，`NDP_SAFE_PRCC_EXIT=0`，并已完成 Same-Clothes 与 C→A 固定评估；此前“仅 smoke”的判断对应更早批次或评估前状态。

状态：`COMPLETE`
