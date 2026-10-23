# NDP-V2 实现与真实训练记录

实验目录：`/home/DIFFER/log_NDP/ndp_v2_20261023_061358`；报告目录：`/home/DIFFER/reports/ndp_v2_20261023_061358`。

## 结论先行

NDP-V2 已真实进入训练图并执行 optimizer step。两 seed、B0/N2 各 596 个有效更新均完成；N2 的 NDP 梯度可到达 EVA02 参数。update596 的固定 LTCC 评估中，N2 相对同起点同 LR 的 B0 在两个 seed 的 Rank-1、mAP、mINP 均为正差值，但 train-only 局部关系指标并非一致改善，因此这是一项有希望的探索性结果，不是机制已证实。

## 固定基线与实现

- 远端 `main` 复核：`e27d59ed74ff348061c43ffd799dc91af8d6e0f0`；本地 `/data/DIFFER` 引擎 commit：`573b0502a231568d509fe3a463664c1a3c42e258`。`/home/DIFFER` 不是 Git worktree，本轮未 push。
- 基础目标保持 `CE + Triplet + clipBio + 0.1 * original GPC-v1`。GPC selector、matched-budget、损失实现和系数没有改动；N2 只在训练入口叠加 NDP。
- NDP：`L_ndp = mean(ReLU(M_ref - M_current - 0.02))`，`lambda_ndp=0.05`。`M_ref` 是 update0 train-only 参考并停止梯度；`M_current` 由当前 live `feat[0]` anchor 与冻结参考端点计算，正样本均值减五个负 PID 相似度最大值。正负端点不要求同 batch 出现。
- V2 图：每个 seed 2048 个合法 train anchor；每 anchor 4 个同 PID 正样本、5 个不同负 PID。update0 历史正确关系分别为 seed1234=`1699/2048`、seed4321=`1660/2048`。图选择未读取未来节点、query/gallery 或测试集。
- 由于没有 E40/E45/E50 完整状态，B0/N2 都从同一 E60 rank state 恢复，并把所有 optimizer param group 的 LR 一致重置为 `1e-6`；没有只提高 NDP LR。两 arm 的 sampler、有效更新预算和评估协议相同。

## 真实命令

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode smoke --arm B0|N2 --seed 1234|4321 \
  --run-id <new> --run-dir /home/DIFFER/log_NDP/ndp_v2_20261023_061358/<dir> \
  --reference-feature <seed-specific update0_full_features.npz> \
  --graph <seed-specific multi_pid_reference_graph_v2.json> \
  --initial-lr 1e-6
```

正式配对命令相同，增加 `--max-effective-updates 596 --save-updates 596`。官方评估命令为 `scripts/ndp_official_ltcc_nodes.py`，一次 seed 同时加载 B0/N2 update596；局部指标命令为 `scripts/ndp_v2_local_metrics.py`。新增 focused test：`/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_minimal_pair_math.py`，结果 `2 passed`；四个新增脚本均通过 `py_compile`。

## 20 步 Smoke

| Seed | Arm | 有效更新 | 候选关系 | 选中 | 实际激活关系 | 激活 batch | 恢复 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1234 | B0 | 20 | 2048 | 0 | 0 | 0/20 (0.000%) | PASS |
| 1234 | N2 | 20 | 2048 | 30 | 14 | 10/20 (50.000%) | PASS |
| 4321 | B0 | 20 | 2048 | 0 | 0 | 0/20 (0.000%) | PASS |
| 4321 | N2 | 20 | 2048 | 32 | 12 | 9/20 (45.000%) | PASS |

N2 seed1234 smoke 的 rank0 统计为 30 selected、14 active、10/20 active batch；N2 seed4321 的 smoke 统计保存在对应 `run_summary.json`。两 seed 的 N2 smoke 都有非零 NDP feature/probe gradient、有限 loss 和可加载 checkpoint。B0 smoke 作为同配置控制完成但 NDP 为 disabled，选中/激活为 0。

## 第一 continuation epoch（update596）

| Seed | Arm | 有效更新 | 实际 batch 行数 | 激活 batch | 选中关系 | 激活关系 | 平均 NDP | AMP skip | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1234 | B0 | 596 | 596 | 0/596 (0.000%) | 0 | 0 | 0.000000 | 0 | None |
| 1234 | N2 | 596 | 597 | 274/597 (45.896%) | 1042 | 423 | 0.009743 | 1 | None |
| 4321 | B0 | 596 | 596 | 0/596 (0.000%) | 0 | 0 | 0.000000 | 0 | None |
| 4321 | N2 | 596 | 596 | 253/596 (42.450%) | 995 | 386 | 0.009163 | 0 | None |

N2 seed1234 在 batch570 有一次 AMP `found_inf=1`，该 step 被跳过、scale 从256降到128，参数仍有限；随后通过额外一个 batch 达到 596 个有效更新。N2 seed4321 无 AMP skip。所有正式 run 的 `parameter_finite=True`，且 update596 rank checkpoint restore check PASS。

## 文件与 SHA256

新增代码：

| 文件 | SHA256 |
|---|---|
| `scripts/ndp_e60_minimal_pair_runner.py` | `9625a3925db8f35efbfa98be28b56831652f0c8293e8a6c5724a35ec2c02384b` |
| `scripts/ndp_v2_reference_graph.py` | `87e570aaa90e5b36543edd0f1d9aa1814f7400d3c0378e2e260125c71d75e16e` |
| `scripts/ndp_v2_feature_export.py` | `04a6291f6b434c066b36ee2e47d67516d80d9365425edc3f56c8a85dafb59b85` |
| `scripts/ndp_v2_local_metrics.py` | `3d07695b2c489963d281b2bb380228103749e8161f2eeeac5a821526fd64f89a` |
| `/data/DIFFER/scripts/gpc_v2_train.py`（只读核验） | `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5` |
| `/data/DIFFER/diagnostics/reconstructed_ltcc_eval_config.yml` | `c7c9e967a5ae824d9bfad9fb27ea77bd1af645e4de009c18111609523d8d33e6` |

完整 checkpoint SHA256、参考图 SHA256、特征导出 SHA256 写在 `/home/DIFFER/log_NDP/ndp_v2_20261023_061358/` 下的 JSON manifest、`run_summary.json` 和 `training_config.json`；原始训练日志未覆盖。

update596 checkpoint SHA256：

| Seed | Arm | Artifact | SHA256 |
|---|---|---|---|
| 1234 | B0 | `model.pth` | `02417106ce6f218be638fd85ad86de4fe791cc909e8c6d643c3e9ddb4365dda1` |
| 1234 | B0 | `rank0_state.pt` | `98872c4ebabb0e8394ef1003a478533c644a68410b678094f92a5c3b7ac3535f` |
| 1234 | B0 | `rank1_state.pt` | `f915b604dfb183c500e08c6499974ac677600bd1c473a8d7ecedd9c0660222c9` |
| 1234 | N2 | `model.pth` | `15a73339190e3405f7ab0d630b9994b8855ac99bd95e0e0a93ba1181b34e482a` |
| 1234 | N2 | `rank0_state.pt` | `97bf57de88cf6b53feca27ca8c0b7862d73b1df613efb1d7237836be7ccdc6d9` |
| 1234 | N2 | `rank1_state.pt` | `bab84328319a41d96bb4c054a2d8e6c996162e85e5f891c6d0ed7b76f2fbd632` |
| 4321 | B0 | `model.pth` | `78bbe39c335591fd75b588c404f3b3464d1a5234ade2c018b6f50c147d45c03c` |
| 4321 | B0 | `rank0_state.pt` | `51e41e5a1999c3fbfd8ea7ef61572222736987a9367e7e9d9a332823e142f2f9` |
| 4321 | B0 | `rank1_state.pt` | `9449bf52354dbe6db5ed2ab6dfd15cb87153f492f59a112fbd37727ae13a560d` |
| 4321 | N2 | `model.pth` | `2e3ec0bcd6c6a62a9408c2337b4fabf219a933c5d37c0fc195bd256077342e3e` |
| 4321 | N2 | `rank0_state.pt` | `89ac3c248165b57e93db8c77f9accfd29ecc26eee807d761f841b3ef12f50504` |
| 4321 | N2 | `rank1_state.pt` | `70134a1189d8717b19a708150ffcdf32b21e730f96d5dbfca935b3215d61ec67` |

## 限制

本轮只完成一个 continuation epoch，LR 是从 E60 状态一致重置的探索性 `1e-6`，不是原始完整 60 epoch 训练协议。PPU 完整 EVA02 A/A 既有约0.4%差异仍然存在，本轮没有无限重复算子审计。结论限定为 matched B0/N2 的探索性证据。
