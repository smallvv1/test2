# NDP-V2 实现与训练记录（最终）

本报告汇总两个独立新时间戳：1 continuation epoch 的 `/home/DIFFER/log_NDP/ndp_v2_20261023_061358`，以及按训练正常后继续执行的 3 continuation epochs `/home/DIFFER/log_NDP/ndp_v2_20261023_070201`。报告目录：`/home/DIFFER/reports/ndp_v2_20261023_070201`。

## 实现结论

NDP-V2 已真实参与训练：N2 的 live anchor 保留 autograd，参考 update0 特征停止梯度；NDP loss、feature gradient 和模型参数更新均被观察到。基础损失始终是 `CE + Triplet + clipBio + 0.1 * original GPC-v1`，GPC selector、matched-budget、EVA02、sampler、Bio、Triplet 和评估协议未修改。

V2 使用每 seed 2048 个 train-only update0 anchor，每 anchor 4 个正样本和 5 个不同负 PID；历史正确关系为 seed1234=`1699/2048`、seed4321=`1660/2048`。`lambda_ndp=0.05`、`epsilon=0.02`。由于没有 E40/E45/E50 完整状态，B0/N2 均从同一 E60 rank state 恢复，并一致设置 LR=`1e-6`。

## 真实运行命令

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/ndp_e60_minimal_pair_runner.py --mode full --arm B0|N2 --seed 1234|4321 \
  --max-effective-updates 1788 --save-updates 596,1192,1788 \
  --run-id ndp_v2_20261023_070201_epoch3_<arm>_s<seed> \
  --run-dir /home/DIFFER/log_NDP/ndp_v2_20261023_070201/epoch3_<arm>_seed<seed> \
  --reference-feature <seed update0 full feature> --graph <seed V2 graph> --initial-lr 1e-6
```

20 步 smoke 命令和 1 epoch 命令保存在 `/home/DIFFER/log_NDP/ndp_v2_20261023_061358` 的 `training_config.json` 与历史报告中；官方评估使用 `scripts/ndp_official_ltcc_nodes.py`，train-only 局部指标使用 `scripts/ndp_v2_local_metrics.py`。测试：`/usr/local/bin/python3.10 -m pytest -q tests/test_ndp_minimal_pair_math.py` 为 `2 passed`，全部新增脚本通过 `py_compile`。

## 3-epoch真实训练统计

| Seed | Arm | 有效更新 | CSV batch行 | 激活 batch/率 | selected | active relation | mean NDP | AMP skip | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1234 | B0 | 1788 | 1790 | 0/1790 (0.00%) | 0 | 0 | 0.000000 | 2 | None |
| 1234 | N2 | 1788 | 1789 | 829/1789 (46.34%) | 3233 | 1262 | 0.009704 | 1 | None |
| 4321 | B0 | 1788 | 1790 | 0/1790 (0.00%) | 0 | 0 | 0.000000 | 2 | None |
| 4321 | N2 | 1788 | 1789 | 729/1789 (40.75%) | 3036 | 1105 | 0.008309 | 1 | None |

seed1234 B0/N2 分别有 2/1 次 AMP skip，seed4321 B0/N2 分别有 2/1 次 AMP skip；skip 后通过额外 batch 达到 1788 个有效更新，参数始终有限。四组 `node_update0596/1192/1788` 和 restore check 均已落盘。NDP 3 epoch active batch 率为 seed1234=`46.34%`、seed4321=`40.75%`，明显高于 N1 历史约10.5%–13.5%。

## update1788 checkpoint SHA256

| Seed | Arm | Artifact | SHA256 |
|---|---|---|---|
| 1234 | B0 | `model.pth` | `1d1681d189d79a29808f0e93459b8ffc4336d17212084a17f6a65da60a565bd0` |
| 1234 | B0 | `rank0_state.pt` | `53622cfb73dbe0b8f965a63ddfc725c18f9929a2fb3d8a8c58fc058229f24747` |
| 1234 | B0 | `rank1_state.pt` | `96937f7e005e6314f57148ec315b25ae844ed866b380b29167fe42e8717f70cb` |
| 1234 | N2 | `model.pth` | `5a0d81049cef14224e10f2c48ccef52a290055aa0c57eb32918c1e12ba36f804` |
| 1234 | N2 | `rank0_state.pt` | `b0cd0b583af5949e0b8d4494225fe4841d614b7a34e6ce5355d634fae2ce9824` |
| 1234 | N2 | `rank1_state.pt` | `f14b50ef37813eb38bae35f535c9a494984fa21e7d8bbcbc2f28b379952534b3` |
| 4321 | B0 | `model.pth` | `9d6fa84ee5c58a65d1b614f5c691e969c0d0daec820495472a622a143ebee5c9` |
| 4321 | B0 | `rank0_state.pt` | `687fa288ba3ae4748382188d8df2181be6dec57a0cd1f8b5fbf042433f10639a` |
| 4321 | B0 | `rank1_state.pt` | `d3010e616f4ed14157b3ebc2717dd7e8aa0ef5c5e7b2e1b8ddfbea133fc4e454` |
| 4321 | N2 | `model.pth` | `b2d18b40df3983d533aa835dbbcd84d158da43684936456e2bd5a5e2b2189ced` |
| 4321 | N2 | `rank0_state.pt` | `98be252546d76d8f8712d25dbf0abbee3bca41ed46da7b6642878e8843d4f744` |
| 4321 | N2 | `rank1_state.pt` | `e6ca7c166a10044646e392f4a915167024f74c89b1f242c1f7dd91b6349120f3` |

## 代码与固定引擎 SHA256

| 文件 | SHA256 |
|---|---|
| `scripts/ndp_e60_minimal_pair_runner.py` | `9625a3925db8f35efbfa98be28b56831652f0c8293e8a6c5724a35ec2c02384b` |
| `scripts/ndp_v2_reference_graph.py` | `87e570aaa90e5b36543edd0f1d9aa1814f7400d3c0378e2e260125c71d75e16e` |
| `scripts/ndp_v2_feature_export.py` | `04a6291f6b434c066b36ee2e47d67516d80d9365425edc3f56c8a85dafb59b85` |
| `scripts/ndp_v2_local_metrics.py` | `3d07695b2c489963d281b2bb380228103749e8161f2eeeac5a821526fd64f89a` |
| `/data/DIFFER/scripts/gpc_v2_train.py`（只读） | `d6d149ebbe8819552689d6ce5f1e053f5adea205a45f4ace3fce81296a1991c5` |
| `reconstructed_ltcc_eval_config.yml` | `c7c9e967a5ae824d9bfad9fb27ea77bd1af645e4de009c18111609523d8d33e6` |

远端 main 复核为 `e27d59ed74ff348061c43ffd799dc91af8d6e0f0`；本地 `/data/DIFFER` commit 为 `573b0502a231568d509fe3a463664c1a3c42e258`；未向 GitHub push。
