# 本轮证据索引

运行时间：`20261021_220823`。远端 main 检查到 `77364d349820296760da857e43fd540ecdd7dfbc`；服务器 engine 工作树审计 HEAD 为 `573b0502a231568d509fe3a463664c1a3c42e258`。PPU 启动命令为 `source /usr/local/PPU_SDK/envsetup.sh ppu`，解释器 `/usr/local/bin/python3.10`，torch 2.6.0。

## 代码 SHA256

详见 `outputs/ndp_margin_consistency_20261021_220823/RUN_MANIFEST.json` 和 `SHA256SUMS.txt`。关键新增源码：

- `ndp_margin_consistency.py`: `3cb00c7e571e5ecefc939f44a034305d9977c248b096d28f9b4e09c834e9c353`
- `ndp_k5_trajectory_feature_export.py`: `0d418230a4f31e5e18007033b9191fb72e60c95f8f0bc7a36321df5cedcf1bd6`
- `ndp_full_multipid_reactive_audit.py`: `958e9666e5dbf657194b481ff25bde30857f9915192ddbd8e532c6e0d6b83d91`
- `ndp_triplet_complementarity_audit.py`: `e1f1370d6057b3aa05929dcd9982333a2e5cc23d37ea1d151cb91722c8fb120b`
- `test_ndp_margin_consistency_current.py`: `e401c1bec50d7ae263ce837dfcdb7fd9c7aa9ec644b14b6a2728db4324503b49`

固定 baseline 内容 hash 未变：`gpc_v2_train.py=d6d149...1991c5`，`make_loss.py=9be543...9043`，`eva02_l_bio.yml=c05899...0b97`，`triplet_loss.py=86656d...2092a`；完整值见 Manifest。update0 checkpoint：seed1234 `e11c90...11177`，seed4321 `d69859...09637`。

## 关键命令

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
/usr/local/bin/python3.10 -u scripts/ndp_k5_trajectory_feature_export.py --seed {1234,4321} --graph-type difficult --k-pid 5
/usr/local/bin/python3.10 -u scripts/ndp_full_multipid_reactive_audit.py --seed {1234,4321} --epsilon 0.02
/usr/local/bin/python3.10 -u scripts/ndp_directed_bidirectional_gradient_audit.py --seed {1234,4321} ...
/usr/local/bin/python3.10 -u scripts/ndp_triplet_complementarity_audit.py --seed {1234,4321} ...
/usr/local/bin/python3.10 -u scripts/ndp_aa_operator_probe.py --mode amp --no-optimizer-step ...
PYTHONPATH=/home/DIFFER/scripts /usr/local/bin/python3.10 -m pytest -q tests
```

测试最终为 `38 passed`。无 checkpoint 被复制到 source snapshot；PPU `.tensors.pt` 仅保留本地无更新反向复核，不上传 GitHub。
