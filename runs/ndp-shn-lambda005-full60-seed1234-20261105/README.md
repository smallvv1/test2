# NDP-SHN λ=0.005 / seed=1234

- LTCC/PRCC：60 epoch 已完成，PPU 训练与官方评估完成。
- 唯一方法变更：`lambda_ndp: 0.001 -> 0.005`。
- GPC 权重固定 `0.1`，代码、正样本选择和采样器未变。
- 日志：`/home/DIFFER/log_NDP/ndp_shn_lambda005_full60_seed1234_20261105/`
- 最终模型仅保留服务器上的 `LTCC/model_final.pth` 与 `PRCC/model_final.pth`，未上传仓库。
- 未生成任何 `rank*state.pt`、`rank*state.pth` 或逐 Rank checkpoint。

结论：PRCC C→A 为 `64.8038/63.7275`，相对固定 Bio+GPC `+0.7056/+0.3132`；LTCC CC 为 `51.5306/28.8818`，相对固定 Bio+GPC `-3.8265/-5.0549`。该权重没有形成稳定跨数据集增益，不再继续做权重扫描。

完整中文报告：`reports/NDP_SHN_LAMBDA005_FULL60_RESULTS_20261105.md`
