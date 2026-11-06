# NDP_SHN_BANK Full60 / seed=1234

- Formal log root: `/home/DIFFER/log_NDP/ndp_shn_bank_full60_seed1234_20261106/`
- Bio-only unified eval root: `/home/DIFFER/log_NDP/bio_only_unified_eval_20261106/`
- LTCC: 60/60 epochs, 35760 steps, PPU, model SHA256 `4ae69fcc68ddf53a9c670a84a4227b3535dc206965b7d46aa49b571a1ab16d6b`
- PRCC: 60/60 epochs, 33060 steps, PPU, model SHA256 `219ac703e3d67ccdf8cf81187749c391884e2e51d456b66401a7b21237561ac8`
- GPC: frozen code/sampler/positive selector, weight `0.1`
- NDP: train-only student feature bank, refresh every 5 epochs after epoch 3, `lambda_ndp=0.001`
- No query/gallery feature used for training or selection.
- No `rank*state.pt`, `rank*state.pth`, or per-rank checkpoint was created.

See [`reports/NDP_SHN_BANK_FULL60_RESULTS_20261106.md`](../../reports/NDP_SHN_BANK_FULL60_RESULTS_20261106.md) for the Chinese paper-ready table and pp deltas.
