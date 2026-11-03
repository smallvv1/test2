# NDP_GPC_IMPACT formal archive

- Seed: `1234`
- Arm: `NDP_GPC_IMPACT`
- Dataset runs: LTCC and PRCC, 60/60 epochs each
- Fixed main loss: Bio+GPC, `lambda_gpc=0.1`
- Train-only relation mining; official query/gallery used only for final evaluation
- Full server log: `/home/DIFFER/log_NDP/ndp_gpc_impact_full60_20261103/`
- The large `model_final.pth` files are intentionally excluded from this GitHub archive.
- No `rank*state.pt`, `rank*state.pth`, or per-rank checkpoint was created.

See [the formal report](../../reports/NDP_GPC_IMPACT_FULL60_RESULTS.md) for method, metrics, deltas, and training status.
