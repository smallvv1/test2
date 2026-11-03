# NDP_GPC_DELTA formal archive

- Seed: `1234`
- Arm: `NDP_GPC_DELTA`
- LTCC and PRCC: 60/60 epochs and official evaluation completed
- Fixed main loss: Bio+GPC, `lambda_gpc=0.1`
- NDP coefficient: `0.001`
- Train-only post-step margin-risk detection; official query/gallery used only for final evaluation
- Full server output: `/home/DIFFER/log_NDP/ndp_gpc_delta_full60_20261103/`
- Large `model_final.pth` files are intentionally excluded from this GitHub archive
- No `rank*state.pt`, `rank*state.pth`, or per-rank checkpoint was created

See [the formal report](../../reports/NDP_GPC_DELTA_FULL60_RESULTS.md) for metrics, deltas, and attribution.
