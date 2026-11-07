# NDP_SHN_RELIABLE Full-60 Run

- Arm: `NDP_SHN_RELIABLE`
- Seed: `1234`
- Dataset runs: LTCC 60/60, PRCC 60/60
- Output root: `/home/DIFFER/log_NDP/ndp_shn_reliable_full60_seed1234_20261107`
- GPC: unchanged, weight `0.1`
- NDP: live `SHN_MP` plus at most two reliable train-only bank positives, auxiliary factor `0.25`, total weight `0.001`
- Query/gallery used during training: `false`
- Per-rank checkpoint files: `false`

Final metrics (Rank-1 / mAP):

| Protocol | Result |
|---|---:|
| LTCC General | 84.38 / 54.05 |
| LTCC Clothes-Changing | 55.87 / 33.71 |
| PRCC Same-Clothes | 100.00 / 99.50 |
| PRCC Clothes-Changing C→A | 64.47 / 63.87 |

See [`reports/NDP_SHN_RELIABLE_FULL60_RESULTS_20261107.md`](../../reports/NDP_SHN_RELIABLE_FULL60_RESULTS_20261107.md) for the three-stage table, pp deltas, ablations, and protocol details.
