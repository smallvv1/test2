# NDP_SR full60 seed=1234 archive

This directory contains only compact training/evaluation artifacts from the formal LTCC and PRCC runs. The final model files are intentionally not copied here because each is about 1.2 GB; their paths and SHA256 values are recorded in `final_model.json` and `reports/NDP_SR_FULL60_RESULTS.md`.

- `train.log`: epoch-level loss, NDP activation, skip count, and learning rate.
- `metrics_per_epoch.csv`: full per-epoch optimization statistics.
- `online_neighbor_summary.csv`: fixed teacher-selected train-only relation graph and refreshed student-bank metadata.
- `eval_final.json`: official General/CC or Same-Clothes/C→A evaluation result.
- `final_model.json`: reproducibility metadata and final model hash.
