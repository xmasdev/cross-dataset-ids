# Metaheuristic Feature Selection — Results

- Optimiser: **GA** (also ran GA and binary PSO)
- Search space: 12 shared features
- Best subset (5 features): duration, dst_pkts, total_pkts, bytes_per_pkt_dst, src_dst_pkt_ratio

## Cross-dataset (mean over 6 source→target pairs)

| Metric | Baseline (all features) | Optimised subset | Change |
|---|---|---|---|
| Accuracy | 0.3726 | 0.4071 | +0.0345 |
| Balanced accuracy | 0.4650 | 0.4714 | +0.0064 |
| Macro-F1 | 0.3076 | 0.3980 | +0.0903 |
| Attack recall | 0.4789 | 0.4361 | -0.0429 |

## Same-dataset (mean over 3 datasets)

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| Accuracy | 0.9666 | 0.9360 | -0.0306 |
| Balanced accuracy | 0.9597 | 0.9307 | -0.0290 |
| Macro-F1 | 0.9581 | 0.9190 | -0.0392 |
