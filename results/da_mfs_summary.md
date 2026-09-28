# DA-MFS — Domain-Aligned Metaheuristic Feature Selection

- Optimiser: **GA** (GA + binary PSO both run)
- Best subset (4/12 features): duration, src_pkts, total_bytes, total_bytes_per_pkt
- Alignment: per-dataset quantile transform -> standard normal
- Fitness: mean cross-dataset 0.5*balanced-accuracy + 0.5*macro-F1

## Cross-dataset results (mean over 6 train→test pairs, FULL data)

| Configuration | Accuracy | Balanced acc | Macro-F1 | Attack recall |
|---|---|---|---|---|
| (a) Raw, all features | 0.3726 | 0.4650 | 0.3076 | 0.4789 |
| (b) Aligned, all features | 0.4776 | 0.5298 | 0.3845 | 0.6614 |
| (c) Aligned + optimised subset | 0.5083 | 0.5689 | 0.4604 | 0.6065 |

## Improvement over raw baseline

| Metric | Raw | DA-MFS | Change |
|---|---|---|---|
| Accuracy | 0.3726 | 0.5083 | +0.1358 |
| Balanced acc | 0.4650 | 0.5689 | +0.1039 |
| Macro-F1 | 0.3076 | 0.4604 | +0.1527 |
| Attack recall | 0.4789 | 0.6065 | +0.1276 |
