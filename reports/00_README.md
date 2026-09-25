# BTP Project — Cross-Dataset Generalization of ML-Based Intrusion Detection

**Topic:** Evaluating and improving the cross-dataset generalization of machine-learning
intrusion detection systems across CIC-IDS-2017, UNSW-NB15, and TON_IoT, using
metaheuristic/bio-inspired feature selection.

## Document set (for mid-semester review)

| # | Document | Purpose |
|---|---|---|
| [01](01_dataset_report.md) | Dataset Report | First-hand analysis of the three datasets: provenance, features, class distributions, quality issues, feature-schema mismatch, and label-taxonomy mapping. |
| [02](02_literature_review.md) | Literature Review | ~33 papers across 4 themes, each summarised (problem/method/datasets/results/limitation), with a gap-synthesis table. |
| [03](03_objectives_and_gaps.md) | Research Gap, Objectives & Scope | Formal gap statements, novelty, 5 objectives, scope decisions, risks. |
| [04](04_methodology_and_plan.md) | Methodology & Detailed Plan | 7-phase plan, evaluation protocol, fitness-function design, baselines, timeline, success criteria. |

Figures referenced by the reports live in `figures/`.

## Supporting artefacts

- `src/analyze_datasets.py` — computes dataset statistics from raw CSVs.
- `src/generate_report_assets.py` — regenerates `dataset_stats.json` and all figures.
- `src/dataset_stats.json` — measured statistics (rows, class distributions, NaN/Inf counts).

## How to read this for the review

1. Start with **03** (gaps + objectives) to see the "why".
2. Read **01** to understand the data and the central feature-mismatch problem.
3. Read **02** for the literature context and the gap evidence.
4. Read **04** for exactly what we will build and when.

## Quick facts (measured from the downloaded data)

- CIC-IDS-2017: 2,830,743 flows, 77 unique features, 80.3% benign, 1,358 NaN + 4,376 Inf cells.
- UNSW-NB15: 257,673 records, 42 features, 9 attack classes + Normal.
- TON_IoT: 211,043 records, 42 features, 10 attack classes (near-balanced).
- The three datasets share **no common column names**; only ~15–25 features are semantically
  alignable across all three.
