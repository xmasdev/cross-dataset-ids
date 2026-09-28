# BTP Project — Cross-Dataset Generalization of ML-Based Intrusion Detection

**Topic:** Evaluating and improving the cross-dataset generalization of machine-learning
intrusion detection systems across CIC-IDS-2017, UNSW-NB15, and TON_IoT, using
metaheuristic/bio-inspired optimization for transferable feature selection.

**Team:** Shivam Chauhan (2023UIN3323), Shantanu Agarwal (2023UIN3330), Deepesh (2023UIN3358)
**Supervisor:** Dr. Preeti Bansal — Department of Information Technology, NSUT

---

## Document set

### Submission deliverables (mid-semester)
| # | Document | Purpose |
|---|---|---|
| [05](05_midsem_report.md) | **Mid-Semester Report** (18 pp) | The signable report with all required sections (title page, abstract, introduction, motivation, literature survey, problem statement, objectives, methodology, implementation, results, conclusion, references). |
| [06](06_presentation.md) | **Presentation** (20 slides) | Slide deck for the 10-minute presentation. Exports: `.pptx` (editable), `.pdf` (handout). |

### Preparation / understanding
| # | Document | Purpose |
|---|---|---|
| [07](07_what_we_are_doing.md) | **What We Are Doing (plain language)** | Team explainer: the whole project in simple words, analogies, glossary. |
| [08](08_viva_qa_prep.md) | **Viva Q&A Prep** | 55 likely questions with model answers (including per-paper questions). |

### Supporting analysis
| # | Document | Purpose |
|---|---|---|
| [01](01_dataset_report.md) | Dataset Report | First-hand analysis of the three datasets, feature-schema mismatch, taxonomy mapping. |
| [02](02_literature_review.md) | Literature Review (15 papers) | Focused review with detailed per-paper summaries and the gap-synthesis table. |
| [03](03_objectives_and_gaps.md) | Research Gap, Objectives & Scope | Formal gap statements, novelty, objectives, scope, risks. |
| [04](04_methodology_and_plan.md) | Methodology & Plan | 7-phase plan, evaluation protocol, fitness-function design, timeline. |

Figures are in `figures/`. Each `.md` also has `.docx` and `.pdf` exports.

## Supporting code

- `src/analyze_datasets.py` — computes dataset statistics from the raw CSVs.
- `src/generate_report_assets.py` — regenerates `dataset_stats.json` and the class-distribution figures.
- `src/generate_extra_figures.py` — generates the pipeline, Gantt and shared-subspace figures.
- `src/baseline_cross_dataset.py` — trains the baseline models and produces 3×3 cross-dataset metric matrices.
- `src/aggregate_baseline.py` — aggregates per-model matrices into `results/baseline_summary.md`.
- `src/baseline_figures.py` — generates the baseline-result figures.
- `src/metaheuristic_fs.py` — GA + binary PSO with a **cross-dataset fitness** (raw-feature baseline attempt).
- `src/domain_aligned_fs.py` — **DA-MFS**: domain alignment + GA/PSO feature selection (main result).
- `src/dataset_stats.json` — measured statistics (rows, class distributions, NaN/Inf counts).

## Baseline result (full datasets)

Five ensemble models trained on the **entire** datasets:

| Model | Same-dataset Acc | Cross-dataset Acc | Cross-dataset Bal-Acc |
|---|---|---|---|
| Random Forest | 0.9666 | 0.3726 | 0.4650 |
| XGBoost | 0.9622 | 0.4150 | 0.4760 |
| Hist. Grad. Boosting | 0.9638 | 0.4179 | 0.4941 |
| Extra Trees | 0.9602 | 0.4027 | 0.4672 |
| Soft-voting | 0.9656 | 0.3968 | 0.4624 |

Cross-dataset balanced accuracy ≈ 0.46–0.49 (random chance = 0.50), reproducing the
generalisation gap of [1], [2].

## DA-MFS optimisation result (main contribution)

Plain feature selection on raw features barely helps, because the same feature has a different
distribution in each dataset. **DA-MFS** first aligns each dataset's feature marginals (per-dataset
quantile transform → normal) and then uses a Genetic Algorithm with a **cross-dataset fitness** to
select a transferable subset. Full-data cross-dataset results (mean over 6 train→test pairs):

| Configuration | Accuracy | Balanced acc | Macro-F1 | Attack recall |
|---|---|---|---|---|
| Raw, all 12 features | 0.3726 | 0.4650 | 0.3076 | 0.4789 |
| Aligned, all 12 features | 0.4776 | 0.5298 | 0.3845 | 0.6614 |
| **DA-MFS** (aligned + 4 selected) | **0.5083** | **0.5689** | **0.4604** | **0.6065** |

**Improvement over raw baseline:** accuracy **+13.6 pp**, balanced accuracy **+10.4 pp**,
macro-F1 **+15.3 pp**, attack recall **+12.8 pp**. Selected features:
`duration, src_pkts, total_bytes, total_bytes_per_pkt`.

## Quick facts (measured from the downloaded data)

- CIC-IDS-2017: 2,830,743 flows, 77 unique features, 80.3% benign, 1,358 NaN + 4,376 Inf cells.
- UNSW-NB15: 257,673 records, 42 features, 9 attack classes + Normal.
- TON_IoT: 211,043 records, 42 features, 10 attack classes (near-balanced).
- The three datasets share **no common column names**; only ~6 raw features are semantically alignable.

## How to read this for the review

1. Start with **07** (plain-language overview), then **05** (the report).
2. **08** for answering questions.
3. **01**–**04** for the detailed analysis behind the report.
