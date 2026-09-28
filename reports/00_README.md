# BTP Project — Cross-Dataset Generalization of ML-Based Intrusion Detection

**Topic:** Evaluating and improving the cross-dataset generalization of machine-learning
intrusion detection systems across CIC-IDS-2017, UNSW-NB15, and TON_IoT, using
metaheuristic/bio-inspired optimization for transferable feature selection.

**Team:** Shivam Chauhan (2023UIN3323), Deepesh (2023UIN3358), Shantanu Agarwal (2023UIN3330)
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
- `src/dataset_stats.json` — measured statistics (rows, class distributions, NaN/Inf counts).

## Quick facts (measured from the downloaded data)

- CIC-IDS-2017: 2,830,743 flows, 77 unique features, 80.3% benign, 1,358 NaN + 4,376 Inf cells.
- UNSW-NB15: 257,673 records, 42 features, 9 attack classes + Normal.
- TON_IoT: 211,043 records, 42 features, 10 attack classes (near-balanced).
- The three datasets share **no common column names**; only ~6 raw features are semantically alignable.

## How to read this for the review

1. Start with **07** (plain-language overview), then **05** (the report).
2. **08** for answering questions.
3. **01**–**04** for the detailed analysis behind the report.
