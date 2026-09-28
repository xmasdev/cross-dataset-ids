# BTP — Cross-Dataset Generalization of ML-Based Intrusion Detection

Evaluating and improving the cross-dataset generalization of machine-learning intrusion
detection systems across **CIC-IDS-2017**, **UNSW-NB15**, and **TON_IoT**, using
metaheuristic/bio-inspired optimization to select features that stay robust across diverse
network environments.

## Repository structure

```
.
├── reports/                     # Project documentation
│   ├── 00_README.md             # Index of the document set
│   ├── 01_dataset_report.md     # Dataset analysis
│   ├── 02_literature_review.md  # Literature review (15 focused papers)
│   ├── 03_objectives_and_gaps.md# Research gap + objectives + scope
│   ├── 04_methodology_and_plan.md# 7-phase methodology & plan
│   ├── 05_midsem_report.md      # Mid-semester report (signable)
│   ├── 06_presentation.md       # Presentation (20 slides) + .pptx/.pdf
│   ├── 07_what_we_are_doing.md  # Plain-language team explainer
│   ├── 08_viva_qa_prep.md       # Likely viva questions + answers
│   ├── figures/                 # Charts embedded in the reports
│   └── *.docx / *.pdf           # Exported versions of each report
├── src/
│   ├── analyze_datasets.py      # Computes dataset statistics from raw CSVs
│   ├── generate_report_assets.py# Regenerates stats JSON + class-distribution figures
│   ├── generate_extra_figures.py# Pipeline, Gantt and shared-subspace figures
│   └── dataset_stats.json       # Measured statistics (rows, classes, NaN/Inf)
├── requirements.txt
└── README.md
```

> **Datasets are not stored in this repository** — they total ~5.5 GB and several files exceed
> GitHub's 100 MB limit. Download them from the official sources below and place them under
> `datasets/` (see `reports/01_dataset_report.md` for the exact expected layout).

## Datasets

| Dataset | Official source | Expected location |
|---|---|---|
| CIC-IDS-2017 | https://www.unb.ca/cic/datasets/ids-2017.html | `datasets/CIC-IDS-2017/MachineLearningCVE/` |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `datasets/UNSW-NB15/` |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets | `datasets/TON_IoT network train-test/Train_Test_Network_dataset/` |

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproducing the dataset statistics and figures

```powershell
.\.venv\Scripts\python.exe src\analyze_datasets.py
.\.venv\Scripts\python.exe src\generate_report_assets.py
```

## Documentation

Start with `reports/03_objectives_and_gaps.md` (why), then `reports/01_dataset_report.md`
(data), `reports/02_literature_review.md` (context), and `reports/04_methodology_and_plan.md`
(what we will build).

## Status

Mid-semester review: dataset analysis, a focused literature review (15 papers), research gaps,
objectives, a detailed plan, the **mid-semester report**, a **20-slide presentation**, a
plain-language explainer, and a viva Q&A guide are complete. Implementation (baselines →
metaheuristic feature selection → analysis) is in progress.
