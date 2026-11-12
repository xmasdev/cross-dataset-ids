# BTP — Cross-Dataset Generalization of ML-Based Intrusion Detection

B.Tech. Project-I, Department of Information Technology, NSUT.
Evaluating and improving the cross-dataset generalization of machine-learning intrusion
detection across **CIC-IDS-2017**, **UNSW-NB15** and **TON_IoT**, with **CSE-CIC-IDS2018** as an
external validation set.

## What the project found

1. **Supervised models do not transfer.** Thirteen models reach about 0.995 ROC AUC on the
   dataset they are trained on and 0.42–0.49 on a different one. Six alignment and
   domain-adaptation baselines do not change this.
2. **Why.** 11 of 30 harmonised features point in opposite directions in different datasets,
   benign traffic of two datasets stays perfectly separable after alignment, and even the same
   attack family rarely transfers.
3. **The mid-semester feature-selection result was a selection effect.** Its gain came from
   choosing the subset on the test pairs; chosen without the target dataset, no subset beats all
   features (`experiments/08_supervised_fs_revisited.py`).
4. **BRACE** (Benign-Relative Anomaly scoring with Calibrated Evolutionary selection): do not
   transfer a classifier. On the target network, use a sample of its own benign traffic and its
   own unlabelled stream: benign-anchored features, a kNN distance score on a feature subset
   chosen by a genetic algorithm with a **label-free fitness**, a benign-vs-unlabelled
   classifier, and a calibrated fusion of the two. No source data and no attack label is used.

Full results, limitations included, are in `reports/09_final_report.md` (PDF and DOCX alongside).

## Repository structure

```
.
├── src/xids/                 # end-semester library
│   ├── harmonise.py          # loaders for the four datasets -> one base table
│   ├── features.py           # the 30-feature pool (7 families) and the 12-feature legacy pool
│   ├── anchor.py             # benign-anchored quantile / robust-z / log-standard representations
│   ├── data.py, core.py      # deployment split, target context, label-free fitness
│   ├── knn.py                # GPU kNN scorer
│   ├── detectors.py          # 13 benign-only detectors
│   ├── pu.py                 # benign-vs-unlabelled classifier
│   ├── optimisers.py         # RS, greedy, SA, GA, binary PSO, binary DE, GWO, WOA, NSGA-II
│   ├── models.py             # 13 supervised models, DANN, Deep CORAL
│   └── metrics.py, config.py
├── experiments/              # one script per experiment, each writes CSVs to results/v2/
│   ├── 01_harmonise.py
│   ├── 02_supervised_benchmark.py
│   ├── 03_benign_detectors.py
│   ├── 04_target_side_methods.py     # first pass (exploration)
│   ├── 05_shift_diagnosis.py
│   ├── 06_landscape.py               # exhaustive 12-feature landscape (written, not run)
│   ├── 07_optimisers.py
│   ├── 08_supervised_fs_revisited.py
│   ├── 09_final.py                   # the frozen method and its ablations
│   ├── 10_figures.py, 11_tables.py
│   └── x02_lowprev_probe.py          # development probe on separate seeds
├── results/v2/               # CSV outputs, TABLES.md (every number in the report), EXPERIMENT_LOG.md
├── reports/                  # reports, slides, figures (md + pdf + docx/pptx)
├── tools/build_docs.py       # markdown -> pdf / docx / pptx
├── src/*.py                  # mid-semester scripts (kept for reference)
└── requirements.txt
```

## Datasets

Datasets are not stored in this repository. Download them from the official sources and place
them under `datasets/`.

| Dataset | Official source | Expected location |
|---|---|---|
| CIC-IDS-2017 | https://www.unb.ca/cic/datasets/ids-2017.html | `datasets/CIC-IDS-2017/MachineLearningCVE/` |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `datasets/UNSW-NB15/` |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets | `datasets/TON_IoT network train-test/Train_Test_Network_dataset/` |
| CSE-CIC-IDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html | `datasets/CSE-CIC-IDS2018/` (per-day CSV files) |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/01_harmonise.py
python experiments/02_supervised_benchmark.py
python experiments/05_shift_diagnosis.py
python experiments/08_supervised_fs_revisited.py
python experiments/09_final.py --seeds 5
python experiments/07_optimisers.py --part B --runs 3 --budget 300 --fitness pu_pauc
python experiments/09_final.py --protocol temporal --datasets CIC-IDS-2017 --tag _temporal
python experiments/09_final.py --datasets CSE-CIC-IDS2018 --tag _external
python experiments/03_benign_detectors.py --seeds 2
python experiments/10_figures.py && python experiments/11_tables.py
python tools/build_docs.py reports/09_final_report.md reports/10_final_presentation.md
```

A CUDA GPU is used for the kNN scorer when available; everything also runs on CPU, more slowly.

## Documents

| File | Content |
|---|---|
| `reports/09_final_report.md` | End-semester report |
| `reports/10_final_presentation.md` | End-semester slides |
| `reports/11_final_viva_notes.md` | Plain-language explanation and likely viva questions |
| `reports/01`–`08` | Mid-semester documents (dataset report, literature review, mid-semester report and slides) |
| `results/v2/TABLES.md` | Every table, generated from the result CSVs |

## Team

Shivam Chauhan (2023UIN3323), Shantanu Agarwal (2023UIN3330), Deepesh (2023UIN3358).
Supervisor: Dr. Preeti Bansal.
