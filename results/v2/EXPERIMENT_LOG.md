# End-semester experiment log

Running notes so the work can be resumed from disk. Newest state at the bottom.

## 2026-11-10 — session 1 (stopped early: usage limit)

### Environment
- `.venv` rebuilt for Linux with `uv` (Python 3.12): numpy, pandas, scikit-learn, lightgbm, xgboost,
  shap, umap-learn, torch (CUDA, RTX 4050), pypandoc_binary + typst (for md -> pdf/docx/pptx), python-pptx.
- No sudo, no system pandoc/typst/TeX. Document export must go through the pip packages above.
- Nothing has been committed. The 33 "modified" files in git are CRLF -> LF only.

### Problems found in the mid-semester work (all read from the code)
1. DA-MFS feature search scores subsets on the same six train->test pairs it reports (target labels leak).
2. "Same-dataset" baseline numbers are training accuracy (fit and predict on the same rows).
3. Search space is 4,095 subsets (12 features from 5 raw counters), so exhaustive search is feasible.
4. Mean cross-dataset accuracy 0.508 is below the majority-class predictor (about 0.74).
5. Harmonisation mixed payload bytes (CIC, TON) with header-inclusive bytes (UNSW).
6. Reference [1] carries the arXiv title with the IEEE Access venue; journal title is
   "Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study".
   References [2], [3], [4], [10] were checked and exist. The rest are unchecked.

### Exploratory probes (scripts in the session scratchpad, mid-semester 12-feature data, single seed)
These are exploration only, not report numbers. They must be re-run through `src/xids` with seeds.

- **Probe 1, supervised transfer with LightGBM.** Mean over 6 pairs, balanced accuracy / AUC:
  raw 0.474 / 0.508; marginal quantile alignment 0.493 / 0.509; benign-anchored 0.452 / 0.484.
  Cross-dataset AUC is about 0.5 whatever the alignment. Benign anchoring does not rescue a
  source-trained classifier.
- **Probe 2, detector fitted on 5,000 target-benign flows only (no source data, no attack labels).**
  AUC CIC / TON / UNSW: kNN-5 in benign-anchored space 0.880 / 0.766 / 0.898; isolation forest
  0.826 / 0.823 / 0.727; Mahalanobis 0.772 / 0.882 / 0.678; additive ECOD-style 0.825 / 0.524 / 0.662.
  TPR at 5% FPR for kNN: 0.77 / 0.67 / 0.55.
- **Probe 3, all 4,095 subsets for the kNN detector, 3 repeats.** All-12 vs best subset AUC:
  CIC 0.881 / 0.900, UNSW 0.878 / 0.933, TON 0.772 / 0.907. Spearman of subset rankings between
  datasets: -0.09, +0.02, +0.16. A subset chosen on two datasets lands at percentile 44 / 60 / 82 on
  the third (mean AUC 0.857 vs 0.844 for all features). Subset quality does not transfer.
- **Probe 4, label-free selection on the target (benign calibration vs unlabelled stream), plus a
  benign-vs-unlabelled LightGBM classifier, at natural / 10% / 1% attack prevalence.**
  Was still running when the session stopped. Output: scratchpad `proto4.csv`. RESULT NOT YET SEEN.

### Direction being tested (not yet established)
Supervised boundaries do not transfer; scoring flows against the target's own benign traffic does.
Candidate method: benign-anchored features + kNN distance to a benign calibration set, with the
feature subset chosen by GA/PSO/DE using a fitness that needs no attack labels (AUC between held-out
benign calibration scores and the unlabelled stream, a monotone function of the true AUC), and the
alarm threshold set from held-out benign scores. Theory reference: Blanchard, Lee, Scott,
"Semi-Supervised Novelty Detection", JMLR 2010. Novelty against prior NIDS work still needs a
proper literature check.

### Code written so far (`src/xids/`)
- `config.py`, `harmonise.py` (v2: wire bytes, protocol, service, attack family), `features.py`
  (30-feature pool, 7 families; 12-feature legacy pool), `anchor.py`, `metrics.py`, `data.py`
  (calibration / adapt / test split), `knn.py` (GPU scorer), `optimisers.py`
  (rs, sfs, sa, ga, pso, de, gwo, woa, nsga2).
- `experiments/01_harmonise.py` ran: 2,830,743 / 257,673 / 211,043 rows, 30 features, no non-finite cells.
- Optimisers passed a toy test with a known optimum, except `pso` (0 of 5 runs reached it) which
  needs checking before it is used in a comparison.
- `anchor.py`, `data.py`, `knn.py`, `metrics.py` are written but not yet exercised.

### Data
- CSE-CIC-IDS2018 (public S3 bucket) was downloading into `datasets/CSE-CIC-IDS2018/` for use as an
  external validation set: 9 day files, 2.8 GB, the 4 GB Tuesday file skipped. Check it completed
  (the loop uses `curl -C -`, so re-running resumes). `load_cic2018()` is written but untested.

### Not started
Leak-free supervised benchmark (5-fold CV, more model families, DA baselines), detector comparison,
calibration-size and prevalence sweeps, optimiser comparison, standard feature-selection baselines,
per-attack analysis, SHAP/UMAP, external validation, guidelines, final report, slides, paper draft.

## 2026-11-12 — session 2

- Scratchpad was on a RAM disk and was lost on reboot (probe 4 output gone). All exploration now
  lives in `experiments/` and writes to `results/v2/`.
- CSE-CIC-IDS2018 downloaded completely (9 day files) and harmonised: 8,284,195 rows, 26.2% attacks,
  day groups available. **Held out: no detection result on it may be looked at until the method is frozen.**
- New code: `core.py` (TargetContext + label-free fitness), `models.py` (13 supervised models + DANN,
  Deep CORAL), `detectors.py` (13 benign-only detectors), `pu.py` (benign-vs-unlabelled classifier).
- `05_shift_diagnosis.py` (done): 11 of 30 features flip attack direction between datasets; benign flows
  of different datasets remain perfectly separable (AUC 1.0) after benign anchoring; family-matched
  transfer mostly fails (DoS UNSW->TON 0.095, TON->UNSW 0.204), web attacks transfer (0.93-0.95).
- `02_supervised_benchmark.py` running detached (3 seeds) -> `supervised_benchmark.csv`, log `log_02_supervised.txt`.
  Quick run already showed same-dataset AUC 0.97-0.99 (5-fold CV) vs cross-dataset about 0.5 for every model/alignment.
- `04_target_side_methods.py` first pass running -> `target_side_first.csv`. First numbers (seed 0):
  TON natural: knn_all 0.989, pu 0.998; TON 2%: knn_all 0.989, pu 0.978, fisher 0.993.
  CIC natural: knn_all 0.822, knn+SFS(pu_auc) 0.880, pu 0.979.
- Observation: the wire-byte 12-feature subset gives CIC kNN AUC 0.748 vs 0.88 with payload bytes
  (session-1 probe). Benign-relative detection does not need cross-dataset feature consistency, so a
  "native features" ablation is worth running.
- GPU is the bottleneck (kNN scoring about 20-40 ms per subset when shared). Budgets are sized accordingly.
- References: [12] exists as arXiv:2608.02274 (GLOBECOM venue unverified). Added to related work list:
  Layeghy & Portmann, Comput. Electr. Eng. 108:108692 (2023); Bates et al., Ann. Statist. 51(1):149-178 (2023);
  Blanchard, Lee, Scott, JMLR 11 (2010); Han et al. OWAD, NDSS 2023; Apruzzese et al. arXiv:2203.04686;
  Goix arXiv:1607.01152; Dilworth & Gudla arXiv:2412.06203 (PU learning in cybersecurity survey).

## 2026-11-12 — session 3 (short path requested by the user; experiments complete)

The user asked for a shorter run instead of the full 12-16 h plan. Scope was cut to what is listed here.

### Method frozen: BRACE
kNN (k=5) on GA-selected subset (fitness `pu_pauc`, budget 300) + bagged benign-vs-unlabelled LightGBM
(`pu_bag_fit`: 5 x 100 trees, 7 leaves, min_child 100, no early stopping; chosen on development seed 100 in
`x02_lowprev_probe.py`) + Fisher fusion of conformal p-values. Script: `experiments/09_final.py`.

### Results (all in `results/v2/TABLES.md`, regenerated by `experiments/11_tables.py`)
- Supervised benchmark (3 seeds): same-dataset AUC about 0.995; pair 0.42-0.49; LODO up to 0.61; best alignment 0.60.
- Leak audit (`08`, seeds: raw 2, dom_quantile 2, benign_quantile 1; the job died after 5 of 9 tables and was
  not restarted): oracle subset 0.65-0.72 vs leak-free 0.46-0.58 vs all features 0.47-0.58.
- Final (`final_methods.csv`, 120 runs = 3 targets x n_cal {6000, 20000} x prev {natural, 0.1, 0.02, 0.005} x 5 seeds):
  natural, n_cal 6000: knn_all 0.822/0.929/0.989, knn_ga 0.835/0.948/0.992, pu 0.988/0.973/0.998,
  brace 0.980/0.972/0.998, sup_target 0.998/0.985/1.000. TPR@1%FPR knn_all 0.30/0.49/0.67 -> knn_ga 0.53/0.71/0.79.
  At 0.5% prevalence brace is slightly below knn_all (limit). FPR control holds (1.1% / 5.1%).
- Optimisers (`07` part B, 3 runs, budget 300): all beat random search; GWO/WOA best; GA mid; small differences.
- Temporal CIC-IDS-2017 (calibrate on Monday): brace 0.922 natural, 0.728 at 2%; FPR 6-9% at a 1% target.
- External CSE-CIC-IDS2018 (3 seeds, run once after freezing): random split brace 0.968 (sup 0.981), FPR 1.0%;
  calibrated on day 1: brace 0.796, FPR 20-23% at a 1% target. Benign drift is the main limitation.
- Detectors (`03`, 2 seeds): kNN best on average (0.917-0.919), then GMM 0.910, LOF 0.887.

### Not run (stated as such in the report's limitations)
`06_landscape.py` (exhaustive label-free landscape), `07` part A, n_adapt sweeps, calibration-contamination
robustness, SHAP/UMAP, native/payload-feature ablation, safe-selection rule, NSGA-II Pareto analysis.

### Deliverables
`reports/09_final_report.{md,pdf,docx}` (27 pages), `reports/10_final_presentation.{md,pdf,pptx}` (22 slides),
`reports/11_final_viva_notes.{md,pdf,docx}`, figures `reports/figures/v2_*.png`, `README.md`, `requirements.txt`.
Build: `python tools/build_docs.py <file.md>`; PPTX via `tools/build_pptx.py`.
Open items for the team: replace the placeholder NSUT logo; submission date on the front page is "November 2026";
nothing is committed to git.
