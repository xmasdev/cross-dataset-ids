# MASTER CONTEXT — BTP Mid-Semester
## Cross-Dataset Generalization of Machine Learning-Based Intrusion Detection Systems

> **How to use this file:** This is the single, complete source of truth for our B.Tech. Project‑I
> (BTP‑I) mid-semester deliverables. Everything we have done, measured, and concluded is here.
> Give this whole file to an LLM and ask it to produce:
> **(1)** a 15–20 page BTP mid-semester **report** with the exact required sections (listed in §2), and
> **(2)** a 15–20 slide **presentation** matching the required structure.
> **Do not invent numbers** — use only the values in this document. All figures are listed in §12
> (filenames under `reports/figures/`).

---

## 1. Project metadata

- **Title:** Evaluating and Improving Cross-Dataset Generalization of Machine-Learning-Based Intrusion Detection Systems
- **Course:** B.Tech. Project – I (IT1/IT2/ITNS), Mid-Semester Evaluation
- **Department:** Department of Information Technology
- **University:** Netaji Subhas University of Technology (NSUT), Dwarka, New Delhi
- **Supervisor:** Dr. Preeti Bansal
- **Students (in this order):**
  1. Shivam Chauhan — 2023UIN3323
  2. Shantanu Agarwal — 2023UIN3330
  3. Deepesh — 2023UIN3358
- **Submission date for mid-sem report/presentation:** 30 September 2026
- **Code repository:** https://github.com/xmasdev/cross-dataset-ids
- **NSUT logo:** a placeholder image exists at `reports/figures/nsut_logo_placeholder.png` (replace with the official logo).

---

## 2. Mid-semester guidelines (from the official PDF)

- Mid-term evaluation of B.Tech Project‑I presentation: **30 September 2026**.
- **Presentation: 15–20 slides**, 10 minutes, followed by Q&A.
- **Report: 15–20 pages, duly signed by the supervisor**, uploaded to Google Classroom.
- **Report must contain these sections (in order):**
  1. Front page (title, supervisor name, student names, NSUT logo)
  2. Table of contents
  3. Abstract
  4. Introduction
  5. Motivation
  6. Literature survey
  7. Problem statement
  8. Objective
  9. Proposed Methodology
  10. Implementation
  11. Results and Discussion
  12. Conclusion and Future work
  13. References
- The reference presentation used an index slide, introduction, problem statement & objective,
  motivation, work elements, methodology, experimental validation, a literature-review **table**,
  a **Gantt chart**, and references.

---

## 3. One-paragraph summary (abstract)

Machine-learning (ML) based Network Intrusion Detection Systems (NIDS) routinely report
near-perfect accuracy, but almost always when trained and tested on the *same* dataset. When such
a model is deployed on a different network, its performance collapses — often to below 40%, or to
random chance — because it learns dataset-specific artefacts (IPs, ports, collection-window
statistics) rather than genuine attack behaviour. This project studies and reduces this
**cross-dataset generalisation gap** across three structurally different benchmarks —
**CIC-IDS-2017**, **UNSW-NB15**, and **TON_IoT**. We (i) build a *common semantic feature subspace*
by harmonising the three datasets, (ii) measure the gap with five ensemble models on the full
datasets, and (iii) propose **DA-MFS (Domain-Aligned Metaheuristic Feature Selection)**: a
bio-inspired optimiser (Genetic Algorithm / binary Particle Swarm Optimization) that searches for a
feature subset maximising a **cross-dataset fitness** after aligning each dataset's feature
distributions via a per-dataset quantile (rank) transform. On the full data, DA-MFS improves
cross-dataset **accuracy from 0.373 to 0.508 (+13.6 pp)**, **balanced accuracy by +10.4 pp**,
**macro-F1 by +15.3 pp**, and **attack recall by +12.8 pp** over the raw baseline, while using only
**4 of 12** shared features. Extending DA-MFS is the end-semester work.

**Keywords:** intrusion detection, cross-dataset generalisation, domain adaptation, metaheuristic
optimisation, feature selection, CIC-IDS-2017, UNSW-NB15, TON_IoT.

---

## 4. Background and Motivation

### 4.1 Background
- A NIDS monitors network traffic to detect malicious activity. Classic signature-based NIDS only
  catch known attacks; ML-based NIDS learn patterns from labelled traffic and can generalise to
  some unseen attacks.
- The three most-used modern benchmarks are CIC-IDS-2017, UNSW-NB15, and TON_IoT.
- Standard evaluation splits **one** dataset into train/test → accuracy > 99%.

### 4.2 The problem with standard evaluation
- Train and test come from the *same* network, sensors, and collection period → optimistic.
- In reality an IDS is trained on one network (or a vendor dataset) and deployed on a *different*
  network with different topology, services, and traffic statistics.
- Under this realistic setting, performance collapses: Cantone et al. found near-random cross-dataset
  accuracy; Hossain et al. found RF at 95–99% in-dataset but under 40% cross-dataset.

### 4.3 Why it fails
1. **Different feature spaces** — the datasets use different extraction tools (CICFlowMeter vs
   Argus/Bro vs Zeek) so columns don't match.
2. **Shortcut / leakage features** — IPs, ports, timestamps, collection-window counters let the
   model "cheat" and memorise the scenario.
3. **Different distributions** — the same semantic feature has a different marginal distribution in
   each dataset; and attack/benign mixes differ.

### 4.4 Motivation
- Deployment reality: organisations must reuse models trained elsewhere; a non-transferable model
  is useless in practice.
- The gap is severe and well-documented (a collapse, not a small degradation).
- Feature selection is the natural lever, and bio-inspired search is well-suited to it.
- **Key insight we found empirically:** plain feature selection barely helps because of
  *distribution shift*; aligning distributions first, then selecting, gives large gains.

---

## 5. Objectives (verbatim, 5 objectives)

1. **Evaluate** the cross-dataset generalisation of multiple ML/DL models trained on one dataset and
   tested on structurally different ones (CIC-IDS-2017, UNSW-NB15, TON_IoT), in addition to
   same-dataset splits.
2. **Characterise** the sources of the generalisation gap — which attack categories, features, and
   traffic patterns hold up across environments versus which are dataset-specific artefacts.
3. **Explore and compare** metaheuristic/bio-inspired optimisation techniques (Genetic Algorithm,
   Particle Swarm Optimization, and additional swarm/evolutionary methods) for selecting features
   and tuning models **specifically for cross-dataset robustness**.
4. **Benchmark** the optimised approach against standard feature-selection and tuning baselines to
   determine which techniques most effectively narrow the gap.
5. **Consolidate** findings into practical guidelines for building IDS that generalise reliably to
   previously unseen network environments.

---

## 6. Datasets (measured first-hand from the downloaded CSVs)

### 6.1 At a glance

| Property | CIC-IDS-2017 | UNSW-NB15 | TON_IoT (network) |
|---|---|---|---|
| Origin | Canadian Institute for Cybersecurity (UNB) | ACCS, UNSW Canberra | UNSW Canberra (Moustafa et al.) |
| Year | 2017 | 2015 | 2019–2020 |
| Collection tool | CICFlowMeter | Argus + Bro | Zeek (Bro) connection logs |
| Network type | Enterprise/office LAN | Emulated enterprise (IXIA PerfectStorm) | IoT/IIoT testbed (edge–fog–cloud) |
| Total records | 2,830,743 | 257,673 (train 175,341 + test 82,332) | 211,043 |
| Feature columns | 78 numeric (one duplicated → 77 unique) + `Label` | 42 features + `id` + `attack_cat` + `label` | 42 features + `label` + `type` |
| Label scheme | 15 classes (`Label`) | 9 attacks + Normal (`attack_cat`) and binary `label` | 10 classes (`type`) and binary `label` |
| Missing cells | 1,358 NaN + 4,376 Inf | 0 | 0 |
| Benign fraction | ≈80.3% | 36.1% (label=0) | 23.7% (label=0) |

### 6.2 CIC-IDS-2017 — class distribution (all 8 daily files)
BENIGN 2,273,097; DoS Hulk 231,073; PortScan 158,930; DDoS 128,027; DoS GoldenEye 10,293;
FTP-Patator 7,938; SSH-Patator 5,897; DoS slowloris 5,796; DoS Slowhttptest 5,499; Bot 1,966;
Web Attack – Brute Force 1,507; Web Attack – XSS 652; Infiltration 36; Web Attack – Sql Injection 21;
Heartbleed 11.
- Extreme imbalance; Heartbleed (11) and Infiltration (36) are effectively unlearnable.
- Non-finite values concentrated in `Flow Bytes/s` and `Flow Packets/s` (division-by-zero).
- One duplicated column (`Fwd Header Length`); `Destination Port` is an identity/leakage feature.

### 6.3 UNSW-NB15 — attack_cat distribution (train + test)
Normal 93,000; Generic 58,871; Exploits 44,525; Fuzzers 24,246; DoS 16,353; Reconnaissance 13,987;
Analysis 2,677; Backdoor 2,329; Shellcode 1,511; Worms 174.
- Binary split: attack (`label=1`) 164,673 vs Normal (`label=0`) 93,000.
- This copy already excludes the raw identity columns (srcip/sport/dstip/dsport).

### 6.4 TON_IoT — type distribution
normal 50,000; backdoor 20,000; ddos 20,000; dos 20,000; injection 20,000; password 20,000;
ransomware 20,000; scanning 20,000; xss 20,000; mitm 1,043.
- Attack classes near-uniformly balanced (20,000 each), except MITM (1,043).
- Heavy categorical/protocol columns (`conn_state`, `dns_*`, `ssl_*`, `http_*`, `weird_*`).

### 6.5 The central structural finding — feature-schema mismatch
- The three datasets share **no common column name**. Only ~6 raw features are semantically
  comparable.
- This mismatch is the root cause of cross-dataset failure and the focus of the project.

---

## 7. Feature harmonisation (the shared subspace)

Because the datasets use different extractors, we map semantically equivalent fields into one common
numeric subspace. We implemented a **12-feature shared subspace**:

| Semantic concept | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| Duration (s) | `Flow Duration` (/1e6) | `dur` | `duration` |
| Source bytes | `Total Length of Fwd Packets` | `sbytes` | `src_bytes` |
| Destination bytes | `Total Length of Bwd Packets` | `dbytes` | `dst_bytes` |
| Source packets | `Total Fwd Packets` | `spkts` | `src_pkts` |
| Destination packets | `Total Backward Packets` | `dpkts` | `dst_pkts` |

The final 12 features used in all experiments (raw + derived):
`duration, src_bytes, dst_bytes, src_pkts, dst_pkts, total_bytes, total_pkts,
bytes_per_pkt_src, bytes_per_pkt_dst, total_bytes_per_pkt, src_dst_byte_ratio, src_dst_pkt_ratio`
(derived: total_bytes = src+dst, total_pkts = src+dst, and per-packet/ratio features).
Identity/leakage features (IPs, ports, IDs) are excluded.

---

## 8. Literature survey (15 core papers + 1 method reference)

### 8.1 Theme A — Cross-dataset generalisation (the core problem)
- **[1] Cantone, Marrocco & Bria (2024), IEEE Access — "On the Cross-Dataset Generalization of ML
  for NIDS".** Four classifiers (DT, RF, KNN, MLP) across four datasets (CIC-IDS-2017,
  CSE-CIC-IDS2018, LycoS-IDS2017, LycoS-Unicas-IDS2018) with a restricted common feature set.
  Same-dataset ≈ 99%; cross-dataset ≈ random chance. Diagnosis only; no mitigation.
- **[2] Hossain et al. (2026) — "Assessing Generalisation Capability of ML Models for Intrusion
  Detection".** RF/LR/NB on UNSW-NB15 ↔ TON_IoT. RF: 95.08% (UNSW) and 99.79% (TON) in-dataset;
  **below 40%** cross-dataset. Observational; no feature selection/mitigation.
- **[3] Hakim, Uddin & Anis (2026) — "Cross-Domain Generalization Failure in Lightweight IDS for
  IIoT".** Four lightweight models across three IIoT datasets; SHAP analysis. Models rely on a
  **port-category shortcut** (96–435× more common in source attacks); the evaluation protocol can
  reverse conclusions; adversarial robustness is unrelated to transfer.
- **[4] Butt, Hotho & Schlör (2026) — "Evaluating Tabular Representation Learning for NIDS".**
  TabICL/autoencoders/transformers; strong dataset–model dependency; transfer varies by pair.

### 8.2 Theme B — Benchmark datasets
- **[5] Sharafaldin, Lashkari & Ghorbani (2018) — CIC-IDS-2017 dataset.**
- **[6] Moustafa & Slay (2015) — UNSW-NB15 dataset.**
- **[7] Moustafa (2021) — TON_IoT datasets.**
- **[14] Ring et al. (2019) — "A Survey of Network-Based Intrusion Detection Data Sets".** Explains
  why datasets differ in collection environment/features/labels.

### 8.3 Theme C — Metaheuristic / bio-inspired feature selection for IDS
- **[8] Fatima & Ali (2022)** — wrapper **Genetic Algorithm** + stacking/bagging on UNSW-NB15 and
  CICDDoS2019. Strong results, but fitness is **same-dataset**.
- **[9] Suman, Tripathy & Saha (2019)** — **NSGA-II** multi-objective unsupervised feature selection
  on KDD-99/NSL-KDD/Kyoto; 99.78% / 99.83% / 99.65%.
- **[10] Li (2026) — MPDGGA** — multi-population diversity-guided GA; best accuracy on 10/11 datasets.
- **[11] Mojtahedi et al. (2022)** — hybrid **Whale Optimization + GA** with KNN (KDDCUP1999).
- **[12] Yang et al. (2026)** — **NSGA-III** AutoML (LightGBM) for weighted F1, latency, model size.

### 8.4 Theme D — Domain adaptation / transfer
- **[13] Layeghy, Baktashmotlagh & Portmann (2022/23) — DI-NIDS.** Adversarial domain adaptation
  (DANN) + One-Class SVM; improves cross-domain but needs aligned feature spaces and target data.

### 8.5 Theme E — Evaluation / dataset-quality critique
- **[15] Vitorino et al. (2024)** — corrected CICIDS2017 ("NewCICIDS") changes model rankings;
  justifies cleaning and removing leakage features.
- **[16] Virmaux et al. (2021) — KRDA (Knothe–Rosenblatt / quantile transport)** — quantile-based
  tabular domain adaptation (method reference for our alignment step).

### 8.6 Literature summary table (use this in the report/presentation)

| # | Title | Author(s) | What it does | Conclusion |
|---|---|---|---|---|
| 1 | Cross-Dataset Generalization of ML for NIDS | Cantone et al., 2024 | Cross-dataset benchmark (4 sets) | Near-chance cross-dataset; diagnosis only |
| 2 | Assessing Generalisation Capability | Hossain et al., 2026 | RF/LR/NB on UNSW ↔ TON | <40% cross-dataset vs 95–99% in-dataset |
| 3 | Cross-Domain Failure in Lightweight IDS | Hakim et al., 2026 | 3 IIoT sets + SHAP | Port shortcut; protocol reverses results |
| 4 | Tabular Representation Learning for NIDS | Butt et al., 2026 | Representation learning | Transfer depends on pair |
| 5 | CIC-IDS-2017 | Sharafaldin et al., 2018 | Dataset | Primary dataset |
| 6 | UNSW-NB15 | Moustafa & Slay, 2015 | Dataset | Second dataset |
| 7 | TON_IoT | Moustafa, 2021 | Dataset | Third dataset (Zeek) |
| 8 | Metaheuristic classifiers for ID | Fatima & Ali, 2022 | GA FS + ensembles | Strong single-dataset; no transfer |
| 9 | Unsupervised FS via NSGA-II | Suman et al., 2019 | Multi-objective FS | Pareto subsets; single-dataset |
| 10 | MPDGGA | Li, 2026 | Multi-population GA FS | Best 10/11 datasets; single-dataset |
| 11 | GA + Whale Optimization FS | Mojtahedi et al., 2022 | Hybrid bio-inspired FS | Swarm hybridisation viable |
| 12 | NSGA-III AutoML IDS | Yang et al., 2026 | Multi-objective AutoML | Efficiency, not transfer |
| 13 | DI-NIDS | Layeghy et al., 2022 | Domain-invariant features | Better cross-domain; needs target data |
| 14 | Survey of NIDS Data Sets | Ring et al., 2019 | Dataset taxonomy | Explains heterogeneity |
| 15 | Adversarial Robustness (NewCICIDS) | Vitorino et al., 2024 | Corrected CICIDS2017 | Corrected data changes conclusions |
| 16 | KRDA (quantile transport) | Virmaux et al., 2021 | Tabular domain adaptation | Quantile alignment method used by us |

### 8.7 Research gap
| Group | Cross-dataset eval? | Metaheuristic FS? | Optimised **for transfer**? |
|---|---|---|---|
| Diagnostic studies [1]–[4] | Yes | No | No |
| Metaheuristic FS [8]–[12] | No (single dataset) | Yes | **No (same-dataset fitness)** |
| Domain adaptation [13] | Yes | No | No (needs target data) |
| **This project** | **Yes (3 datasets)** | **Yes (GA, PSO)** | **Yes** |

**Gap statement:** arXiv searches return **0 results** for `domain adaptation + feature selection +
intrusion detection` and for `domain-invariant + feature selection + genetic/evolutionary`. No prior
work combines domain alignment with metaheuristic feature selection under a cross-dataset objective
for NIDS. This is our contribution.

---

## 9. Proposed Methodology (7 phases)

1. **Data cleaning & preprocessing** — strip whitespace headers; remove duplicated column; fix
   NaN/Inf; normalise labels; drop identity/leakage features; encode categoricals; per-dataset scaling.
2. **Feature harmonisation** — build the shared 12-feature semantic subspace.
3. **Baseline benchmark** — train on source, test on target (3×3 matrix), same-dataset vs cross-dataset.
4. **Metaheuristic feature selection for transfer** — GA/PSO binary mask; **fitness = cross-dataset
   performance** (not same-dataset accuracy).
5. **Baseline comparison** — mutual information, RF importance, PCA, RFE, LASSO at matched subset size.
6. **Analysis** — feature/attack transfer analysis (SHAP/UMAP), ablations.
7. **Guidelines** — practical recommendations for generalisable IDS.

**Evaluation protocol:** binary classification (benign vs attack); models RF, Extra Trees, HistGB,
XGBoost, soft-voting; metrics = accuracy, balanced accuracy, macro-F1, attack recall; strict
train-on-source/test-on-target.

---

## 10. Baseline experiment (implemented, full datasets)

### 10.1 Setup
- Data: full harmonised datasets (2,830,743 / 257,673 / 211,043 rows), 12 shared features.
- Models: Random Forest (200 trees), Extra Trees (200), HistGradientBoosting (300 iters),
  XGBoost (400 trees, hist), soft-voting ensemble. All `min_samples_leaf=5`, balanced class weights.
- Protocol: train on each dataset, test on all three → 3×3 matrices. Seed 42.
- Script: `src/baseline_cross_dataset.py`; aggregation `src/aggregate_baseline.py`; figures `src/baseline_figures.py`.

### 10.2 Summary table (mean over datasets / cross pairs)

| Model | Same-dataset Acc | Cross-dataset Acc | Same-dataset Bal-Acc | Cross-dataset Bal-Acc | Same-dataset Macro-F1 | Cross-dataset Macro-F1 |
|---|---|---|---|---|---|---|
| Extra Trees (et) | 0.9602 | 0.4027 | 0.9531 | 0.4672 | 0.9508 | 0.3444 |
| HistGB (hgb) | 0.9638 | 0.4179 | 0.9510 | 0.4941 | 0.9549 | 0.3799 |
| Random Forest (rf) | 0.9666 | 0.3726 | 0.9597 | 0.4650 | 0.9581 | 0.3076 |
| Soft-voting (vote) | 0.9656 | 0.3968 | 0.9558 | 0.4624 | 0.9573 | 0.3453 |
| XGBoost (xgb) | 0.9622 | 0.4150 | 0.9552 | 0.4760 | 0.9534 | 0.3597 |

### 10.3 Per-model 3×3 ACCURACY matrices (rows = trained on, cols = tested on)

Extra Trees:
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.9803 | 0.3610 | 0.4393 |
| UNSW | 0.2802 | 0.9314 | 0.6869 |
| TON | 0.3757 | 0.2728 | 0.9690 |

HistGB:
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.9831 | 0.3591 | 0.4399 |
| UNSW | 0.4383 | 0.9350 | 0.4633 |
| TON | 0.3360 | 0.4709 | 0.9732 |

Random Forest:
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.9817 | 0.3610 | 0.2618 |
| UNSW | 0.2630 | 0.9457 | 0.6601 |
| TON | 0.4038 | 0.2858 | 0.9723 |

Soft-voting:
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.9837 | 0.3609 | 0.4398 |
| UNSW | 0.2765 | 0.9402 | 0.5289 |
| TON | 0.3359 | 0.4386 | 0.9731 |

XGBoost:
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.9807 | 0.3610 | 0.4429 |
| UNSW | 0.2874 | 0.9332 | 0.6515 |
| TON | 0.3533 | 0.3942 | 0.9727 |

### 10.4 Discussion
- Same-dataset accuracy 0.96–0.97; cross-dataset only 0.37–0.42.
- Balanced accuracy cross-dataset 0.46–0.49 (≈random); several cells exactly 0.500 (e.g. XGBoost
  CIC→UNSW = 0.5000).
- **Ensembling does not fix the gap** → the problem is the features/distribution, not model capacity.
- Asymmetric: UNSW→TON transfers better than TON→UNSW.
- Reproduces Cantone et al. [1] and Hossain et al. [2].

---

## 11. DA-MFS — Domain-Aligned Metaheuristic Feature Selection (main contribution)

### 11.1 Motivation
A first attempt — a GA over the **raw** shared features (script `src/metaheuristic_fs.py`) —
improved cross-dataset accuracy only marginally (0.3726 → 0.4071; macro-F1 0.3076 → 0.3980;
balanced accuracy 0.4650 → 0.4714; recall 0.4789 → 0.4361). The gain was within noise. We concluded
the bottleneck is **distribution shift**, not merely feature choice.

### 11.2 Method (two stages)
1. **Domain alignment.** For each dataset independently, apply a **QuantileTransformer**
   (`output_distribution="normal"`, 1000 quantiles) to the 12 shared features, aligning every
   feature's marginal distribution to a standard normal. (In the spirit of quantile/Knothe–Rosenblatt
   transport for tabular domain adaptation [16].)
2. **Metaheuristic selection.** A **Genetic Algorithm** (binary mask, pop 24, 15 generations;
   tournament selection, uniform crossover, bit-flip mutation, elitism) and a **binary Particle
   Swarm Optimization** (20 particles, 15 iterations) search for the subset of aligned features that
   maximises the **cross-dataset fitness** = mean over the 6 source→target pairs of
   `0.5·balanced-accuracy + 0.5·macro-F1`, evaluated with a fast Random Forest on a stratified
   20,000-row sample per dataset. The winning subset is then validated on the **full** datasets.

**Selected subset (GA, best):** `duration, src_pkts, total_bytes, total_bytes_per_pkt` (**4 features**).
GA sample fitness 0.5508 vs PSO 0.5447 (baseline aligned all-features fitness 0.4808).

### 11.3 Result table (FULL data, mean over 6 cross-dataset pairs)

| Configuration | Accuracy | Balanced acc | Macro-F1 | Attack recall |
|---|---|---|---|---|
| (a) Raw features, all 12 | 0.3726 | 0.4650 | 0.3076 | 0.4789 |
| (b) Aligned features, all 12 | 0.4776 | 0.5298 | 0.3845 | 0.6614 |
| (c) **DA-MFS** (aligned + 4 selected) | **0.5083** | **0.5689** | **0.4604** | **0.6065** |

**Improvement of DA-MFS over raw baseline:** accuracy **+13.58 pp**, balanced accuracy **+10.39 pp**,
macro-F1 **+15.27 pp**, attack recall **+12.76 pp**.

### 11.4 DA-MFS 3×3 matrices (rows = trained on, cols = tested on)

**Accuracy — raw baseline:**
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.982 | 0.361 | 0.262 |
| UNSW | 0.263 | 0.946 | 0.660 |
| TON | 0.404 | 0.286 | 0.972 |

**Accuracy — DA-MFS:**
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.980 | 0.374 | 0.506 |
| UNSW | 0.431 | 0.946 | 0.590 |
| TON | 0.458 | 0.691 | 0.947 |

**Balanced accuracy — raw baseline:**
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.984 | 0.500 | 0.516 |
| UNSW | 0.490 | 0.952 | 0.442 |
| TON | 0.561 | 0.281 | 0.943 |

**Balanced accuracy — DA-MFS:**
| | CIC | UNSW | TON |
|---|---|---|---|
| CIC | 0.983 | 0.506 | 0.636 |
| UNSW | 0.629 | 0.952 | 0.478 |
| TON | 0.547 | 0.619 | 0.926 |

**Per-pair change (accuracy):** CIC→UNSW 0.361→0.374 (+0.013); CIC→TON 0.262→0.506 (+0.244);
UNSW→CIC 0.263→0.431 (+0.168); UNSW→TON 0.660→0.590 (−0.070); TON→CIC 0.404→0.458 (+0.054);
TON→UNSW 0.286→0.691 (+0.405). **5 of 6 pairs improve.**

Additional matrices (macro-F1 and attack recall, all three configs) are in
`results/da_mfs_matrices.md`.

### 11.5 Discussion
- **Large, consistent improvement** across every metric; balanced accuracy crosses from below to
  clearly above the 0.5 random line.
- **Both components contribute:** alignment alone gives +10.5 pp accuracy; the metaheuristic adds
  +3.1 pp more and shrinks the representation to 4 features.
- **Recall now improves** (+12.8 pp), resolving the recall regression of the plain-GA attempt.
- **Novelty:** no prior work couples quantile-based domain alignment with metaheuristic feature
  selection under a cross-dataset fitness for NIDS.
- **Honest caveat:** absolute accuracy ~51% is still far from deployment; one pair (UNSW→TON
  accuracy) dips slightly. Single-seed result; multi-seed validation is end-semester work.

---

## 12. Figures (all under `reports/figures/`)

| File | What it shows |
|---|---|
| `fig_pipeline.png` | Proposed end-to-end 7-phase methodology (flow chart) |
| `fig_shared_subspace.png` | Shared vs dataset-specific features (schema mismatch) |
| `fig_feature_counts.png` | Feature count per dataset (78 / 42 / 42) |
| `fig_cicids_class_dist.png` | CIC-IDS-2017 class distribution (log scale) |
| `fig_unsw_class_dist.png` | UNSW-NB15 class distribution (log scale) |
| `fig_ton_class_dist.png` | TON_IoT class distribution (log scale) |
| `fig_baseline_same_vs_cross.png` | Same-dataset vs cross-dataset performance for all 5 baseline models |
| `fig_baseline_heatmaps.png` | 3×3 cross-dataset accuracy heatmaps for all 5 models |
| `fig_baseline_xgb_balanced.png` | XGBoost balanced-accuracy 3×3 heatmap |
| `fig_meta_convergence.png` | Plain GA/PSO convergence (first raw-feature attempt) |
| `fig_meta_before_after.png` | Plain GA result (first attempt) |
| `fig_damfs_summary.png` | **Raw vs aligned vs DA-MFS** cross-dataset bar chart (main result) |
| `fig_damfs_convergence.png` | DA-MFS GA/PSO convergence |
| `fig_gantt.png` | Project Gantt chart |
| `nsut_logo_placeholder.png` | Placeholder NSUT logo |

---

## 13. Code, scripts, and repository

Repository: https://github.com/xmasdev/cross-dataset-ids

`src/` scripts:
- `analyze_datasets.py` — computes dataset statistics from raw CSVs.
- `generate_report_assets.py` — regenerates `dataset_stats.json` + class-distribution figures.
- `generate_extra_figures.py` — pipeline, Gantt, shared-subspace figures.
- `baseline_cross_dataset.py` — trains 5 baseline models, produces 3×3 matrices.
- `aggregate_baseline.py` — aggregates per-model matrices into a summary.
- `baseline_figures.py` — baseline result figures.
- `metaheuristic_fs.py` — plain GA + binary PSO with cross-dataset fitness (first attempt).
- `domain_aligned_fs.py` — **DA-MFS** (domain alignment + GA/PSO) — main result.
- `da_mfs_tables.py` — renders the 3×3 before/after matrices.

`results/`: `baseline_summary.md`, `baseline_long_all.csv`, per-model matrices,
`metaheuristic_summary.md`, `da_mfs_summary.md`, `da_mfs_results.csv`, `da_mfs_matrices.md`,
`da_mfs_search.json`.

Environment: Python venv with pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.
Reproduce: `python src/baseline_cross_dataset.py --models rf,et,hgb,xgb,vote` then
`python src/domain_aligned_fs.py`.

---

## 14. Key numbers cheat sheet (for the writer)

- Datasets: 2,830,743 / 257,673 / 211,043 records.
- Shared feature subspace: 12 features; DA-MFS selected 4.
- Baseline: same-dataset 0.96–0.97, cross-dataset 0.37–0.42, cross balanced acc 0.46–0.49 (~random).
- DA-MFS: accuracy 0.373→0.508 (**+13.6 pp**); balanced acc 0.465→0.569 (+10.4 pp);
  macro-F1 0.308→0.460 (+15.3 pp); attack recall 0.479→0.607 (+12.8 pp). 5 of 6 pairs improved.
- Alignment alone: 0.373→0.478 accuracy.
- GA selected: `duration, src_pkts, total_bytes, total_bytes_per_pkt`.
- Novelty: 0 prior arXiv results combining domain adaptation + metaheuristic FS for NIDS.

---

## 15. Likely viva questions (condensed answers)

- **What is your project?** Study why ML intrusion detectors fail on new networks and use
  bio-inspired optimisation to pick features that transfer; our DA-MFS method improves cross-dataset
  accuracy by +13.6 pp and recall by +12.8 pp.
- **Why cross-dataset?** Real IDS are trained on one network and deployed on another; same-dataset
  99% accuracy is misleading.
- **What is novel?** Domain alignment (quantile transform) combined with metaheuristic feature
  selection under a cross-dataset fitness — no prior NIDS work does this.
- **Why not just use all features?** Features have different distributions/meanings per dataset;
  raw selection barely helps; alignment + selection does.
- **What is the fitness function?** Mean over 6 source→target pairs of 0.5·balanced-accuracy +
  0.5·macro-F1 on aligned features.
- **Why GA and PSO?** Combinatorial feature-subset search; nature-inspired; GA slightly beat PSO.
- **Metrics?** Accuracy, balanced accuracy, macro-F1, attack recall (accuracy alone misleads due to
  imbalance).
- **Limitations?** ~51% absolute accuracy still not deployable; single seed; one pair slightly worse;
  recall-aware/multi-objective and multi-seed validation are future work.

---

## 16. References (IEEE style)

[1] M. Cantone, C. Marrocco, and A. Bria, "On the cross-dataset generalization of machine learning
for network intrusion detection," *IEEE Access*, vol. 12, pp. 144489–144508, 2024.

[2] M. Z. Hossain, M. A. R. Khan, M. R. Islam, S. M. S. Islam, and T. Gedeon, "Assessing
generalisation capability of machine learning models for intrusion detection," arXiv:2605.04407, 2026.

[3] M. A. Hakim, M. S. Uddin, and T. I. Anis, "Cross-domain generalization failure in lightweight
intrusion detection models for IIoT networks," arXiv:2607.00553, 2026.

[4] M. U. Butt, A. Hotho, and D. Schlör, "Evaluating tabular representation learning for network
intrusion detection," in *Proc. IEEE CSR*, 2026.

[5] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection
dataset and intrusion traffic characterization," in *Proc. ICISSP*, 2018.

[6] N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection
systems," in *Proc. MilCIS*, 2015.

[7] N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the
edge: Network TON_IoT datasets," *Sustain. Cities Soc.*, vol. 72, 2021.

[8] Z. Fatima and A. Ali, "Effective metaheuristic based classifiers for multiclass intrusion
detection," arXiv:2210.02678, 2022.

[9] C. Suman, S. Tripathy, and S. Saha, "Building an effective intrusion detection system using
unsupervised feature selection in multi-objective optimization framework," arXiv:1905.06562, 2019.

[10] C. Li, "Multi-population diversity-guided genetic algorithm for feature selection in network
intrusion detection," arXiv:2605.19864, 2026.

[11] A. Mojtahedi et al., "Feature selection-based intrusion detection system using genetic whale
optimization algorithm and sample-based classification," arXiv:2201.00584, 2022.

[12] L. Yang et al., "A multi-objective AutoML-based efficient intrusion detection system for EV
charging networks," in *Proc. IEEE GLOBECOM*, 2026.

[13] S. Layeghy, M. Baktashmotlagh, and M. Portmann, "DI-NIDS: Domain invariant network intrusion
detection system," *Knowledge-Based Systems*, 2023.

[14] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, "A survey of network-based
intrusion detection data sets," *Computers & Security*, vol. 86, pp. 147–167, 2019.

[15] J. Vitorino, M. Silva, E. Maia, and I. Praça, "An adversarial robustness benchmark for
enterprise network intrusion detection," in *Proc. FPS*, 2023.

[16] A. Virmaux, I. Saffar, J. Zhang, and B. Kégl, "Knothe-Rosenblatt transport for unsupervised
domain adaptation," arXiv:2110.02716, 2021.

---

### END OF MASTER CONTEXT
