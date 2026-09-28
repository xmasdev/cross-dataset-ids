# Literature Review — Cross-Dataset Generalization of ML-Based Intrusion Detection

**Project:** Evaluating and Improving Cross-Dataset Generalization of ML-Based Intrusion Detection Systems
**Document:** 02 — Literature Review (focused set of **15 core papers**)
**Citation style:** IEEE (numbered)

---

## 1. How this review is organised

This review covers **15 core papers** divided into five themes. Each paper is summarised
using the same structure — **problem, method, datasets, key results, limitation** — followed by
*"what it means for our project"* so the team can answer questions on any single paper.

**Theme A — Cross-dataset generalization (the core problem):** [1]–[4]
**Theme B — Benchmark datasets:** [5]–[7], [14]
**Theme C — Metaheuristic / bio-inspired feature selection:** [8]–[12]
**Theme D — Domain adaptation & transfer learning:** [13]
**Theme E — Evaluation and dataset-quality critique:** [15]

---

## 2. Theme A — Cross-Dataset Generalization

### [1] Cantone, Marrocco & Bria (2024) — *On the Cross-Dataset Generalization of ML for NIDS*, IEEE Access

- **Problem:** Do ML-based NIDS still work when trained on one network and tested on a
  different one?
- **Method:** Four classifiers (Decision Tree, Random Forest, KNN, MLP) are trained and tested
  across **four** datasets (CIC-IDS-2017, CSE-CIC-IDS2018, LycoS-IDS2017, LycoS-Unicas-IDS2018)
  using a **restricted common feature representation**; the authors also introduce a corrected
  version of CSE-CIC-IDS2018 and use data visualisation to diagnose failures.
- **Datasets:** CIC-IDS-2017, CSE-CIC-IDS2018, LycoS-IDS2017, LycoS-Unicas-IDS2018.
- **Key results:** (i) near-perfect accuracy when training and testing on the same dataset
  (~0.99); (ii) when tested cross-dataset, accuracy falls to roughly **random chance** for most
  source→target pairs; (iii) failures are caused by **dataset-specific artefacts** — identity
  fields (IP addresses, ports) and collection methodology — not by weak models.
- **Limitation:** It only *diagnoses* the problem. It does **not** propose or optimise a fix —
  it uses a fixed common feature set and default hyper-parameters.
- **What it means for us:** This is our closest prior work and our baseline. Our project
  *extends* it by adding a **metaheuristic that selects features for transfer** instead of using
  a fixed subset, and by using a third structurally different dataset (TON_IoT).

### [2] Hossain et al. (2026) — *Assessing Generalisation Capability of ML Models for Intrusion Detection*

- **Problem:** Does high accuracy within one dataset imply reliability on an unseen dataset?
- **Method:** Random Forest, Logistic Regression, and Naïve Bayes evaluated under both
  same-dataset and cross-dataset settings on **UNSW-NB15 ↔ TON_IoT**.
- **Datasets:** UNSW-NB15 and TON_IoT (two of our three datasets).
- **Key results:** Random Forest reaches **95.08%** accuracy on UNSW-NB15 and **99.79%** on
  TON_IoT in-dataset, but cross-dataset accuracy drops **below 40%** in both directions — a
  very large generalisation gap.
- **Limitation:** Purely observational. No feature selection, no optimisation, no mitigation,
  and no CIC-IDS-2017.
- **What it means for us:** This is almost a subset of our project. It proves the gap exists on
  UNSW/TON; we add CIC-IDS-2017 and, crucially, attempt to *close* the gap with transfer-
  oriented optimisation. We must cite and differentiate from it.

### [3] Hakim, Uddin & Anis (2026) — *Cross-Domain Generalization Failure in Lightweight IDS for IIoT*

- **Problem:** Do lightweight IDS models transfer to structurally different IIoT networks?
- **Method:** Four lightweight models are trained on one IIoT source and tested (no retraining)
  on two targets, using a feature representation restricted to attributes common to all three;
  SHAP-style explainability, plus adversarial-robustness and limited-adaptation probes.
- **Key results:** (i) the best models rely overwhelmingly on **coarse port-category
  features**, which appear **96–435× more often** in the source attack traffic than in the
  targets — a **shortcut**, not a genuine attack signature; (ii) the evaluation protocol
  (how class imbalance is handled) can **reverse** which target looks harder; (iii) adversarial
  robustness is **unrelated** to cross-network generalisation.
- **Limitation:** Again diagnostic; it does not select features for transfer.
- **What it means for us:** Directly shapes our design — we must **remove port/identity
  features** and evaluate under **realistic class distributions** to avoid the shortcut trap.

### [4] Butt, Hotho & Schlör (2026) — *Evaluating Tabular Representation Learning for NIDS*

- **Problem:** Can learned representations (instead of hand-crafted features) transfer better
  across networks?
- **Method:** Systematic evaluation of state-of-the-art tabular representation-learning methods
  (e.g. TabICL), autoencoders, and end-to-end transformers on several NetFlow datasets, with
  comprehensive hyper-parameter search and cross-dataset transfer experiments.
- **Key results:** Strong **dataset–model dependency** — no method dominates everywhere; cross-
  dataset transfer can work, but performance varies greatly depending on the source–target pair.
- **Limitation:** Model/representation-centric; no feature selection; no optimisation targeting
  transfer.
- **What it means for us:** Supports our premise that *what we feed the model* matters as much
  as the model itself, and warns us to report per-pair results rather than a single average.

---

## 3. Theme B — Benchmark Datasets

### [5] Sharafaldin, Lashkari & Ghorbani (2018) — CIC-IDS-2017

- **Contribution:** Introduces CIC-IDS-2017 (CICFlowMeter, 78 flow features, 5 days of
  realistic enterprise traffic, 14 attack types), with a documented testbed and attack
  schedule.
- **Key facts:** Realistic benign traffic via B-Profile; flows extracted with CICFlowMeter;
  the most-used NIDS benchmark to date.
- **Known issues (from later work):** identity/port features inflate same-dataset scores;
  extreme class imbalance; occasional duplicate/NaN/Inf values.
- **What it means for us:** Primary **source** dataset; we use its flow statistics and map them
  to the shared subspace.

### [6] Moustafa & Slay (2015) — UNSW-NB15

- **Contribution:** Introduces UNSW-NB15 (IXIA PerfectStorm generator; Argus + Bro features;
  9 attack families + Normal) plus a fixed train/test split.
- **Key facts:** 45-column public split; both binary `label` and 9-class `attack_cat`.
- **What it means for us:** Second dataset; its `ct_*` time-window features are exactly the kind
  of collection-dependent signal we expect not to transfer.

### [7] Moustafa (2021) — TON_IoT

- **Contribution:** Introduces the TON_IoT IoT/IIoT datasets collected from a three-layer
  (edge–fog–cloud) SDN/NFV testbed; the network portion uses raw Zeek (Bro) log fields.
- **Key facts:** 10 attack classes (near-uniformly balanced), heavy categorical/protocol fields.
- **What it means for us:** Third dataset and the most realistic *target* domain (rawest
  schema); its balanced classes make it ideal for clean macro-metric evaluation.

### [14] Ring et al. (2019) — *A Survey of Network-Based Intrusion Detection Data Sets*

- **Contribution:** A comprehensive taxonomy and comparison of NIDS datasets, covering
  collection environment, feature semantics, labelling method, and reuse.
- **Key facts:** Shows datasets differ radically in how they were built — which is the root
  cause of cross-dataset mismatch.
- **What it means for us:** Justifies why we must build a **common feature subspace** before
  comparing datasets.

---

## 4. Theme C — Metaheuristic / Bio-Inspired Feature Selection for IDS

### [8] Fatima & Ali (2022) — GA wrapper features + ensembles

- **Problem:** Too many features slow detection and hurt accuracy.
- **Method:** Wrapper **Genetic Algorithm (GA)** feature selection combined with stacking and
  bagging ensembles for **multi-class** detection.
- **Datasets:** UNSW-NB15 and CICDDoS2019.
- **Key results:** GA + stacking improves accuracy, detection rate, and lowers false-alarm rate
  versus baselines.
- **Limitation:** Fitness is **same-dataset** accuracy — the subset is optimised for the
  training distribution, so transfer is not guaranteed.
- **What it means for us:** This is the exact pattern we invert: we keep a GA but change its
  **fitness to cross-dataset performance**.

### [9] Suman, Tripathy & Saha (2019) — NSGA-II unsupervised feature selection

- **Method:** **NSGA-II** multi-objective optimisation of three information-theoretic measures
  (mutual information, standard deviation, information gain) for *unsupervised* selection; the
  Pareto-optimal subsets feed SVM/DT/KNN.
- **Datasets:** KDD-99, NSL-KDD, Kyoto 2006+.
- **Key results:** 99.78% (KDD-99), 99.83% (NSL-KDD 20% test), 99.65% (Kyoto).
- **Limitation:** Legacy datasets; single-dataset evaluation; no transfer objective.
- **What it means for us:** Motivates our optional **multi-objective** variant
  (cross-dataset F1 vs number of features).

### [10] Li (2026) — MPDGGA

- **Method:** A multi-population, diversity-guided GA with an information-gain-ratio operator
  to maintain population diversity when selecting features from high-dimensional traffic.
- **Datasets:** NSL-KDD, UNSW-NB15, and 9 UCI datasets.
- **Key results:** Best accuracy on 10/11 datasets while selecting very few features
  (≥2.26% of features).
- **Limitation:** Still same-dataset accuracy; no cross-dataset protocol.
- **What it means for us:** Shows modern GA variants can be very compact — useful when our
  shared subspace is small.

### [11] Mojtahedi et al. (2022) — GA + Whale Optimization hybrid

- **Method:** Hybrid **Whale Optimization Algorithm (WOA) + GA** feature selection with KNN.
- **Datasets:** KDDCUP1999.
- **Key results:** Better accuracy than prior methods by extracting class-relevant features.
- **Limitation:** Legacy dataset; accuracy-only fitness; single domain.
- **What it means for us:** Demonstrates that **swarm-intelligence hybridisation** (like our
  planned PSO/WOA) is a legitimate, published design choice.

### [12] Yang et al. (2026) — NSGA-III AutoML IDS

- **Method:** **NSGA-III** jointly optimises a LightGBM feature-selection threshold and key
  hyper-parameters under three objectives: weighted F1, 99th-percentile inference latency, and
  model size.
- **Datasets:** CICEVSE2024 and CICIDS2017.
- **Key results:** Competitive F1 with lower latency and smaller models than baselines.
- **Limitation:** Objectives are deployment-efficiency, not generalisation; single-dataset.
- **What it means for us:** Clarifies our distinction — we optimise for **generalisation**,
  not latency/size — and gives a template for multi-objective optimisation.

---

## 5. Theme D — Domain Adaptation / Transfer Learning

### [13] Layeghy, Baktashmotlagh & Portmann (2022/23) — DI-NIDS

- **Problem:** Supervised NIDS degrade badly on out-of-distribution networks.
- **Method:** Adversarial domain adaptation (DANN-style) learns **domain-invariant features**
  from labelled sources; a One-Class SVM then detects anomalies in the invariant space.
- **Datasets:** NFv2-CIC-2018 and NFv2-UNSW-NB15.
- **Key results:** Superior cross-domain performance vs prior domain-adaptation approaches.
- **Limitation:** Needs aligned (NetFlow v2) feature spaces; unsupervised detection only;
  requires target-domain data during adaptation.
- **What it means for us:** Represents the "model-centric transfer" alternative to our
  "feature-selection transfer" approach; we compare against it conceptually.

---

## 6. Theme E — Evaluation & Dataset-Quality Critique

### [15] Vitorino, Silva, Maia & Praça (2024) — Adversarial robustness benchmark / NewCICIDS

- **Method:** Benchmarks decision-tree ensembles (RF, XGB, LGBM, EBM) with constrained
  adversarial examples on the original **CICIDS2017**, a corrected version ("**NewCICIDS**"),
  and the more recent HIKARI dataset.
- **Key results:** The corrected CICIDS2017 yields **better and different** model rankings,
  showing the original dataset's flaws materially affect conclusions.
- **What it means for us:** Justifies (i) cleaning CICIDS2017 (NaN/Inf, duplicate column) and
  (ii) removing identity/leakage features before any generalisation claim.

---

## 7. Literature Summary Table (for the report/presentation)

| # | Title | Author(s) | What it does | Conclusion / relevance |
|---|---|---|---|---|
| 1 | On the Cross-Dataset Generalization of ML for NIDS | Cantone et al., 2024 | Cross-dataset benchmark over 4 datasets with a common feature set | Accuracy collapses to near chance cross-dataset; diagnosis only |
| 2 | Assessing Generalisation Capability of ML Models for ID | Hossain et al., 2026 | RF/LR/NB on UNSW-NB15 ↔ TON_IoT | <40% cross-dataset vs 95–99% in-dataset; no mitigation |
| 3 | Cross-Domain Generalization Failure in Lightweight IDS | Hakim et al., 2026 | Lightweight models across 3 IIoT sets; SHAP | Port-category shortcut; protocol can reverse conclusions |
| 4 | Evaluating Tabular Representation Learning for NIDS | Butt et al., 2026 | Representation learning + cross-dataset transfer | Transfer depends on source–target pair |
| 5 | CIC-IDS-2017 | Sharafaldin et al., 2018 | Dataset | Primary source dataset |
| 6 | UNSW-NB15 | Moustafa & Slay, 2015 | Dataset | Second dataset |
| 7 | TON_IoT network datasets | Moustafa, 2021 | Dataset | Third dataset (Zeek features) |
| 8 | Metaheuristic classifiers for multiclass ID | Fatima & Ali, 2022 | GA feature selection + ensembles | Strong single-dataset results, no transfer |
| 9 | Unsupervised FS via NSGA-II | Suman et al., 2019 | Multi-objective FS | Pareto subsets; single-dataset |
| 10 | MPDGGA | Li, 2026 | Multi-population GA FS | Best on 10/11 datasets; single-dataset |
| 11 | GA + Whale Optimization FS | Mojtahedi et al., 2022 | Hybrid bio-inspired FS | Swarm hybridisation is viable |
| 12 | NSGA-III AutoML IDS | Yang et al., 2026 | Multi-objective AutoML | Efficiency objectives, not transfer |
| 13 | DI-NIDS | Layeghy et al., 2022 | Domain-adversarial invariant features | Improves cross-domain but needs target data |
| 14 | Survey of NIDS Data Sets | Ring et al., 2019 | Dataset taxonomy | Explains dataset heterogeneity |
| 15 | Adversarial Robustness Benchmark (NewCICIDS) | Vitorino et al., 2024 | Robustness on corrected CICIDS2017 | Corrected data changes conclusions |

---

## 8. Research Gap (synthesis)

| Group | Cross-dataset eval? | Metaheuristic FS? | Optimised **for transfer**? |
|---|---|---|---|
| Cantone [1], Hossain [2], Hakim [3], Butt [4] | Yes | No | No (diagnosis only) |
| Metaheuristic FS [8]–[12] | No (single dataset) | Yes | **No (same-dataset fitness)** |
| Domain adaptation [13] | Yes | No | No (aligns features; needs target data) |
| **This project** | **Yes (3 datasets)** | **Yes (GA, PSO, …)** | **Yes (cross-dataset fitness)** |

> **Gap:** Metaheuristic feature selection for IDS is always optimised for **same-dataset**
> accuracy, while cross-dataset studies are **diagnostic** or rely on domain adaptation that
> needs target-domain data. **No published work selects features using a cross-dataset
> generalisation objective.** Our project fills exactly this gap.

---

## References

[1] M. Cantone, C. Marrocco, and A. Bria, "On the cross-dataset generalization of machine
learning for network intrusion detection," *IEEE Access*, vol. 12, pp. 144489–144508, 2024.

[2] M. Z. Hossain, M. A. R. Khan, M. R. Islam, S. M. S. Islam, and T. Gedeon, "Assessing
generalisation capability of machine learning models for intrusion detection,"
arXiv:2605.04407, 2026.

[3] M. A. Hakim, M. S. Uddin, and T. I. Anis, "Cross-domain generalization failure in
lightweight intrusion detection models for IIoT networks," arXiv:2607.00553, 2026.

[4] M. U. Butt, A. Hotho, and D. Schlör, "Evaluating tabular representation learning for
network intrusion detection," in *Proc. IEEE CSR*, 2026.

[5] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion
detection dataset and intrusion traffic characterization," in *Proc. ICISSP*, 2018.

[6] N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion
detection systems (UNSW-NB15 network data set)," in *Proc. MilCIS*, 2015.

[7] N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the
edge: Network TON_IoT datasets," *Sustain. Cities Soc.*, vol. 72, 2021.

[8] Z. Fatima and A. Ali, "Effective metaheuristic based classifiers for multiclass intrusion
detection," arXiv:2210.02678, 2022.

[9] C. Suman, S. Tripathy, and S. Saha, "Building an effective intrusion detection system using
unsupervised feature selection in multi-objective optimization framework," arXiv:1905.06562, 2019.

[10] C. Li, "Multi-population diversity-guided genetic algorithm for feature selection in
network intrusion detection," arXiv:2605.19864, 2026.

[11] A. Mojtahedi, F. Sorouri, A. N. Souha, A. Molazadeh, and S. S. Mehr, "Feature
selection-based intrusion detection system using genetic whale optimization algorithm and
sample-based classification," arXiv:2201.00584, 2022.

[12] L. Yang et al., "A multi-objective AutoML-based efficient intrusion detection system for EV
charging networks," in *Proc. IEEE GLOBECOM*, 2026.

[13] S. Layeghy, M. Baktashmotlagh, and M. Portmann, "DI-NIDS: Domain invariant network
intrusion detection system," *Knowledge-Based Systems*, 2023.

[14] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, "A survey of network-based
intrusion detection data sets," *Computers & Security*, vol. 86, pp. 147–167, 2019.

[15] J. Vitorino, M. Silva, E. Maia, and I. Praça, "An adversarial robustness benchmark for
enterprise network intrusion detection," in *Proc. FPS*, 2023.
