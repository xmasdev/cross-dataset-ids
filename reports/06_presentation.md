---
title: "Evaluating and Improving Cross-Dataset Generalization of Machine-Learning-Based Intrusion Detection Systems"
subtitle: "B.Tech. Project – I | Mid-Semester Evaluation"
author: |
  | Under the supervision of **Dr. Preeti Bansal**
  |
  | **Shivam Chauhan** (2023UIN3323)
  | **Deepesh** (2023UIN3358)
  | **Shantanu Agarwal** (2023UIN3330)
  |
  | Department of Information Technology, NSUT, New Delhi
date: "30 September 2026"
---

```{=typst}
#set page(paper: "presentation-16-9", margin: 1.1cm)
#set text(size: 12pt)
```

# Index

1. Introduction
2. Motivation
3. Problem Statement & Objectives
4. The Three Datasets
5. Core Challenge: Feature-Schema Mismatch
6. Literature Review
7. Research Gap
8. Proposed Methodology
9. Feature Harmonisation
10. Metaheuristic Feature Selection
11. Evaluation Protocol & Baselines
12. Implementation & Progress
13. Results & Discussion
14. Gantt Chart
15. Conclusion & Future Work
16. References

```{=typst}
#pagebreak()
```

# Introduction

- **Network Intrusion Detection Systems (NIDS)** monitor traffic to detect attacks.
- ML-based IDS learn attack patterns from labelled traffic and dominate current research.
- Standard benchmarks: **CIC-IDS-2017**, **UNSW-NB15**, **TON_IoT**.
- **Usual evaluation:** train and test on the *same* dataset → accuracy > 99%.
- **Reality:** an IDS is trained on one network and deployed on a *different* one.
- **Cross-dataset generalisation:** how well does a model trained on dataset A work on dataset B?

```{=typst}
#pagebreak()
```

# Motivation

- **Deployment reality:** organisations rarely have large labelled data for their own network — they must reuse models trained elsewhere.
- **The gap is severe:** independent studies show cross-dataset accuracy falling to **<40%** or near **random chance**.
- **It is a feature problem:** models rely on dataset-specific features (IPs, ports, collection-window statistics) instead of real attack behaviour.
- **Metaheuristic feature selection is mature** — but *always* optimises **same-dataset** accuracy.
- **Our idea:** re-aim the optimiser at **cross-dataset robustness** — simple, powerful, unexplored.

```{=typst}
#pagebreak()
```

# Problem Statement & Objectives

**Problem statement**

- ML-based IDS are evaluated with same-dataset splits, overstating real-world value.
- On a structurally different network, performance collapses [1], [2].
- Existing feature-selection methods optimise only same-dataset accuracy [8]–[12].
- **Need:** characterise the cross-dataset gap, and select features *for transfer*.

**Objectives**

1. Evaluate cross-dataset generalisation of ML/DL models across the three datasets.
2. Characterise which features and attack classes transfer vs. which are artefacts.
3. Compare metaheuristic optimisers (GA, PSO, …) for transfer-oriented feature selection.
4. Benchmark against standard feature-selection baselines.
5. Produce practical guidelines for generalisable IDS.

```{=typst}
#pagebreak()
```

# The Three Datasets

| Dataset | Source / tool | Records | Features | Classes |
|---|---|---|---|---|
| **CIC-IDS-2017** | CIC, CICFlowMeter | 2,830,743 | 78 | 15 |
| **UNSW-NB15** | ACCS, Argus + Bro | 257,673 | 42 | 9 + Normal |
| **TON_IoT** | UNSW, Zeek (Bro) | 211,043 | 42 | 10 |

- Collected on **different networks** with **different feature-extraction tools**.
- Different label vocabularies and very different class balances.
- This diversity is exactly what makes them suitable for a generalisation study.

```{=typst}
#pagebreak()
```

# Core Challenge: Feature-Schema Mismatch

- The three datasets share **no common column name**.
- CICFlowMeter statistics ≠ Argus/Bro counters ≠ raw Zeek log fields.
- Only ~**6 raw features** are semantically shared (duration, bytes, packets…); ~15–25 after encoding + derived ratios.

![Shared vs dataset-specific features](figures/fig_shared_subspace.png){width=13cm}

**This mismatch is the root cause of the generalisation gap — and the focus of our work.**

```{=typst}
#pagebreak()
```

# Proposed Methodology (overview)

![Proposed end-to-end methodology](figures/fig_pipeline.png){width=13.5cm}

```{=typst}
#pagebreak()
```

# Feature Harmonisation (shared subspace)

We map semantically equivalent features across datasets into one common subspace:

| Concept | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| Duration | `Flow Duration` | `dur` | `duration` |
| Source bytes | `Total Length of Fwd Packets` | `sbytes` | `src_bytes` |
| Dest bytes | `Total Length of Bwd Packets` | `dbytes` | `dst_bytes` |
| Source packets | `Total Fwd Packets` | `spkts` | `src_pkts` |
| Dest packets | `Total Backward Packets` | `dpkts` | `dst_pkts` |
| Protocol / Service / State | (encoded) | `proto`,`service`,`state` | `proto`,`service`,`conn_state` |

- Plus derived ratios: bytes/packet, packets/second, forward/backward ratio.
- **Remove identity/leakage features** (IPs, ports, IDs).

```{=typst}
#pagebreak()
```

# Metaheuristic Feature Selection (core contribution)

- **Solution = binary mask** over the shared subspace (which features to keep).
- **Fitness = cross-dataset generalisation** (the novelty):

```
fitness(mask) = average  balanced-accuracy / macro-F1
                of model trained on source[mask]
                and evaluated on target[mask]
                over all source→target pairs
```

- **Optimisers compared:** Genetic Algorithm (GA), Particle Swarm Optimization (PSO),
  Differential Evolution / Whale Optimization.
- Optional multi-objective NSGA-II: maximise cross-dataset F1 **and** minimise number of features.
- **Key difference from prior work:** the optimiser is rewarded for *transfer*, not for fitting one dataset.

```{=typst}
#pagebreak()
```

# Evaluation Protocol & Baselines

**Protocol**

- *Same-dataset:* stratified 5-fold cross-validation.
- *Cross-dataset:* train on source, test on target → full **3×3** matrix.
- *Metrics:* balanced accuracy, macro-F1, per-class F1 (not overall accuracy — imbalance is dataset-specific).
- *Models:* RF, XGBoost/LightGBM, SVM, Logistic Regression, small MLP.
- Fixed seeds, repeated runs for reproducibility.

**Baselines for feature selection**

- No feature selection; mutual information; random-forest importance; PCA; RFE; LASSO — all matched to the same subset size as the metaheuristic.

```{=typst}
#pagebreak()
```

# Literature Review (1/2)

| # | Title | Author(s) | What it does | Conclusion |
|---|---|---|---|---|
| 1 | Cross-Dataset Generalization of ML for NIDS | Cantone et al., 2024 | Cross-dataset benchmark (4 sets) | Near-chance cross-dataset; diagnosis only |
| 2 | Assessing Generalisation Capability | Hossain et al., 2026 | RF/LR/NB on UNSW ↔ TON | <40% cross-dataset vs 95–99% in-dataset |
| 3 | Cross-Domain Failure in Lightweight IDS | Hakim et al., 2026 | 3 IIoT sets + SHAP | Port shortcut; protocol reverses results |
| 4 | Tabular Representation Learning for NIDS | Butt et al., 2026 | Representation learning | Transfer depends on source→target pair |
| 5 | CIC-IDS-2017 | Sharafaldin et al., 2018 | Dataset | Primary dataset |
| 6 | UNSW-NB15 | Moustafa & Slay, 2015 | Dataset | Second dataset |
| 7 | TON_IoT | Moustafa, 2021 | Dataset | Third dataset (Zeek) |
| 8 | Metaheuristic classifiers for ID | Fatima & Ali, 2022 | GA FS + ensembles | Strong single-dataset; no transfer |

```{=typst}
#pagebreak()
```

# Literature Review (2/2)

| # | Title | Author(s) | What it does | Conclusion |
|---|---|---|---|---|
| 9 | Unsupervised FS via NSGA-II | Suman et al., 2019 | Multi-objective FS | Pareto subsets; single-dataset |
| 10 | MPDGGA | Li, 2026 | Multi-population GA FS | Best 10/11 datasets; single-dataset |
| 11 | GA + Whale Optimization FS | Mojtahedi et al., 2022 | Hybrid bio-inspired FS | Swarm hybridisation viable |
| 12 | NSGA-III AutoML IDS | Yang et al., 2026 | Multi-objective AutoML | Efficiency, not transfer |
| 13 | DI-NIDS | Layeghy et al., 2022 | Domain-invariant features | Better cross-domain; needs target data |
| 14 | Survey of NIDS Data Sets | Ring et al., 2019 | Dataset taxonomy | Explains heterogeneity |
| 15 | Adversarial Robustness (NewCICIDS) | Vitorino et al., 2024 | Corrected CICIDS2017 | Corrected data changes conclusions |

```{=typst}
#pagebreak()
```

# Research Gap

| Group | Cross-dataset eval? | Metaheuristic FS? | Optimised **for transfer**? |
|---|---|---|---|
| Diagnostic studies [1]–[4] | Yes | No | No |
| Metaheuristic FS [8]–[12] | No | Yes | **No (same-dataset fitness)** |
| Domain adaptation [13] | Yes | No | No (needs target data) |
| **This project** | **Yes (3 datasets)** | **Yes (GA, PSO, …)** | **Yes** |

- **No published work selects features using a cross-dataset generalisation objective for NIDS.**
- Our project fills exactly this gap.

```{=typst}
#pagebreak()
```

# Implementation & Progress

**Completed**

- Reproducible Python environment + version-controlled repository.
- All three datasets downloaded and inspected.
- Automated **dataset-analysis pipeline** (rows, class distributions, NaN/Inf counts).
- Automated **figure-generation pipeline** for all report charts.
- Feature-harmonisation mapping and label taxonomy specified.
- Cleaning rules for Inf/NaN, duplicated column, label-encoding artefact, identity features.

**In progress / next**

- Cleaning & harmonisation code; cached harmonised matrices.
- Cross-dataset model benchmark.
- GA/PSO/DE optimisers + baseline feature selectors.

```{=typst}
#pagebreak()
```

# Results & Discussion (dataset analysis)

**Measured findings so far**

- CIC-IDS-2017: 2,830,743 flows, **80.3% benign**, 4,376 Inf + 1,358 NaN cells, one duplicated column.
- UNSW-NB15: 257,673 records, 9 attack classes + Normal.
- TON_IoT: 211,043 records, **near-balanced** attack classes (20,000 each, except MITM 1,043).
- **Schema mismatch:** no shared column names; only ~6 raw features semantically common.

**Discussion**

- **Overall accuracy is misleading** → we use balanced accuracy and macro-F1.
- **Identity/port features inflate same-dataset scores** and destroy transfer [1], [3].
- Baseline model benchmarking (expected: high same-dataset, near-chance cross-dataset) is the next step.

```{=typst}
#pagebreak()
```

# Gantt Chart

![Project Gantt chart](figures/fig_gantt.png){width=15cm}

```{=typst}
#pagebreak()
```

# Conclusion & Future Work

**Conclusion**

- Three structurally different benchmarks analysed; **feature-schema mismatch** is the root cause of cross-dataset failure.
- Literature review (15 core papers) confirms a clear gap: no transfer-optimised metaheuristic feature selection for NIDS.
- Objectives and a seven-phase methodology defined; codebase and tooling set up.

**Future work**

1. Cleaning & feature harmonisation.
2. Cross-dataset baseline benchmark (3×3 matrix).
3. GA / PSO / DE transfer-oriented feature selection.
4. Comparison vs. standard feature-selection baselines.
5. Analysis of transferable features/attacks + ablations.
6. Practical guidelines and final report.

```{=typst}
#pagebreak()
```

# References

[1] Cantone, Marrocco, Bria, "On the cross-dataset generalization of machine learning for network intrusion detection," *IEEE Access*, 2024.

[2] Hossain et al., "Assessing generalisation capability of machine learning models for intrusion detection," arXiv:2605.04407, 2026.

[3] Hakim, Uddin, Anis, "Cross-domain generalization failure in lightweight intrusion detection models for IIoT networks," arXiv:2607.00553, 2026.

[4] Butt, Hotho, Schlör, "Evaluating tabular representation learning for network intrusion detection," *IEEE CSR*, 2026.

[5] Sharafaldin, Lashkari, Ghorbani, "Toward generating a new intrusion detection dataset…" (CIC-IDS-2017), *ICISSP*, 2018.

[6] Moustafa, Slay, "UNSW-NB15: a comprehensive data set…", *MilCIS*, 2015.

[7] Moustafa, "Network TON_IoT datasets," *Sustainable Cities and Society*, 2021.

[8]–[12] Fatima & Ali (2022); Suman et al. (2019); Li (2026); Mojtahedi et al. (2022); Yang et al. (2026).

[13] Layeghy, Baktashmotlagh, Portmann, "DI-NIDS," *Knowledge-Based Systems*, 2023.

[14] Ring et al., "A survey of network-based intrusion detection data sets," *Computers & Security*, 2019.

[15] Vitorino et al., "An adversarial robustness benchmark for enterprise NIDS," *FPS*, 2023.

```{=typst}
#pagebreak()
```

# Thank You

**Questions?**

Evaluating and Improving Cross-Dataset Generalization of ML-Based Intrusion Detection Systems

Shivam Chauhan · Deepesh · Shantanu Agarwal

Under the supervision of Dr. Preeti Bansal, Department of IT, NSUT
