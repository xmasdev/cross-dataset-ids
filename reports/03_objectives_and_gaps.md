# Research Gaps, Objectives, and Scope

**Project:** Evaluating and Improving Cross-Dataset Generalization of ML-Based Intrusion Detection Systems
**Document:** 03 — Research Gap, Objectives & Scope
**Status:** Prepared for mid-semester review

---

## 1. Problem Statement (one paragraph)

Machine-learning-based Intrusion Detection Systems (IDS) are routinely reported with
near-perfect accuracy — but almost always when *trained and tested on the same dataset*.
Recent work [4]–[7] shows that a model trained on one network-traffic dataset and deployed on
a *different* one often performs no better than random guessing, because it has learned
dataset-specific artefacts (identity features, ports, collection-window statistics, protocol
quirks) rather than genuine attack behaviour. At the same time, a mature body of work uses
bio-inspired/metaheuristic optimisation (GA, PSO, NSGA-II/III, etc.) to select features for
IDS [9]–[13] — but **always with a fitness function that maximises single-dataset accuracy**.
This project connects the two: we use metaheuristic optimisation to search for **features and
models that remain accurate across datasets**, and we benchmark whether doing so narrows the
generalisation gap.

## 2. Research Gaps (formal)

Derived from the literature review (`02_literature_review.md`), three gaps exist:

- **Gap 1 — Optimisation-for-transfer is unexplored.** No published work performs
  metaheuristic/bio-inspired *feature selection* whose fitness is **cross-dataset**
  generalisation. Existing metaheuristic FS optimises same-dataset accuracy [9]–[13]; existing
  cross-dataset studies are diagnostic [4]–[6] or model-centric domain adaptation that needs
  target-domain data [8], [14]–[20].
- **Gap 2 — No shared benchmark across the three heterogeneous datasets.** There is no
  systematic study that (i) harmonises **CIC-IDS-2017, UNSW-NB15, and TON_IoT** onto a common
  feature subspace, (ii) evaluates the *same* models cross-dataset, and (iii) reports *which
  features and attack classes transfer* versus which are collection artefacts.
- **Gap 3 — No consolidated feature-level guidance.** No work translates the generalisation
  evidence into practical, feature-level guidelines for building IDS that transfer to unseen
  network environments.

## 3. Novelty / Contribution

1. **A transfer-oriented optimisation objective.** We define and implement a metaheuristic
   feature-selection fitness that rewards *cross-dataset* robustness (average F1/balanced
   accuracy across source→target pairs), not same-dataset accuracy. To our knowledge this is
   the first such objective in the NIDS literature.
2. **A reproducible 3-dataset cross-dataset benchmark** over a documented shared feature
   subspace (see `01_dataset_report.md` §6.2), with leakage features removed and macro
   metrics reported under realistic class distributions (addressing the protocol pitfalls in
   [6], [23]).
3. **Characterisation of the generalisation gap**, identifying transferable vs
   dataset-specific features and attack classes (fills Gap 2).
4. **Comparison of multiple bio-inspired optimisers** (GA, PSO, and at least one more —
   Differential Evolution or Whale Optimization) against standard feature-selection baselines
   (mutual information, RF importance, PCA, RFE, LASSO).
5. **Practical guidelines** distilled from the results (fills Gap 3).

## 4. Objectives

The five objectives below refine and operationalise the project proposal. Each maps to a
deliverable in `04_methodology_and_plan.md`.

1. **Evaluate cross-dataset generalisation of multiple ML/DL models** — train Random Forest,
   XGBoost/LightGBM, SVM, Logistic Regression, and a small MLP on one dataset and test on the
   structurally different ones (CIC-IDS-2017, UNSW-NB15, TON_IoT), in addition to same-dataset
   splits. *(Metrics: per-class F1, balanced accuracy, macro-F1; full 3×3 source→target
   matrix.)*
2. **Characterise the sources of the generalisation gap** — determine which attack
   categories, features, and traffic patterns hold up across environments vs which are
   dataset-specific artefacts (identity features, ports, collection-window `ct_*` features).
   *(Methods: feature-importance/SHAP analysis, t-SNE/UMAP visualisation, per-class
   confusion.)*
3. **Explore and compare metaheuristic/bio-inspired optimisers for transfer-oriented feature
   selection and model tuning** — Genetic Algorithm, Particle Swarm Optimization, and
   additional swarm/evolutionary methods (e.g. Differential Evolution, Whale Optimization),
   with fitness = cross-dataset robustness.
4. **Benchmark against standard feature-selection and tuning baselines** — mutual
   information, random-forest importance, PCA, recursive feature elimination, LASSO, and
   default hyper-parameters — to quantify how much each technique narrows the gap.
5. **Consolidate findings into practical guidelines** for building IDS that generalise
   reliably to previously unseen network environments.

## 5. Scope and Design Decisions

These decisions were fixed after a first analysis of the data and literature (see Dataset
Report §6–7); they bound the work to keep it tractable for a two-person BTech project.

| Decision | Choice | Rationale |
|---|---|---|
| Classification granularity | **Binary (benign vs attack) as primary; coarse multi-class (~6 classes) as secondary** | Binary is the dominant generalisation framing [4], [5]; multi-class class vocabularies barely overlap across datasets (Dataset Report §6.3) |
| Model family | **Classic ML (RF, XGBoost/LGBM, SVM, LR) + one small MLP** | Fast enough to run *inside* a GA/PSO fitness loop; interpretable for feature characterisation |
| Optimisation target | **Feature subset** (and lightweight hyper-parameter tuning) | Feature selection is where the transfer signal lives; it is cheap to evaluate |
| Feature space | **Shared semantic subspace (~15–25 dims)** for cross-dataset runs; full features only for same-dataset baselines | Schema mismatch makes full-feature transfer impossible (Dataset Report §6) |
| Leakage handling | **Strip IP/port/ID/timestamp features** | Otherwise models memorise identity, not behaviour [4], [6] |
| Metrics | **Balanced accuracy, macro-F1, per-class F1** (not overall accuracy) | Class imbalance is dataset-specific and severe in CICIDS2017 |
| Evaluation protocol | **Train-on-source / test-on-target, fixed seeds, repeated runs** | Reproducibility; avoids the assembly pitfalls of [23] |

## 6. Expected Outcomes

1. A quantitative answer to *"how bad is the generalisation gap across CIC-IDS-2017,
   UNSW-NB15, TON_IoT?"* (baseline matrix).
2. Evidence on *"does transfer-oriented metaheuristic feature selection beat standard feature
   selection at narrowing that gap?"* (the core claim).
3. A ranked list of *transferable vs artefact* features and a per-attack-class transfer map.
4. A short set of actionable guidelines for building generalisable IDS.

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Shared subspace too small → weak models | Expand subspace with derived features (rates, ratios); report subspace size sensitivity |
| GA/PSO fitness too slow (MLP inside loop) | Use fast inner models (RF/XGB) for FS; reserve MLP for post-selection evaluation |
| Cross-dataset scores near random (per [4]) | Frame contribution as *characterisation* + *relative improvement over baselines*, not absolute accuracy |
| Memory/time on large CSVs | Downsample with stratification; cache harmonised matrices as `.parquet` |

## 8. References

Numbered as in `02_literature_review.md`.

[4] M. Cantone, C. Marrocco, and A. Bria, "On the cross-dataset generalization of machine
learning for network intrusion detection," *IEEE Access*, vol. 12, pp. 144489–144508, 2024.

[5] M. Z. Hossain et al., "Assessing generalisation capability of machine learning models for
intrusion detection," arXiv:2605.04407, 2026.

[6] M. A. Hakim, M. S. Uddin, and T. I. Anis, "Cross-domain generalization failure in
lightweight intrusion detection models for IIoT networks," arXiv:2607.00553, 2026.

[7] M. U. Butt, A. Hotho, and D. Schlör, "Evaluating tabular representation learning for
network intrusion detection," in *Proc. IEEE CSR*, 2026.

[8] S. Layeghy, M. Baktashmotlagh, and M. Portmann, "DI-NIDS: Domain invariant network
intrusion detection system," *Knowledge-Based Systems*, 2023.

[9] Z. Fatima and A. Ali, "Effective metaheuristic based classifiers for multiclass intrusion
detection," arXiv:2210.02678, 2022.

[10] C. Suman, S. Tripathy, and S. Saha, "Building an effective intrusion detection system
using unsupervised feature selection in multi-objective optimization framework,"
arXiv:1905.06562, 2019.

[11] C. Li, "Multi-population diversity-guided genetic algorithm for feature selection in
network intrusion detection," arXiv:2605.19864, 2026.

[12] A. Mojtahedi et al., "Feature selection-based intrusion detection system using genetic
whale optimization algorithm and sample-based classification," arXiv:2201.00584, 2022.

[13] L. Yang et al., "A multi-objective AutoML-based efficient intrusion detection system for
EV charging networks," in *Proc. IEEE GLOBECOM*, 2026.

[14] J. Wu et al., "Joint semantic transfer network for IoT intrusion detection," *IEEE
Internet Things J.*, 2022.

[15] J. Wu et al., "Heterogeneous domain adaptation for IoT intrusion detection: A geometric
graph alignment approach," *IEEE Internet Things J.*, 2023.

[16] J. Wu et al., "Adaptive bi-recommendation and self-improving network for heterogeneous
domain adaptation-assisted IoT intrusion detection," *IEEE Internet Things J.*, 2023.

[17] J. Wu et al., "Open set dandelion network for IoT intrusion detection," *ACM Trans.
Internet Technol.*, 2023.

[18] M. U. Varghese and Z. Taghiyarrenani, "Intrusion detection in heterogeneous networks with
domain-adaptive multi-modal learning," arXiv:2508.03517, 2025.

[19] L. Wang, "Clustering-enhanced domain adaptation for cross-domain intrusion detection in
industrial control systems," arXiv:2604.12183, 2026.

[20] L. G. Nguyen and K. Watabe, "A method for network intrusion detection using flow sequence
and BERT framework," in *Proc. IEEE ICC*, 2023.

[23] M. A. Youssef, "Stream assembly is an uncontrolled treatment in streaming
intrusion-detection benchmarks," arXiv:2605.24696, 2026.
