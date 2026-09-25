# Literature Review — Cross-Dataset Generalization of ML-Based Intrusion Detection

**Project:** Evaluating and Improving Cross-Dataset Generalization of ML-Based Intrusion Detection Systems
**Document:** 02 — Literature Review
**Citation style:** IEEE (numbered)
**Status:** Prepared for mid-semester review

---

## 1. Scope and Method

This review surveys the two bodies of literature that the project connects: (i) studies of
**cross-dataset generalization** of machine-learning intrusion-detection systems (IDS), and
(ii) **metaheuristic / bio-inspired feature selection** for IDS. It also covers the domain-
adaptation literature and the dataset-evaluation-critique literature, because both directly
inform how the project should be designed. Papers were retrieved primarily from the arXiv
open-access index and the original dataset publications; each is summarised with the same
structure: **problem → method → datasets → key results → limitation**.

The central finding of the review, expanded in §6, is that the two bodies of work **have not
yet been combined**: metaheuristic feature selection for IDS is overwhelmingly evaluated on a
*single* dataset, while the cross-dataset community has not used metaheuristics to select
transferable features. This intersection is the project's contribution.

---

## 2. Foundational Datasets

**[1] I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani (2018) — CIC-IDS-2017.**
Introduces the CICFlowMeter-based CIC-IDS-2017 dataset (78 flow features, realistic
enterprise traffic, 14 attack types across five days). It is the de-facto standard for
flow-based IDS research, but its identity features (ports) and class imbalance have since been
shown to inflate in-dataset accuracy [4], [22].

**[2] N. Moustafa, J. Slay (2015) — UNSW-NB15.**
Introduces UNSW-NB15 (Argus + Bro hybrid, 9 attack families), created specifically because
older datasets (KDD/NSL-KDD) were outdated. Provides a fixed train/test split (175,341 /
82,332 records) and both a binary `label` and a 9-class `attack_cat`.

**[3] N. Moustafa (2021) — TON_IoT.**
Describes the TON_IoT heterogeneous IoT/IIoT testbed (edge–fog–cloud, SDN/NFV) and its
network dataset, whose features are raw Zeek (Bro) log fields. Its attack classes are
near-uniformly balanced, making it a strong *target* domain for transfer studies.

---

## 3. Theme 1 — Cross-Dataset Generalization of NIDS

This is the literature the project directly extends.

### [4] Cantone, Marrocco, Bria (2024), *"On the Cross-Dataset Generalization of ML for Network Intrusion Detection,"* IEEE Access — **most closely related work**

- **Problem:** Do ML-based NIDS generalise when trained on one network and tested on another?
- **Method:** Four classifiers (DT, RF, KNN, MLP) trained and tested across **four** datasets
  (CIC-IDS-2017, CSE-CIC-IDS2018, LycoS-IDS2017, LycoS-Unicas-IDS2018) using a **restricted
  common feature representation**; extensive data visualisation to diagnose failures.
- **Key results:** (i) near-perfect same-dataset accuracy; (ii) cross-dataset accuracy drops
  to roughly random-chance levels for most source→target pairs; (iii) the failures are traced
  to **dataset-specific artefacts** (identity features such as IPs/ports, and collection
  methodology) rather than a lack of model capacity.
- **Limitation:** It *diagnoses* the problem but does **not** propose or optimise a solution —
  it uses a fixed common feature set and standard hyper-parameters. This leaves the "what
  should we actually do to narrow the gap" question open, which is precisely the project's
  contribution.

### [5] Hossain et al. (2026), *"Assessing Generalisation Capability of ML Models for Intrusion Detection"* — **direct competitor on two of our three datasets**

- **Problem:** Does strong in-dataset performance imply cross-dataset reliability?
- **Method:** RF, Logistic Regression, and Naïve Bayes evaluated same-dataset and
  cross-dataset on **UNSW-NB15 ↔ TON_IoT** (exactly two of our datasets).
- **Key results:** RF reaches 95.08% (UNSW) / 99.79% (TON) in-dataset, but cross-dataset
  accuracy falls **below 40%** in both directions.
- **Limitation:** Purely observational — it *measures* the gap but performs **no feature
  selection, no optimisation, and no mitigation**. It also omits CIC-IDS-2017 and any
  deep/metaheuristic component. Our project subsumes and extends it by adding CIC-IDS-2017,
  a shared-feature-subspace design, and (crucially) a metaheuristic that *optimises for*
  cross-dataset robustness.

### [6] Hakim, Uddin, Anis (2026), *"Cross-Domain Generalization Failure in Lightweight IDS Models for IIoT"*

- **Problem:** Do lightweight models trained on one IIoT dataset transfer to structurally
  different IIoT datasets?
- **Method:** Four lightweight models trained on one IIoT source and tested (no retraining)
  on two targets using a feature representation restricted to attributes shared by all three;
  explainability (SHAP-style) analysis; adversarial-robustness and limited-adaptation probes.
- **Key results:** (i) both top models rely overwhelmingly on **coarse port-category
  features**, which occur 96–435× more often in the source attack traffic than in targets —
  i.e. a **shortcut**; (ii) the evaluation protocol (class-imbalance handling) can **reverse**
  which target appears harder; (iii) adversarial robustness is *unrelated* to cross-network
  generalisation.
- **Limitation:** Like [4]/[5], it diagnoses; it does not select features *for* transfer.
  **Directly informs our design:** we must strip port/identity features and evaluate under
  realistic class distributions.

### [7] Butt, Hotho, Schlör (2026), *"Evaluating Tabular Representation Learning for NIDS"*

- **Problem:** Can representation learning replace hand-crafted NetFlow features and improve
  transfer?
- **Method:** Systematic evaluation of SOTA tabular representation-learning methods (incl.
  TabICL), autoencoders, and end-to-end transformers on several NetFlow datasets, with
  cross-dataset transfer experiments.
- **Key results:** strong **dataset–model dependency** (no method dominates everywhere);
  cross-dataset transfer works *sometimes* but varies greatly by source–target pair.
- **Limitation:** Representation-level, not feature-selection-level; no optimisation for
  transfer. Supports our claim that the *choice of what to feed the model* matters as much as
  the model.

### [31] Pontes et al. (2021), *"A New Method for Flow-based NIDS Using the Inverse Potts Model"*

- **Problem:** Classical ML classifiers overfit a single distribution and lack domain
  adaptation.
- **Method:** Energy-based Flow Classifier (EFC), an anomaly detector trained only on benign
  flows via inverse statistical modelling.
- **Key results:** EFC is **more adaptable across distributions** than supervised baselines on
  CIDDS-001, CICIDS17, and CICDDoS19.
- **Limitation:** Binary anomaly detection only; not a supervised multi-class transfer
  solution. Relevant as evidence that *distribution-agnostic* representations transfer better.

### [30] Xu, Liu (2025), *"Robust Anomaly Detection in Network Traffic: Evaluating ML Models on CICIDS2017"*

- **Problem:** How do supervised vs unsupervised models generalise to *unseen attack types*
  within a dataset?
- **Method:** MLP, 1D-CNN, OCSVM, LOF on CICIDS2017 under "known attacks" vs "novel attacks"
  scenarios.
- **Key results:** supervised MLP/CNN are near-perfect on familiar attacks but collapse on
  novel ones; boundary-based OCSVM trades off precision/recall best.
- **Limitation:** Within-dataset novelty, not cross-dataset. Reinforces that supervised
  models are fragile to distribution shift — the same effect we study across datasets.

---

## 4. Theme 2 — Domain Adaptation and Transfer Learning for NIDS

This literature shows the "sophisticated transfer" route, against which a *feature-selection*
route can be compared.

### [8] Layeghy, Baktashmotlagh, Portmann (2022/23), *"DI-NIDS: Domain Invariant NIDS"*

- **Method:** Adversarial domain adaptation (DANN-style) to learn domain-invariant features
  from labelled sources, then a One-Class SVM on the invariant space.
- **Datasets/results:** NFv2-CIC-2018 and NFv2-UNSW-NB15; superior cross-domain performance
  vs prior DA approaches.
- **Limitation:** Requires aligned (NetFlow v2) feature spaces; unsupervised detection only.

### [14] Wu et al. (2022), *"JSTN: Joint Semantic Transfer Network for IoT Intrusion Detection"*

- **Method:** Multi-source heterogeneous domain adaptation transferring three "semantics"
  (scenario, weighted-implicit, hierarchical-explicit alignment) from a network-intrusion
  source to a data-scarce IoT target.
- **Results:** ~10.3% average accuracy boost vs SOTA baselines (IEEE IoT-J).
- **Limitation:** Targets *data-scarcity* (few labelled IoT samples), a different motivation
  than our *generalisation-under-schema-mismatch* problem.

### [15] Wu et al. (2023), *"GGA: Heterogeneous DA via Geometric Graph Alignment"*

- **Method:** Aligns intrusion *category graphs* (vertices = categories, edges = relations)
  between domains to transfer knowledge despite heterogeneous feature spaces.
- **Limitation:** Requires overlapping *category* semantics; heavy machinery; evaluated for
  IoT-target transfer.

### [16] Wu et al. (2023), *"ABRSI: Adaptive Bi-Recommendation and Self-Improving Network"*

- **Method:** Unsupervised heterogeneous DA with recommender-style bi-matching, hybrid
  pseudo-labelling, and error-knowledge learning.
- **Limitation:** Same family of data-scarcity transfer; no feature-selection interpretation.

### [17] Wu et al. (2023), *"OSDN: Open-Set Dandelion Network"*

- **Method:** Open-set heterogeneous DA; "dandelion" feature-space shaping for
  inter-class separability; detects target classes unseen in the source.
- **Results:** Outperforms three SOTA baselines by 16.9% (ACM ToIT).
- **Limitation:** Open-set and label-scarce focus; not a benchmark of *classical* models under
  schema mismatch.

### [18] Varghese, Taghiyarrenani (2025), *"Intrusion Detection in Heterogeneous Networks with Domain-Adaptive Multi-Modal Learning"*

- **Method:** Deep model integrating multi-modal learning with DA, processing multiple
  datasets cyclically to adapt to varying feature spaces.
- **Limitation:** Requires joint access to multiple datasets; black-box.

### [19] Wang (2026), *"Clustering-Enhanced Domain Adaptation for Cross-Domain IDS in ICS"*

- **Method:** Spectral-transform feature alignment + K-Medoids/PCA clustering enhancement for
  industrial-control traffic.
- **Results:** up to +49% accuracy vs five baselines; clustering adds up to +26%.
- **Limitation:** ICS-specific; still requires target-domain exposure (transductive).

### [20] Nguyen, Watabe (2023), *"Flow-Sequence NIDS via BERT"*

- **Method:** Encode *sequences of flows* with a BERT transformer to improve domain
  adaptation.
- **Limitation:** Needs ordered flow sequences (not plain tabular rows); early-stage results.

### [33] Hossain et al. (2026), *"Cross-Dataset Zero-Day IDS via Siamese Network + RL"*

- **Method:** Siamese network for anomaly correlation + PPO reinforcement-learning adaptive
  defence for zero-day detection.
- **Results:** 99.07% unknown-attack accuracy, 93.94% zero-day ratio on IoT benchmarks.
- **Limitation:** Unsupervised/anomaly framing; not a reproducible cross-dataset *benchmark*
  with classical models; no feature-selection lens.

**Summary of Theme 2:** domain adaptation works, but (a) it usually assumes overlapping label
semantics and aligned or learnable-aligned feature spaces, (b) it requires target-domain data,
and (c) it is model-centric. **None of these works use metaheuristic feature selection**, and
none frame the problem as "choose a small transferable feature subset evaluated under a strict
train-on-source/test-on-target protocol."

---

## 5. Theme 3 — Metaheuristic / Bio-inspired Feature Selection for IDS

This is the "optimisation" literature the project uses as its engine.

### [9] Fatima, Ali (2022), *"Effective Metaheuristic-Based Classifiers for Multiclass IDS"*

- **Method:** Wrapper **Genetic Algorithm (GA)** feature selection + stacking/bagging
  ensembles on UNSW-NB15 and CICDDoS2019.
- **Results:** GA + stacking improves multi-class accuracy significantly; better DR and lower
  FAR than baselines.
- **Limitation:** Fitness = **same-dataset** accuracy. The selected subset is optimised for
  the *training* distribution, so nothing guarantees transfer. **This is the exact pattern the
  project reverses** (we optimise for cross-dataset robustness instead).

### [10] Suman, Tripathy, Saha (2019), *"Building an Effective IDS using Unsupervised Feature Selection in a Multi-objective Optimization Framework"*

- **Method:** **NSGA-II** optimising three information-theoretic objectives (mutual
  information, std-dev, information gain) for *unsupervised* feature selection; Pareto-optimal
  subsets feed SVM/DT/KNN.
- **Results:** 99.78% accuracy on KDD-99, 99.83% on NSL-KDD (20% test), 99.65% on Kyoto.
- **Limitation:** Single-dataset evaluation on legacy datasets; the multi-objective idea
  (accuracy vs #features) is useful for us, but our objectives will be *cross-dataset*.

### [11] Li (2026), *"MPDGGA: Multi-population Diversity-guided GA for Feature Selection in NIDS"*

- **Method:** Chained multi-population GA with an information-gain-ratio diversity operator.
- **Results:** Best accuracy on 10/11 datasets (NSL-KDD, UNSW-NB15, 9 UCI) while selecting
  ≥2.26% of features.
- **Limitation:** Still single-dataset accuracy; no cross-dataset protocol.

### [12] Mojtahedi et al. (2022), *"FS-based IDS using Genetic Whale Optimization + Sample-based Classification"*

- **Method:** Hybrid **WOA + GA** feature selection with KNN on KDDCUP1999.
- **Limitation:** Legacy dataset, single-domain, accuracy-only fitness.

### [13] Yang et al. (2026), *"Multi-Objective AutoML-based Efficient IDS for EV Charging Networks"*

- **Method:** **NSGA-III** jointly optimising LightGBM feature threshold + hyper-parameters
  under three objectives (weighted F1, inference latency, model size).
- **Datasets/results:** CICEVSE2024 and CICIDS2017; competitive F1 with lower latency/size.
- **Limitation:** Deployment-efficiency objectives, not generalisation; single-dataset.

### [32] Barati (2025), *"Quantum GA-enhanced Self-Supervised IDS for WSN/IoT"*

- **Method:** Quantum-inspired GA for feature selection + self-supervised representation
  learning.
- **Limitation:** Niche (quantum-inspired); single-dataset IoT evaluation.

**Summary of Theme 3:** the metaheuristic FS literature is mature and effective, but its
fitness functions are uniformly **same-dataset accuracy/F1**. To our knowledge, **no published
work optimises a feature subset for cross-dataset generalisation.** This is the gap.

---

## 6. Theme 4 — Dataset & Evaluation Critique (why careful protocol matters)

- **[25] Ring et al. (2019), "A Survey of Network-Based Intrusion Detection Data Sets":**
  canonical taxonomy of NIDS datasets and their properties; documents that datasets differ in
  collection environment, feature semantics, and label quality — the root of the
  generalisation problem.
- **[22] Vitorino et al. (2024), "An Adversarial Robustness Benchmark for Enterprise NIDS":**
  shows a *corrected* CICIDS2017 ("NewCICIDS") changes model rankings; underlines that the
  dataset's known flaws materially affect results.
- **[23] Youssef (2026), "Stream Assembly Is an Uncontrolled Treatment in Streaming IDS
  Benchmarks":** demonstrates that *how* evaluation streams are assembled (interleaving,
  pooling, replay) changes what is measured and can reverse rankings — a warning we take
  seriously for our train/test protocol.
- **[21] Corsini, Yang, Apruzzese (2021), "On the Evaluation of Sequential ML for NIDS":**
  establishes fair-comparison methodology for flow-based models; motivates our decision to use
  matched, reproducible splits and macro metrics.
- **[24] El Mahdaouy et al. (2026), "Deep Learning for Contextualized NetFlow-based NIDS":**
  survey that lists **weak cross-dataset generalization** among the key failure modes that
  inflate reported results — direct justification for our benchmark design.

---

## 7. Synthesis and Research Gap

The following table summarises the gap. "Cross-dataset evaluation" means a strict
train-on-source / test-on-target protocol; "metaheuristic FS" means GA/PSO/NSGA-style feature
selection; "optimises for transfer" means the fitness function targets cross-dataset
performance rather than same-dataset accuracy.

| Work | Datasets (n≥3, heterogeneous) | Cross-dataset eval | Metaheuristic FS | Optimises for transfer | Mitigates gap |
|---|---|---|---|---|---|
| Cantone et al. [4] | ✓ (4) | ✓ | ✗ | ✗ | ✗ (diagnosis only) |
| Hossain et al. [5] | ~ (2) | ✓ | ✗ | ✗ | ✗ |
| Hakim et al. [6] | ✓ (3 IIoT) | ✓ | ✗ | ✗ | ✗ |
| Butt et al. [7] | ✓ | ✓ | ✗ | ✗ | partial (representation) |
| DI-NIDS [8] / JSTN [14] / GGA [15] / OSDN [17] | ✓ | ✓ (DA) | ✗ | ✗ (aligns features, not selects) | partial (needs target data) |
| GA/PSO/NSGA FS [9]–[13], [32] | ✗ (single) | ✗ | ✓ | ✗ (same-dataset fitness) | ✗ |
| **This project** | **✓ (3, heterogeneous)** | **✓** | **✓ (GA, PSO, +)** | **✓ (cross-dataset fitness)** | **✓ (goal)** |

### 7.1 Explicit gap statements

1. **Gap 1 (primary):** No published work applies **metaheuristic/bio-inspired feature
   selection whose fitness is cross-dataset generalisation** to network intrusion detection.
   All existing metaheuristic FS optimises same-dataset accuracy [9]–[13]; all existing
   cross-dataset work is either diagnostic [4]–[6] or model-centric domain adaptation
   [8], [14]–[20].
2. **Gap 2:** There is no **systematic benchmark** that jointly evaluates the *same* ML/DL
   models across **CIC-IDS-2017, UNSW-NB15, and TON_IoT** under a shared feature subspace and
   reports which *features* and *attack classes* transfer versus which are dataset artefacts.
3. **Gap 3:** No work consolidates these findings into **practical, feature-level guidelines**
   for building IDS that generalise to unseen networks.

These three gaps map one-to-one onto the project's five objectives (see
`03_objectives_and_gaps.md`).

---

## References

[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion
detection dataset and intrusion traffic characterization," in *Proc. ICISSP*, 2018.

[2] N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion
detection systems (UNSW-NB15 network data set)," in *Proc. MilCIS*, 2015.

[3] N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at
the edge: Network TON_IoT datasets," *Sustain. Cities Soc.*, vol. 72, 2021.

[4] M. Cantone, C. Marrocco, and A. Bria, "On the cross-dataset generalization of machine
learning for network intrusion detection," *IEEE Access*, vol. 12, pp. 144489–144508, 2024,
doi:10.1109/ACCESS.2024.3472907.

[5] M. Z. Hossain, M. A. R. Khan, M. R. Islam, S. M. S. Islam, and T. Gedeon, "Assessing
generalisation capability of machine learning models for intrusion detection," arXiv:2605.04407, 2026.

[6] M. A. Hakim, M. S. Uddin, and T. I. Anis, "Cross-domain generalization failure in
lightweight intrusion detection models for IIoT networks," arXiv:2607.00553, 2026.

[7] M. U. Butt, A. Hotho, and D. Schlör, "Evaluating tabular representation learning for
network intrusion detection," in *Proc. IEEE Int. Conf. Cyber Secur. Resilience (CSR)*, 2026.

[8] S. Layeghy, M. Baktashmotlagh, and M. Portmann, "DI-NIDS: Domain invariant network
intrusion detection system," *Knowledge-Based Systems*, 2023, doi:10.1016/j.knosys.2023.110626.

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

[14] J. Wu, Y. Wang, B. Xie, S. Li, H. Dai, K. Ye, and C. Xu, "Joint semantic transfer network
for IoT intrusion detection," *IEEE Internet Things J.*, 2022.

[15] J. Wu, H. Dai, Y. Wang, K. Ye, and C. Xu, "Heterogeneous domain adaptation for IoT
intrusion detection: A geometric graph alignment approach," *IEEE Internet Things J.*, 2023.

[16] J. Wu, Y. Wang, H. Dai, C. Xu, and K. B. Kent, "Adaptive bi-recommendation and
self-improving network for heterogeneous domain adaptation-assisted IoT intrusion detection,"
*IEEE Internet Things J.*, 2023.

[17] J. Wu, H. Dai, K. B. Kent, J. Yen, C. Xu, and Y. Wang, "Open set dandelion network for
IoT intrusion detection," *ACM Trans. Internet Technol.*, 2023.

[18] M. U. Varghese and Z. Taghiyarrenani, "Intrusion detection in heterogeneous networks with
domain-adaptive multi-modal learning," arXiv:2508.03517, 2025.

[19] L. Wang, "Clustering-enhanced domain adaptation for cross-domain intrusion detection in
industrial control systems," arXiv:2604.12183, 2026.

[20] L. G. Nguyen and K. Watabe, "A method for network intrusion detection using flow sequence
and BERT framework," in *Proc. IEEE ICC*, 2023.

[21] A. Corsini, S. J. Yang, and G. Apruzzese, "On the evaluation of sequential machine learning
for network intrusion detection," in *Proc. ACM ARES*, 2021.

[22] J. Vitorino, M. Silva, E. Maia, and I. Praça, "An adversarial robustness benchmark for
enterprise network intrusion detection," in *Proc. FPS*, 2023.

[23] M. A. Youssef, "Stream assembly is an uncontrolled treatment in streaming
intrusion-detection benchmarks," arXiv:2605.24696, 2026.

[24] A. El Mahdaouy, I. Ait Yahia, S. Oualil, and I. Berrada, "Deep learning for contextualized
NetFlow-based network intrusion detection: Methods, data, evaluation and deployment,"
arXiv:2602.05594, 2026.

[25] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, "A survey of network-based
intrusion detection data sets," *Comput. Secur.*, vol. 86, pp. 147–167, 2019.

[26] R. Kale, Z. Lu, K. W. Fok, and V. L. L. Thing, "A hybrid deep learning anomaly detection
framework for intrusion detection," in *Proc. IEEE BigDataSecurity*, 2022.

[27] A. Pasdar, S. K. Kermanshahi, N. Moustafa, and V.-T. Pham, "Collaborative zone-adaptive
zero-day intrusion detection for IoBT," arXiv:2602.16098, 2026.

[28] N. Moustafa, M. Keshk, E. Debie, and H. Janicke, "Federated TON_IoT Windows datasets for
evaluating AI-based security applications," arXiv:2010.08522, 2020.

[29] M. Hassanin, M. Keshk, S. Salim, M. Alsubaie, and D. Sharma, "PLLM-CS: Pre-trained large
language model for cyber threat detection in satellite networks," arXiv:2405.05469, 2024.

[30] Z. Xu and Y. Liu, "Robust anomaly detection in network traffic: Evaluating machine learning
models on CICIDS2017," arXiv:2506.19877, 2025.

[31] C. Pontes, M. Souza, J. Gondim, M. Bishop, and M. Marotta, "A new method for flow-based
network intrusion detection using the inverse Potts model," *IEEE Trans. Netw. Serv. Manag.*,
vol. 18, no. 2, pp. 1125–1136, 2021.

[32] H. Barati, "A quantum genetic algorithm-enhanced self-supervised intrusion detection system
for wireless sensor networks in the Internet of Things," arXiv:2509.03744, 2025.

[33] M. M. Hossain, S. D. Turja, S. Tasnim, M. F.-U.-A. Juboraj, and M. I. Hossain, "A
cross-dataset based zero-day intrusion detection system by integrating Siamese network and
reinforcement learning," arXiv:2609.26115, 2026.
