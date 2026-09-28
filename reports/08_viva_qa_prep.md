# Viva / Q&A Preparation — Likely Questions with Answers

**Project:** Cross-Dataset Generalization of ML-Based Intrusion Detection Systems
**Use:** Practise these out loud. Answers are written in a natural speaking style.

---

## A. Project basics (almost certain)

**Q1. What is your project about?**
> We study why machine-learning intrusion detection systems fail when moved from one network to
> another, and we use bio-inspired optimisation (genetic algorithms, particle swarm) to choose
> the features that let a model work across different networks instead of just one.

**Q2. What is an intrusion detection system?**
> A system that monitors network traffic and raises an alarm when it sees malicious activity. A
> network IDS (NIDS) looks at flows of traffic and classifies each as benign or an attack.

**Q3. Why machine learning instead of traditional signature-based detection?**
> Signature-based systems only catch attacks whose "signature" is already known. ML can learn
> patterns from examples and can flag novel or slightly modified attacks. But ML has its own
> weakness — poor generalisation — which is exactly what we study.

**Q4. What problem are you solving?**
> The generalisation gap. Models get 99% when tested on the same dataset but often below 40% —
> sometimes near random — on a different dataset. We want to measure this gap and reduce it.

**Q5. What is your main contribution / novelty?**
> Metaheuristic feature selection for NIDS has always been optimised for *same-dataset* accuracy.
> We are the first to use a **cross-dataset fitness function** — we select features that transfer
> to unseen networks. We also build a common feature subspace across three very different
> datasets and characterise what transfers.

---

## B. Fundamental concepts

**Q6. What is a network flow?**
> A group of packets that belong to the same conversation, identified by source/destination
> IP and port and protocol. Instead of analysing every packet, we summarise the flow with
> statistics (duration, bytes, packets, inter-arrival times).

**Q7. What is a feature? Give examples from your datasets.**
> A numeric column describing a flow. CIC-IDS-2017 has 78 (e.g. `Flow Duration`,
> `Total Fwd Packets`); UNSW-NB15 has 42 (e.g. `dur`, `sbytes`); TON_IoT has 42 Zeek fields
> (e.g. `duration`, `src_bytes`, `conn_state`).

**Q8. What is supervised learning here?**
> We train a model on labelled flows (benign vs attack) so it learns to predict the label of a
> new flow. We use classifiers like Random Forest, XGBoost, SVM and a small neural network.

**Q9. What is overfitting, and why does it matter here?**
> Overfitting is when a model memorises training data instead of learning general patterns. On a
> new network it fails. Dataset-specific features make overfitting worse — a key reason for the
> generalisation gap.

---

## C. Cross-dataset generalisation

**Q10. Explain cross-dataset generalisation with an example.**
> Train a model on UNSW-NB15 (an Australian lab network) and test it on TON_IoT (an IoT testbed).
> The model has never seen TON_IoT's traffic patterns; if it learned real attack behaviour it
> will still work — if it learned UNSW-specific shortcuts, it will fail.

**Q11. Why do models fail cross-dataset?**
> Three reasons: (1) **different feature spaces** — the columns don't even match; (2)
> **dataset-specific shortcuts** — IPs, ports and collection-window features that don't exist or
> don't mean the same thing in another network; (3) **different attack distributions** — the mix
> of attacks and benign traffic differs.

**Q12. Is it even possible to fully solve this?**
> Probably not fully — the distributions genuinely differ. Our goal is to *measure* the gap and
> to *narrow* it using transfer-oriented feature selection, and to report honest, relative
> improvements over baselines rather than claiming perfect accuracy.

**Q13. What is domain adaptation, and why not just use that?**
> Domain adaptation aligns feature distributions between a source and target dataset. It works,
> but usually needs some target-domain data and complex neural training, and assumes label
> semantics overlap. We chose feature selection because it is simpler, interpretable, and
> directly identifies *which features* are the problem — which also gives practical guidelines.

---

## D. Datasets

**Q14. Which datasets and why?**
> CIC-IDS-2017, UNSW-NB15 and TON_IoT. They are the most-used modern benchmarks and are
> **structurally different** (different networks and different feature-extraction tools:
> CICFlowMeter, Argus/Bro, and Zeek), which is essential for studying generalisation.

**Q15. How big are they?**
> CIC-IDS-2017: 2,830,743 flows, 78 features (one duplicated), 15 classes. UNSW-NB15: 257,673
> records, 42 features, 9 attack classes plus Normal. TON_IoT: 211,043 records, 42 features,
> 10 classes.

**Q16. What problems did you find in the data?**
> CIC-IDS-2017 has 4,376 infinite and 1,358 missing values (division by zero in rate columns), a
> duplicated `Fwd Header Length` column, a non-ASCII label character, and very severe class
> imbalance (only 11 Heartbleed and 36 Infiltration samples). TON_IoT is nearly balanced
> (20,000 per attack). None have missing values apart from CIC-IDS-2017.

**Q17. Why is the class imbalance a problem?**
> If 80% of CIC-IDS-2017 is benign, a lazy model can get 80% accuracy by always saying
> "benign". That's why we use balanced accuracy, macro-F1 and per-class F1 instead of plain
> accuracy.

**Q18. What is the "feature-schema mismatch"?**
> The three datasets share **no common column name**. CICFlowMeter computes statistics; Argus/Bro
> computes counters; Zeek logs raw protocol fields. Only about 6 raw features are semantically
> equivalent across all three. This mismatch is the root cause of cross-dataset failure.

---

## E. Methodology, feature selection, metaheuristics

**Q19. What is feature selection and why use it?**
> Choosing a smaller, more useful subset of features. It makes models faster and less prone to
> overfitting. For us it is the key lever: if we keep only features that are meaningful on *any*
> network, the model transfers better.

**Q20. What is a metaheuristic? Why not brute force?**
> A nature-inspired search that finds good solutions without checking all possibilities. With 20
> features there are over a million subsets, so brute force is wasteful; GA and PSO find good
> subsets quickly.

**Q21. How does a Genetic Algorithm work?**
> It keeps a population of candidate feature-subsets. Each is scored (fitness). The best ones
> "reproduce" via crossover (mixing two subsets), and we add random mutations, over many
> generations. Good subsets survive and improve.

**Q22. How does PSO work?**
> It simulates a flock of birds. Each "particle" is a candidate subset moving through the search
> space, and each is pulled toward its own best position and the swarm's best. It converges on
> good regions of the search space.

**Q23. What exactly is your fitness function?**
> It is the average cross-dataset performance: we train a fast model (Random Forest or XGBoost)
> on the source dataset using only the selected features, test it on the target dataset, and
> use balanced accuracy / macro-F1. The optimiser is rewarded when the subset transfers well.
> This is the crucial difference from previous work, which uses same-dataset accuracy.

**Q24. Why standardise features per dataset and not globally?**
> To avoid leaking target-dataset information into training. Each dataset is normalised using its
> own statistics.

**Q25. What is the "shared subspace"?**
> A common set of semantically equivalent features — duration, source/destination bytes, source/
> destination packets, protocol, service, connection state, plus derived ratios — about 15–25
> dimensions after encoding. We run cross-dataset experiments in this common space.

**Q26. What are "leakage" or "shortcut" features?**
> Features that accidentally reveal the answer without real detection — e.g. IP addresses, port
> numbers, timestamps, or collection-window counters. A model can "cheat" with them on one
> dataset but fail elsewhere. We remove them.

**Q27. Why remove the destination port if it's a real network feature?**
> Ports are highly dataset-specific (a lab's attack used a particular port). Keeping them lets
> the model memorise the scenario instead of behaviour. Research [3] showed models relying on
> port categories that were 96–435× more common in the source attacks — a classic shortcut.

---

## F. Evaluation

**Q28. What models will you use and why?**
> Random Forest, XGBoost/LightGBM, SVM, Logistic Regression and a small MLP. Classic models are
> fast enough to sit inside a GA/PSO fitness loop and are easy to interpret; the MLP gives a
> neural baseline.

**Q29. What metrics and why not accuracy?**
> Balanced accuracy, macro-F1, and per-class F1. Because the datasets have very different and
> severe class imbalances, plain accuracy hides failures on rare classes.

**Q30. How will you test cross-dataset?**
> Train on one dataset, test on another, for all ordered pairs — a 3×3 matrix (the diagonal is
> same-dataset). We repeat with fixed seeds and report mean and standard deviation.

**Q31. What are your baselines?**
> Feature selection baselines: no selection, mutual information, random-forest importance, PCA,
> RFE and LASSO — each matched to the same subset size. The comparison tells us whether
> transfer-optimised selection actually beats standard selection.

**Q32. What does "success" look like?**
> Reproducing the documented collapse, then showing a **statistically reliable improvement** in
> cross-dataset F1 for the metaheuristic-selected features over (i) no feature selection and (ii)
> the best standard feature-selection baseline. Also, identifying which features/attacks transfer.

---

## G. Literature-specific questions (they can ask about ANY paper)

**Q33. What did Cantone et al. (2024) find? [1]**
> They benchmarked four classifiers across four datasets (CIC-IDS-2017, CSE-CIC-IDS2018,
> LycoS-IDS2017, LycoS-Unicas-IDS2018) using a restricted common feature set. Same-dataset
> results were near-perfect; cross-dataset results fell to near random chance. They concluded
> the cause is dataset-specific artefacts, and they only *diagnosed* the problem.

**Q34. How is Cantone different from you?**
> Same problem, but they use a fixed common feature set and do nothing to improve transfer.
> We add an optimiser that *selects features for transfer*, and a third dataset (TON_IoT).

**Q35. What did Hossain et al. (2026) do? [2]**
> They evaluated Random Forest, Logistic Regression and Naïve Bayes on UNSW-NB15 and TON_IoT.
> RF got 95.08% (UNSW) and 99.79% (TON) in-dataset but under 40% cross-dataset. Pure
> measurement, no mitigation.

**Q36. What did Hakim et al. (2026) show? [3]**
> On three IIoT datasets, lightweight models relied heavily on coarse port-category features —
> a shortcut — and the evaluation protocol (how imbalance is handled) could even reverse which
> target looked harder. Adversarial robustness was unrelated to cross-network generalisation.

**Q37. What did Butt et al. (2026) find? [4]**
> Evaluated tabular representation learning (including TabICL), autoencoders and transformers for
> NetFlow intrusion detection. Transfer sometimes worked, but depended strongly on the specific
> source→target pair — no universal winner.

**Q38. What did Fatima & Ali (2022) do? [8]**
> Used a wrapper Genetic Algorithm to select features with stacking/bagging ensembles on
> UNSW-NB15 and CICDDoS2019, improving multi-class accuracy and detection rate — but the fitness
> was same-dataset accuracy.

**Q39. What is NSGA-II used for in the literature? [9]**
> Suman et al. (2019) used NSGA-II to optimise three information-theoretic objectives at once for
> unsupervised feature selection on KDD-99/NLS-KDD/Kyoto, obtaining Pareto-optimal subsets. It
> motivates our optional multi-objective version (F1 vs number of features).

**Q40. What is MPDGGA? [10]**
> A multi-population, diversity-guided Genetic Algorithm by Li (2026) that maintains diversity to
> select very few features while getting the best accuracy on 10 of 11 datasets — all
> single-dataset.

**Q41. What did Yang et al. (2026) optimise? [12]**
> NSGA-III jointly optimised a LightGBM feature threshold and hyper-parameters for weighted F1,
> inference latency and model size on EV-charging and CICIDS2017 data — efficiency objectives,
> not generalisation.

**Q42. What is DI-NIDS? [13]**
> Layeghy et al. use adversarial domain adaptation to learn domain-invariant features, then a
> One-Class SVM to detect anomalies, improving cross-domain performance — but it needs aligned
> feature spaces and target-domain data.

**Q43. Why cite Ring et al. (2019)? [14]**
> Their survey of NIDS datasets explains that datasets differ in collection environment, feature
> semantics and labelling — the fundamental reason cross-dataset generalisation is hard.

**Q44. What did Vitorino et al. (2024) show? [15]**
> Using a corrected CICIDS2017 ("NewCICIDS") changed model rankings compared to the original,
> showing the dataset's flaws materially affect conclusions — which justifies our careful
> cleaning and leakage removal.

---

## H. Tough questions & how to handle them

**Q45. If cross-dataset accuracy stays near random, is your project a failure?**
> No. Our success metric is *relative improvement over baselines*, not absolute accuracy.
> Characterising the gap, identifying transferable features, and producing guidelines are
> valuable contributions even if absolute numbers remain modest.

**Q46. Could deep learning solve it better than feature selection?**
> Deep models are also affected — they overfit the same shortcuts. Feature selection is
> complementary and interpretable, and it directly targets the identified cause. We include an
> MLP to compare, and note deep approach as future work.

**Q47. How do you know you removed *all* leakage?**
> We remove obvious identity features (IPs, ports, IDs, timestamps) and use SHAP/importance to
> check for remaining suspiciously dominant features. We also run an ablation with and without
> leakage features to quantify the effect.

**Q48. What about the different attack labels across datasets?**
> We first map everything to binary (benign vs attack). For multi-class we use a coarse taxonomy
> (DoS/DDoS, Recon/Scanning, Brute-force, Backdoor, Web/Injection, etc.). Some categories don't
> map cleanly — we report that as a finding.

**Q49. Why is your shared subspace so small?**
> Because the datasets use different tools. Only fields like duration, byte/packet counts and
> protocol/service/state are truly comparable. A small, honest subspace is better than forcing
> incompatible features together.

**Q50. What if the optimiser just overfits the target dataset?**
> We design the protocol so the target is only used for *scoring*, and we use a held-out source
> fold for training. We report results across multiple seeds and target pairs, not a single
> lucky split.

**Q51. How is this different from just combining all datasets into one big training set?**
> Combining still leaves mismatched columns and different scales; you'd need alignment anyway.
> Our shared-subspace approach makes the alignment explicit and tests transfer *without*
> target labels — a stricter, more realistic setting.

**Q52. Which part did each team member do?** *(prepare a real division of work)*
> e.g. Dataset acquisition/cleaning and analysis; literature review and methodology design;
> implementation of models and optimisers. Be ready to name specific files/scripts each person
> wrote.

---

## I. Logistics

**Q53. What is your timeline?**
> Cleaning and harmonisation (weeks 3–6), cross-dataset baseline (weeks 6–8), metaheuristic
> feature selection (weeks 8–11), baseline comparison and analysis (weeks 11–14), final report
> and guidelines.

**Q54. What tools are you using?**
> Python with pandas/NumPy/scikit-learn/XGBoost/LightGBM for data and models, DEAP for the
> Genetic Algorithm and PySwarms (or custom code) for PSO, and Matplotlib/Seaborn for figures.
> Everything is in a version-controlled repository.

**Q55. What have you completed so far?**
> Dataset analysis (real measured statistics and charts), the 15-paper literature review with the
> identified gap, the objectives and 7-phase methodology, and the reproducible codebase.

---

## J. One-line memory hooks

- **Problem:** 99% in the lab, ~random in the real world.
- **Cause:** mismatched features + dataset-specific shortcuts (leakage).
- **Solution:** select a shared set of transferable features.
- **How:** GA / PSO with a **cross-dataset fitness function**.
- **Novelty:** nobody has optimised features *for transfer* in NIDS before.
- **Success:** beat the baselines and explain what transfers.
