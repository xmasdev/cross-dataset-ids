# End-Semester Notes for the Team: Plain-Language Summary and Viva Preparation

Read this before the presentation. Every number here is in `results/v2/TABLES.md` and in the
final report (`09_final_report.pdf`).

## 1. The project in six sentences

1. A detector trained on one network's data does not work on another network: every model we
   tried scores about 0.995 AUC on its own dataset and 0.42–0.49 on a different one.
2. The reason is that the same feature means opposite things on different networks, so a rule
   learned on one network is wrong on the next.
3. Our mid-semester feature-selection result looked good only because the subset was chosen
   using the test data; done properly, it gives no gain. We say this ourselves in the report.
4. Our new method, BRACE, does not move a classifier between networks at all. It learns what
   *normal* looks like on the new network and flags what is unusual there.
5. It needs a sample of clean traffic and the live unlabelled traffic of the new network, and
   no attack labels. It reaches 0.97–0.998 AUC, within 0.02 of a model that has the labels.
6. A genetic algorithm picks the features using a score that can be computed without labels.

## 2. Terms you must be able to explain

- **ROC AUC.** The probability that a randomly chosen attack gets a higher score than a randomly
  chosen benign flow. 0.5 is a coin toss, 1.0 is perfect. Below 0.5 means the model ranks
  benign flows as more suspicious than attacks.
- **Cross-dataset.** Train on dataset A, test on dataset B.
- **Benign calibration flows.** Traffic recorded on the target network during a period known to
  be clean. This is the only "label" BRACE uses.
- **Unlabelled stream.** Live traffic of the target: mostly benign, with an unknown number of
  attacks mixed in.
- **Benign-anchored quantile.** Each feature value is replaced by its rank among the target's
  benign flows. "Zero" then means "typical here".
- **kNN score.** Distance from a flow to its five nearest benign flows. Far from all benign
  flows = suspicious.
- **Benign-vs-unlabelled classifier.** A model trained to tell calibration flows from stream
  flows. The only systematic difference between the two is the attacks hidden in the stream, so
  the model ends up ranking attacks highest.
- **Conformal p-value.** The fraction of held-out benign flows that score at least as high as
  this flow. It lets us fix the false-alarm rate in advance.
- **Fisher fusion.** Combine two p-values as −log p₁ − log p₂.
- **Label-free fitness.** See question 6 below.
- **Data leakage / selection on the test set.** Using test data to make a choice (here: which
  features), then reporting the score on that same test data.

## 3. Numbers to remember

| What | Value |
|---|---|
| Supervised, same dataset (5-fold CV) | about 0.995 AUC |
| Supervised, other dataset | 0.42–0.49 AUC |
| Features that flip direction | 11 of 30 |
| Mid-sem selection: chosen on test pairs / chosen properly | 0.65–0.72 / 0.46–0.58 |
| BRACE (CIC-IDS-2017 / UNSW-NB15 / TON_IoT) | 0.980 / 0.972 / 0.998 |
| Supervised with the target's labels | 0.998 / 0.985 / 1.000 |
| Best supervised transfer (oracle choice) | 0.870 / 0.721 / 0.846 |
| kNN detection at 1% false alarms, before → after GA | 0.30→0.53, 0.49→0.71, 0.67→0.79 |
| External dataset CSE-CIC-IDS2018, BRACE | 0.968 (labelled reference 0.981) |
| Calibrated on a different day | AUC 0.922 / 0.796; false alarms 6–23% instead of 1% |

## 4. Likely questions and answers

**Q1. Your mid-semester report said DA-MFS improved cross-dataset performance. Now you say it
does not. Which is it?**
It does not. In the mid-semester code the GA scored every subset on the same six train→test
pairs that we then reported, so the subset was picked using the test labels. We re-ran it on
all 4,095 subsets: picked on the test pairs the gain is 0.15–0.18 AUC; picked using only the
other two datasets the gain disappears. We found this ourselves and corrected it.

**Q2. Why does a supervised model fail on another dataset?**
Three measured reasons. (a) 11 of 30 features flip direction: attacks receive fewer reply
packets than benign flows in UNSW-NB15 and more in TON_IoT. (b) Benign traffic of two datasets
does not overlap; a classifier separates them perfectly. (c) The same attack family (for example
DoS) leaves different flow statistics in each testbed: UNSW→TON DoS transfer is 0.095.

**Q3. Why not use domain adaptation?**
We tried: per-dataset standardisation, quantile alignment, CORAL, Deep CORAL, DANN, and
benign anchoring. Best average is 0.60. They align distributions, but when the feature–label
relationship itself is reversed, alignment cannot help.

**Q4. What is new in BRACE?**
The pieces exist in the statistics literature (semi-supervised novelty detection, conformal
p-values). What we did not find in prior work is their use for the cross-dataset NIDS problem
together with metaheuristic feature selection driven by a label-free fitness on the target
network. We state it as "to our knowledge".

**Q5. Is BRACE just anomaly detection?**
The kNN part is. The second part is not: the benign-vs-unlabelled classifier learns from the
attacks hidden in the stream, and finds attacks that look normal feature by feature, such as
port scans (kNN detects 2%, the classifier 94%).

**Q6. How can a GA select features without labels?**
Fix the threshold from benign flows so that 5% of benign flows would alarm. Then count the
alarms on the unlabelled stream. Alarm rate = π·TPR + (1−π)·α, where π is the unknown attack
share. π and α are the same for every subset, so the subset with the most alarms has the
highest true detection rate. We never need to know which flows are attacks.

**Q7. How much does the GA actually help?**
For the kNN detector at 1% false alarms: +0.11 detection on average, 99 wins in 120 runs.
In AUC the gain is small, and inside the fused detector it adds only +0.007. We report both.

**Q8. Which optimiser is best?**
Grey wolf and whale reach the highest fitness; GA is mid-table; all beat random search; the
differences are small. Greedy forward selection is competitive at a budget of 300 evaluations.

**Q9. Where do you get clean benign traffic in practice?**
From a period the operator trusts: a maintenance window, a sandboxed baseline, or traffic already
reviewed by analysts. It is a time window, not per-flow labelling. A few thousand flows suffice
for the kNN score.

**Q10. What if there are very few attacks in the live traffic?**
The classifier part weakens: on UNSW-NB15 it falls from 0.973 to 0.768 when attacks are 0.5% of
the stream. The kNN part does not use the stream and stays at 0.929. The fusion follows the
better one, but at 0.5% it is slightly below plain kNN, so in that regime kNN alone should be used.

**Q11. What is the biggest weakness?**
Drift of benign traffic. If the calibration sample is from a different day than the monitored
traffic, false alarms rise to 6–9% (CIC-IDS-2017) and 20–23% (CSE-CIC-IDS2018) when 1% was
requested, and AUC falls to 0.922 and 0.796. The calibration sample must be recent and refreshed.

**Q12. Which attacks does it miss?**
Web attacks (7% detected at 5% false alarms) and botnet traffic (34%) on CIC-IDS-2017. They are
few and look like ordinary HTTP flows in 30 flow statistics.

**Q13. Why AUC and not accuracy?**
Attack share is 20%, 64% and 76% in the three datasets. Accuracy rewards guessing the majority
class; our mid-semester cross-dataset accuracy (0.508) was below the majority rate.

**Q14. Did you tune on the test data this time?**
The classifier settings and the fitness were chosen on separate development seeds. CSE-CIC-IDS2018
was not evaluated until the method was frozen, then run once.

**Q15. Why was the fourth dataset added?**
As an external check: a dataset the method was not designed on. BRACE reaches 0.968 there.

**Q16. Is the result statistically significant?**
Over 120 paired runs (Wilcoxon signed-rank): BRACE vs kNN p = 1×10⁻⁹, BRACE vs classifier alone
p = 7×10⁻⁸. The classifier alone vs kNN is not significant (p = 0.93): each wins where the other
fails, which is why we fuse them.

**Q17. What is left to do?**
Automatic drift detection and recalibration, a test that rejects a selected subset when the gain
is within noise, robustness to a contaminated calibration sample, richer features for web and
botnet attacks, and a paper.

## 5. Things not to claim

- Do not say BRACE "solves" cross-dataset generalisation. It avoids the transfer problem and
  has the limits listed above.
- Do not say the GA is the main source of the improvement. The main source is scoring relative
  to the target's own benign traffic.
- Do not quote the mid-semester DA-MFS numbers as a positive result.
- The logo on the report's front page is a placeholder; replace it with the official NSUT logo
  before submission.
