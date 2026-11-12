"""Benign-vs-unlabelled classification (semi-supervised novelty detection, Blanchard et al. 2010).

Train a classifier to tell benign calibration flows (class 0) from the unlabelled stream (class 1).
Because the unlabelled stream is a mixture  pi * attacks + (1 - pi) * benign, the optimal score of
this classifier is an increasing function of the attack / benign likelihood ratio, i.e. it ranks
flows the way a fully supervised in-network detector would, without a single attack label.

Everything that needs tuning is tuned on a held-out part of the SAME two samples (early stopping on
benign-vs-unlabelled log-loss), so no label is used anywhere.
"""
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


def pu_fit(Zb, Zu, seed=0, cols=None, n_estimators=600, num_leaves=15, min_child=40, lr=0.05):
    rng = np.random.default_rng(seed)
    if cols is not None:
        Zb, Zu = Zb[:, cols], Zu[:, cols]
    ib, iu = rng.permutation(len(Zb)), rng.permutation(len(Zu))
    vb, vu = ib[:len(ib) // 5], iu[:len(iu) // 5]; tb, tu = ib[len(ib) // 5:], iu[len(iu) // 5:]
    Xtr = np.vstack([Zb[tb], Zu[tu]]); ytr = np.r_[np.zeros(len(tb)), np.ones(len(tu))]
    Xva = np.vstack([Zb[vb], Zu[vu]]); yva = np.r_[np.zeros(len(vb)), np.ones(len(vu))]
    m = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=lr, min_child_samples=min_child,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, class_weight="balanced",
                       n_jobs=8, random_state=seed, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
          callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
    return m


def pu_predict(m, Z, cols=None):
    Z = Z if cols is None else Z[:, cols]
    return m.predict_proba(Z)[:, 1]


def pu_bag_fit(Zb, Zu, seed=0, K=5, n_estimators=100, num_leaves=7, min_child=100, lr=0.05):
    """Frozen variant used by the final method: K small, strongly regularised boosters, each trained
    without one fold of the benign and of the unlabelled flows; no early stopping (at low attack
    prevalence the validation log-loss is too flat to stop on). Chosen on development seeds >= 100."""
    rng = np.random.default_rng(seed); fb = rng.integers(0, K, len(Zb)); fu = rng.integers(0, K, len(Zu)); ms = []
    for k in range(K):
        tb, tu = fb != k, fu != k
        m = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=lr, min_child_samples=min_child,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, class_weight="balanced",
                           n_jobs=8, random_state=seed + k, verbose=-1)
        ms.append(m.fit(np.vstack([Zb[tb], Zu[tu]]), np.r_[np.zeros(tb.sum()), np.ones(tu.sum())]))
    return ms


def pu_bag_predict(ms, Z):
    return np.mean([m.predict_proba(Z)[:, 1] for m in ms], axis=0)
