"""Target-side context: everything a method may use when it is deployed on one network.

    ref    benign calibration flows used as the kNN reference set and to fit the anchor
    sel    benign calibration flows used only inside the label-free selection fitness
    thr    benign calibration flows used only to set the alarm threshold
    adapt  unlabelled stream (labels hidden from the method)
    test   disjoint stream, used only for the reported numbers

Label-free fitness
------------------
For a feature subset m, let s_m(x) be the kNN anomaly score. With F_U the score
distribution of the unlabelled stream (attack fraction pi) and F_B that of benign flows,

    AUC(adapt vs sel)   = 0.5 + pi * (AUC_true - 0.5)                    ("pu_auc")
    alarm rate at FPR a = pi * TPR_a + (1 - pi) * a                      ("alarm")

so both are increasing functions of the true detection quality and neither needs a
single attack label. "pu_pauc" is the alarm rate averaged over a in (0, a_max].
"""
import numpy as np
import torch

from .anchor import make_anchor
from .data import load, make_target_split
from .knn import knn_score, auc_t, to_dev, DEV
from .metrics import auc as auc_np


class TargetContext:
    def __init__(self, name, seed, pool, n_cal=6000, n_adapt=30000, prevalence=None, anchor="quantile",
                 k=5, test_cap=200000, frac=(0.4, 0.3, 0.3), split=None, data=None):
        self.name, self.seed, self.pool, self.k = name, seed, list(pool), k
        X, y, meta = data if data is not None else load(name, pool)
        self.X, self.y, self.meta = X, y, meta
        sp = split if split is not None else make_target_split(y, seed, n_cal, n_adapt, prevalence, test_cap)
        self.split = sp
        cal = sp["cal"]; n = len(cal)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        self.i_ref, self.i_sel, self.i_thr = cal[:a], cal[a:b], cal[b:]
        self.i_adapt, self.i_test = sp["adapt"], sp["test"]
        self.anchor = make_anchor(anchor).fit(X[self.i_ref])
        t = lambda idx: to_dev(self.anchor.transform(X[idx]))
        self.Z = {"ref": t(self.i_ref), "sel": t(self.i_sel), "thr": t(self.i_thr),
                  "adapt": t(self.i_adapt), "test": t(self.i_test)}
        self.y_adapt, self.y_test = y[self.i_adapt], y[self.i_test]
        self.yt = torch.as_tensor(self.y_test.astype(bool), device=DEV)
        self.d = len(self.pool)

    # ------------------------------------------------------------------ scoring
    def scores(self, mask, part, weights=None):
        cols = np.flatnonzero(np.asarray(mask).astype(bool))
        return knn_score(self.Z[part], self.Z["ref"], k=self.k, cols=cols, weights=weights)

    def fitness(self, mask, kind="pu_auc", alpha=0.05):
        """Label-free fitness of a subset (uses ref, sel and the unlabelled adapt stream only)."""
        su, sb = self.scores(mask, "adapt"), self.scores(mask, "sel")
        return fitness_from_scores(su, sb, kind, alpha)

    def test_auc(self, mask):
        s = self.scores(mask, "test")
        return auc_t(s[self.yt], s[~self.yt])

    def evaluate(self, mask, alphas=(0.01, 0.05)):
        """Reported numbers for a subset: threshold from `thr`, metrics on `test`."""
        s = self.scores(mask, "test"); sthr = self.scores(mask, "thr")
        out = {"auc": auc_t(s[self.yt], s[~self.yt])}
        for a in alphas:
            th = torch.quantile(sthr, 1 - a)
            al = s > th
            out[f"tpr@{a}"] = float(al[self.yt].float().mean()); out[f"fpr@{a}"] = float(al[~self.yt].float().mean())
        out["bal_acc@0.05"] = 0.5 * (out["tpr@0.05"] + 1 - out["fpr@0.05"])
        return out


def fitness_from_scores(su, sb, kind="pu_auc", alpha=0.05):
    if kind == "pu_auc":
        return auc_t(su, sb)
    if kind == "alarm":
        return float((su > torch.quantile(sb, 1 - alpha)).float().mean())
    if kind == "pu_pauc":
        qs = torch.linspace(1 - alpha, 1 - alpha / 10, 10, device=sb.device)
        th = torch.quantile(sb, qs)
        return float((su[:, None] > th[None, :]).float().mean())
    raise ValueError(kind)


def all_masks(d):
    return np.array([[(m >> j) & 1 for j in range(d)] for m in range(1, 2 ** d)], dtype=bool)
