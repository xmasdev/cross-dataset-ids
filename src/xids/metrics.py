"""Evaluation metrics. Labels: 1 = attack, 0 = benign. Scores: higher = more attack-like."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, average_precision_score)


def auc(y, s):
    """ROC-AUC with average ranks for ties (fast, no sklearn overhead)."""
    y = np.asarray(y).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auc_two(pos, neg):
    """AUC that a score drawn from `pos` exceeds one drawn from `neg`."""
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    return auc(y, s)


def threshold_metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else float("nan")
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y, pred), "tpr": tpr, "fpr": fpr}


def score_metrics(y, s):
    return {"auc": auc(y, s), "auprc": float(average_precision_score(y, s))}


def tpr_at_fpr(y, s, alpha):
    """TPR when the threshold is set to give FPR = alpha on the evaluated benign rows (oracle threshold)."""
    y = np.asarray(y).astype(bool)
    thr = np.quantile(np.asarray(s)[~y], 1.0 - alpha)
    return float((np.asarray(s)[y] > thr).mean())


def all_metrics(y, s, thr=None, alphas=(0.01, 0.05)):
    """Ranking metrics, plus threshold metrics when a decision threshold is given."""
    out = score_metrics(y, s)
    for a in alphas:
        out[f"tpr@{int(a*100)}%fpr"] = tpr_at_fpr(y, s, a)
    if thr is not None:
        out.update(threshold_metrics(y, (np.asarray(s) > thr).astype(int)))
    return out
