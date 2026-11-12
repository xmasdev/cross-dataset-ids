"""Data access and the split protocols used by every experiment.

Deployment split of a target dataset (make_target_split)
--------------------------------------------------------
    cal    benign calibration flows ("traffic recorded while the network was known to be clean")
    adapt  unlabelled stream the method may adapt on (labels exist but are hidden from the method)
    test   disjoint stream used only for scoring

The three parts never overlap, so nothing a method does with `cal` and `adapt`
can leak into the `test` numbers.
"""
import numpy as np
import pandas as pd

from .harmonise import get_base
from .features import derive, POOL_V2

_cache = {}


def load(name, pool=None, payload=False):
    """(X float32 [n, d], y int [n], meta DataFrame) for one dataset."""
    key = (name, payload)
    if key not in _cache:
        base = get_base(name)
        if payload:
            base = base.dropna(subset=["src_payload", "dst_payload"]).reset_index(drop=True)
        X = derive(base, payload=payload)
        _cache[key] = (X, base["label"].to_numpy().astype(np.int8),
                       base[["attack_cat", "family", "grp"]].astype(str))
    X, y, meta = _cache[key]
    cols = list(pool) if pool is not None else POOL_V2
    return X[cols].to_numpy(np.float32), y, meta


def stratified_cap(y, n, rng, strata=None):
    """Indices of a random sample of at most n rows, proportional within strata (default: the label)."""
    if n is None or len(y) <= n:
        return np.arange(len(y))
    s = np.asarray(strata if strata is not None else y)
    idx = []
    for v in pd.unique(s):
        ids = np.flatnonzero(s == v)
        k = max(1, int(round(n * len(ids) / len(s))))
        idx.append(rng.choice(ids, size=min(k, len(ids)), replace=False))
    return np.sort(np.concatenate(idx))


def make_target_split(y, seed, n_cal=6000, n_adapt=30000, prevalence=None, test_cap=200000,
                      cal_idx=None, pool_idx=None):
    """Random deployment split of one dataset.

    prevalence : attack fraction of the unlabelled adaptation stream (None = the dataset's own).
                 The test stream always keeps the dataset's natural composition; AUC, TPR and FPR
                 do not depend on it.
    cal_idx / pool_idx : optional explicit candidate rows for the calibration set and for the
                 adapt/test streams (used by the temporal protocol: calibrate on one day, run on others).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    allr = np.arange(len(y))
    if cal_idx is None:
        ben = rng.permutation(allr[y == 0])
        cal, ben_rest = ben[:n_cal], ben[n_cal:]
        att = rng.permutation(allr[y == 1])
    else:
        cal_idx = np.asarray(cal_idx); pool_idx = np.asarray(pool_idx)
        cal = rng.permutation(cal_idx[y[cal_idx] == 0])[:n_cal]
        ben_rest = rng.permutation(pool_idx[y[pool_idx] == 0])
        att = rng.permutation(pool_idx[y[pool_idx] == 1])
    hb, ha = len(ben_rest) // 2, len(att) // 2
    ben_a, ben_t = ben_rest[:hb], ben_rest[hb:]
    att_a, att_t = att[:ha], att[ha:]

    p = len(att_a) / max(len(att_a) + len(ben_a), 1) if prevalence is None else float(prevalence)
    na = int(round(n_adapt * p)); nb = n_adapt - na
    if nb > len(ben_a):
        nb = len(ben_a); na = int(round(nb * p / max(1 - p, 1e-9)))
    if na > len(att_a):
        na = len(att_a); nb = min(len(ben_a), int(round(na * (1 - p) / max(p, 1e-9))))
    adapt = rng.permutation(np.concatenate([att_a[:na], ben_a[:nb]]))

    test = np.concatenate([att_t, ben_t])
    if test_cap is not None and len(test) > test_cap:
        test = test[stratified_cap(y[test], test_cap, rng)]
    return {"cal": cal, "adapt": adapt, "test": np.sort(test)}
