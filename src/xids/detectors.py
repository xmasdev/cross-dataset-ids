"""Benign-only anomaly detectors. Each is fitted on benign reference rows (already transformed)
and returns an anomaly score (higher = more unusual) for query rows.

    detect(name, Zref, Zq, seed) -> np.ndarray
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from .knn import knn_score, DEV

DETECTORS = ["knn1", "knn5", "knn10", "knn20", "lof", "iforest", "ocsvm", "gmm", "mahalanobis",
             "ecod", "hbos", "pca", "autoencoder"]


def _ae(Zref, Zq, seed, epochs=60):
    torch.manual_seed(seed)
    d = Zref.shape[1]; h = max(4, d // 2); b = max(2, d // 4)
    net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, b), nn.ReLU(), nn.Linear(b, h), nn.ReLU(), nn.Linear(h, d)).to(DEV)
    X = torch.as_tensor(Zref, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        for i in range(0, len(X), 256):
            xb = X[perm[i:i + 256]]; loss = ((net(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Zq), 65536):
            q = torch.as_tensor(Zq[i:i + 65536], dtype=torch.float32, device=DEV)
            out.append(((net(q) - q) ** 2).mean(1).cpu())
    return torch.cat(out).numpy()


def detect(name, Zref, Zq, seed=0):
    Zref = np.asarray(Zref, dtype=np.float32); Zq = np.asarray(Zq, dtype=np.float32)
    if name.startswith("knn"):
        return knn_score(Zq, Zref, k=int(name[3:])).cpu().numpy()
    if name == "lof":
        m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=8).fit(Zref)
        return -m.score_samples(Zq)
    if name == "iforest":
        m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=8).fit(Zref)
        return -m.score_samples(Zq)
    if name == "ocsvm":
        rng = np.random.default_rng(seed); r = Zref[rng.choice(len(Zref), min(len(Zref), 3000), replace=False)]
        m = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(r)
        return np.concatenate([-m.score_samples(Zq[i:i + 20000]) for i in range(0, len(Zq), 20000)])
    if name == "gmm":
        m = GaussianMixture(8, covariance_type="full", reg_covar=1e-2, random_state=seed).fit(Zref)
        return -m.score_samples(Zq)
    if name == "mahalanobis":
        mu = Zref.mean(0); P = np.linalg.inv(np.cov(Zref.T) + 1e-2 * np.eye(Zref.shape[1]))
        dq = Zq - mu
        return np.einsum("ij,jk,ik->i", dq, P, dq)
    if name == "ecod":                      # per-feature two-sided tail surprisal under the reference ECDF, summed
        n = len(Zref); s = np.sort(Zref, axis=0); out = np.zeros(len(Zq))
        for j in range(Zq.shape[1]):
            F = (np.searchsorted(s[:, j], Zq[:, j], "left") + np.searchsorted(s[:, j], Zq[:, j], "right")) / (2.0 * n)
            F = np.clip(F, 0.5 / n, 1 - 0.5 / n)
            out += -np.log(2 * np.minimum(F, 1 - F))
        return out
    if name == "hbos":
        out = np.zeros(len(Zq))
        for j in range(Zq.shape[1]):
            lo, hi = Zref[:, j].min(), Zref[:, j].max()
            edges = np.linspace(lo, hi + 1e-9, 21); h, _ = np.histogram(Zref[:, j], edges)
            dens = (h + 0.5) / (h.sum() + 10)
            b = np.clip(np.searchsorted(edges, Zq[:, j], "right") - 1, 0, 19)
            inside = (Zq[:, j] >= lo) & (Zq[:, j] <= hi)
            out += -np.log(np.where(inside, dens[b], 0.5 / (h.sum() + 10)))
        return out
    if name == "pca":
        m = PCA(n_components=0.95, random_state=seed).fit(Zref)
        return ((Zq - m.inverse_transform(m.transform(Zq))) ** 2).sum(1)
    if name == "autoencoder":
        return _ae(Zref, Zq, seed)
    raise KeyError(name)
