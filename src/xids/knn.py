"""GPU k-nearest-neighbour anomaly scoring against a benign reference set.

score(x) = mean Euclidean distance from x to its k nearest reference flows,
           computed on the selected feature columns and divided by sqrt(#features)
           so that scores of different subsets are on a comparable scale.
"""
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_dev(a):
    return a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEV)


def _sqdist(q, r):
    d = (q * q).sum(1, keepdim=True) + (r * r).sum(1)[None, :] - 2.0 * (q @ r.T)
    return d.clamp_min_(0.0)


@torch.no_grad()
def knn_score(q, ref, k=5, cols=None, weights=None, loo=False, chunk=16384):
    """Anomaly score of every row of q. Returns a torch tensor on the device.

    cols    : indices of the features to use (None = all)
    weights : optional non-negative per-feature weights (applied to squared differences)
    loo     : q is the reference set itself -> ignore each point's zero self-distance
    """
    q, ref = to_dev(q), to_dev(ref)
    if cols is not None:
        cols = torch.as_tensor(np.asarray(cols), device=DEV, dtype=torch.long)
        q, ref = q[:, cols], ref[:, cols]
    d_eff = q.shape[1]
    if weights is not None:
        w = to_dev(weights)
        if cols is not None:
            w = w[cols]
        sw = w.sqrt()
        q, ref = q * sw, ref * sw
        d_eff = float(w.sum().clamp_min(1e-9))
    kk = min(k + (1 if loo else 0), ref.shape[0])
    out = []
    for i in range(0, q.shape[0], chunk):
        d = _sqdist(q[i:i + chunk], ref)
        v = d.topk(kk, dim=1, largest=False).values.sqrt_()
        if loo:
            v = v[:, 1:]
        out.append(v.mean(1))
    return torch.cat(out) / (d_eff ** 0.5)


def auc_t(pos, neg):
    """AUC (pos scores above neg scores) on the device, with average ranks for ties."""
    s = torch.cat([pos, neg]).double()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, device=s.device, dtype=torch.double)
    us, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)
    ranks = (torch.zeros_like(us).scatter_add_(0, inv, ranks) / cnt)[inv]
    npos, nneg = len(pos), len(neg)
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
