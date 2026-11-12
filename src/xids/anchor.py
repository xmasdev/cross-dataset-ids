"""Benign-anchored representations.

Every transform here is fitted on a *reference* sample and applied to other rows.
What differs between the alignment levels compared in the report is only which
sample is used as reference:

    marginal alignment  -> reference = all (unlabelled) rows of the dataset
    benign anchoring    -> reference = a benign calibration sample of the dataset

`quantile` is the transform used by the proposed method: each feature value is
replaced by the normal score of its mid-rank position in the reference ECDF, so
that "0" means "typical for this network's benign traffic" and large |z| means
"rare for this network's benign traffic", whatever the raw units are.
"""
import numpy as np
from scipy.stats import norm


class QuantileAnchor:
    """Per-feature mid-rank ECDF -> standard-normal score."""

    def fit(self, ref):
        ref = np.asarray(ref, dtype=np.float64)
        self.sorted_ = np.sort(ref, axis=0)
        self.n_ = ref.shape[0]
        return self

    def cdf(self, X):
        X = np.asarray(X, dtype=np.float64)
        F = np.empty_like(X)
        for j in range(X.shape[1]):
            s = self.sorted_[:, j]
            F[:, j] = np.searchsorted(s, X[:, j], "left") + np.searchsorted(s, X[:, j], "right")
        F /= 2.0 * self.n_
        return np.clip(F, 0.5 / self.n_, 1.0 - 0.5 / self.n_)

    def transform(self, X):
        return norm.ppf(self.cdf(X)).astype(np.float32)


class RobustZAnchor:
    """Non-saturating alternative: (log1p(x) - median) / robust scale of the reference."""

    @staticmethod
    def _log(X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))

    def fit(self, ref):
        L = self._log(ref)
        self.med_ = np.median(L, axis=0)
        q75, q25 = np.percentile(L, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        std = L.std(axis=0)
        scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1.0))
        self.scale_ = scale
        return self

    def transform(self, X):
        return np.clip((self._log(X) - self.med_) / self.scale_, -50, 50).astype(np.float32)


class LogStandard:
    """Plain log1p + z-score with the reference mean / std (the usual preprocessing)."""

    def fit(self, ref):
        L = RobustZAnchor._log(ref)
        self.mu_ = L.mean(axis=0)
        sd = L.std(axis=0)
        self.sd_ = np.where(sd > 1e-9, sd, 1.0)
        return self

    def transform(self, X):
        return ((RobustZAnchor._log(X) - self.mu_) / self.sd_).astype(np.float32)


ANCHORS = {"quantile": QuantileAnchor, "robust_z": RobustZAnchor, "log_std": LogStandard}


def make_anchor(kind):
    return ANCHORS[kind]()
