"""Supervised model zoo for the transfer benchmark.

fit_predict(name, Xtr, ytr, Xte, seed, Xtgt_unlab=None) -> attack score in [0, 1] for every row of Xte.

Inputs are expected to be already preprocessed (see experiments/02_supervised_benchmark.py).
Tree models are insensitive to the monotone preprocessing; linear / neural / distance models need it.

Domain-adaptation variants (need unlabelled target rows, passed as Xtgt_unlab):
    dann    domain-adversarial network (gradient reversal)
    dcoral  Deep CORAL (covariance alignment of the last hidden layer)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NJ = 12


def _spw(y):
    return float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)


def _sk(name, y, seed):
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
    if name == "linsvm":
        return SGDClassifier(loss="hinge", alpha=1e-5, max_iter=30, tol=None, class_weight="balanced", random_state=seed)
    if name == "nb":
        return GaussianNB()
    if name == "dt":
        return DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced", n_jobs=NJ, random_state=seed)
    if name == "et":
        return ExtraTreesClassifier(200, min_samples_leaf=5, class_weight="balanced", n_jobs=NJ, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", scale_pos_weight=_spw(y), n_jobs=NJ, random_state=seed, verbosity=0)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, class_weight="balanced", n_jobs=NJ, random_state=seed, verbose=-1)
    raise KeyError(name)


SK_MODELS = ["majority", "lr", "linsvm", "nb", "dt", "rf", "et", "hgb", "xgb", "lgbm"]
TORCH_MODELS = ["knn", "mlp", "ftt", "dann", "dcoral"]
TREE_MODELS = {"dt", "rf", "et", "hgb", "xgb", "lgbm", "majority"}


# ----------------------------------------------------------------------------- kNN (GPU)
@torch.no_grad()
def _knn_clf(Xtr, ytr, Xte, k=5, chunk=1536):
    R = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV); yr = torch.as_tensor(ytr, dtype=torch.float32, device=DEV)
    rn = (R * R).sum(1)[None, :]; out = []
    for i in range(0, len(Xte), chunk):
        q = torch.as_tensor(Xte[i:i + chunk], dtype=torch.float32, device=DEV)
        d = (q * q).sum(1, keepdim=True) + rn - 2 * q @ R.T
        idx = d.topk(k, dim=1, largest=False).indices
        out.append(yr[idx].mean(1).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- neural nets
class MLP(nn.Module):
    def __init__(self, d, h=256, p=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(h, h // 2), nn.ReLU())
        self.head = nn.Linear(h // 2, 1)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)


class FTTransformer(nn.Module):
    """Small FT-Transformer: one learned token per numeric feature + CLS, pre-norm encoder."""

    def __init__(self, d, dt=32, heads=4, layers=2, p=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, dt) * 0.1); self.b = nn.Parameter(torch.zeros(d, dt))
        self.cls = nn.Parameter(torch.zeros(1, 1, dt))
        layer = nn.TransformerEncoderLayer(dt, heads, dt * 2, p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(dt), nn.ReLU(), nn.Linear(dt, 1))

    def features(self, x):
        t = x[:, :, None] * self.w[None] + self.b[None]
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.enc(t)[:, 0]

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def _coral(a, b):
    ca = torch.cov(a.T); cb = torch.cov(b.T)
    return ((ca - cb) ** 2).sum() / (4 * a.shape[1] ** 2)


def _train_net(kind, Xtr, ytr, Xte, seed, Xu=None, epochs=12, bs=1024, lr=2e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    net = (FTTransformer(d) if kind == "ftt" else MLP(d)).to(DEV)
    params = list(net.parameters())
    dom = None
    if kind == "dann":
        dom = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEV); params += list(dom.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV); yt = torch.as_tensor(ytr, dtype=torch.float32, device=DEV)
    Xut = torch.as_tensor(Xu, dtype=torch.float32, device=DEV) if Xu is not None else None
    pw = torch.tensor(_spw(ytr), device=DEV)
    n = len(Xt); steps = epochs * (n // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    step = 0
    for ep in range(epochs):
        net.train(); perm = torch.as_tensor(rng.permutation(n), device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 8:
                continue
            xb, yb = Xt[idx], yt[idx]
            if kind in ("dann", "dcoral") and Xut is not None:
                xu = Xut[torch.randint(0, len(Xut), (len(idx),), device=DEV)]
                fs, fu = net.features(xb), net.features(xu)
                loss = nn.functional.binary_cross_entropy_with_logits(net.head(fs).squeeze(1), yb, pos_weight=pw)
                prog = step / steps
                if kind == "dann":
                    lam = 2.0 / (1.0 + np.exp(-10 * prog)) - 1.0
                    f = _GradRev.apply(torch.cat([fs, fu]), lam)
                    dl = torch.cat([torch.zeros(len(fs)), torch.ones(len(fu))]).to(DEV)
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(dom(f).squeeze(1), dl)
                else:
                    loss = loss + 10.0 * prog * _coral(fs, fu)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 16384):
            out.append(torch.sigmoid(net(torch.as_tensor(Xte[i:i + 16384], dtype=torch.float32, device=DEV))).cpu())
    return torch.cat(out).numpy()


def fit_predict(name, Xtr, ytr, Xte, seed, Xtgt_unlab=None):
    if name == "knn":
        return _knn_clf(Xtr, ytr, Xte)
    if name in ("mlp", "ftt", "dann", "dcoral"):
        return _train_net(name, Xtr, ytr, Xte, seed, Xu=Xtgt_unlab, epochs=8 if name == "ftt" else 12)
    m = _sk(name, ytr, seed).fit(Xtr, ytr)
    if hasattr(m, "predict_proba"):
        return m.predict_proba(Xte)[:, 1]
    s = m.decision_function(Xte)
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
