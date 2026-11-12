"""Binary metaheuristic optimisers for feature-subset search (all written from scratch).

Every optimiser maximises fitness(mask) over mask in {0,1}^d and is given the same
budget, counted in UNIQUE fitness evaluations (re-visiting a subset is free because
results are cached). The Evaluator records the best value found after every unique
evaluation, so convergence curves of different algorithms are directly comparable.

    rs     uniform random search                     (reference)
    sfs    greedy sequential forward selection       (deterministic wrapper reference)
    sa     simulated annealing (bit-flip moves)
    ga     genetic algorithm (tournament, uniform crossover, bit-flip mutation, elitism)
    pso    binary particle swarm (V-shaped transfer function)
    de     binary differential evolution (DE/rand/1/bin on [0,1]^d, threshold 0.5)
    gwo    binary grey wolf optimiser
    woa    binary whale optimisation algorithm
    nsga2  NSGA-II, two objectives: maximise fitness, minimise number of features
"""
import numpy as np


class BudgetExhausted(Exception):
    pass


class Evaluator:
    def __init__(self, fitness, d, budget):
        self.f, self.d, self.budget = fitness, d, int(budget)
        self.cache = {}
        self.calls = 0
        self.best_fit, self.best_mask = -np.inf, None
        self.history = []                     # best-so-far after each unique evaluation

    @property
    def evals(self):
        return len(self.cache)

    def __call__(self, mask):
        mask = np.asarray(mask).astype(bool)
        if not mask.any():
            return -np.inf
        key = mask.tobytes()
        self.calls += 1
        if key in self.cache:
            if self.calls > 40 * self.budget:     # converged population that only re-visits old subsets
                raise BudgetExhausted
            return self.cache[key]
        if len(self.cache) >= self.budget:
            raise BudgetExhausted
        v = float(self.f(mask))
        self.cache[key] = v
        if v > self.best_fit:
            self.best_fit, self.best_mask = v, mask.copy()
        self.history.append(self.best_fit)
        return v

    def many(self, masks):
        return np.array([self(m) for m in masks])

    def result(self):
        hist = self.history + [self.best_fit] * (self.budget - len(self.history))
        return {"best_mask": self.best_mask, "best_fit": self.best_fit,
                "history": np.array(hist[:self.budget]), "evals": self.evals,
                "evaluated": self.cache}


def _rand_mask(rng, d, p=0.5):
    m = rng.random(d) < p
    if not m.any():
        m[rng.integers(d)] = True
    return m


def _repair(rng, m):
    if not m.any():
        m[rng.integers(len(m))] = True
    return m


# ----------------------------------------------------------------------------- references
def rs(ev, rng):
    while True:
        ev(_rand_mask(rng, ev.d, rng.uniform(0.1, 0.9)))


def sfs(ev, rng):
    cur = np.zeros(ev.d, bool); cur_fit = -np.inf
    while not cur.all():
        cand = []
        for j in np.flatnonzero(~cur):
            m = cur.copy(); m[j] = True; cand.append((ev(m), j))
        f, j = max(cand)
        if f <= cur_fit:
            break
        cur[j] = True; cur_fit = f
    raise BudgetExhausted                     # greedy search has converged; it does not use the rest of the budget


def sa(ev, rng, t0=0.02, t1=0.0005):
    cur = _rand_mask(rng, ev.d); fc = ev(cur)
    while True:
        frac = min(ev.evals / ev.budget, 1.0)
        T = t0 * (t1 / t0) ** frac
        nb = cur.copy()
        for j in rng.choice(ev.d, size=1 + int(rng.random() < 0.2), replace=False):
            nb[j] = ~nb[j]
        nb = _repair(rng, nb); fn = ev(nb)
        if fn >= fc or rng.random() < np.exp((fn - fc) / T):
            cur, fc = nb, fn


# ----------------------------------------------------------------------------- GA
def ga(ev, rng, pop=30, p_cross=0.9, p_mut=None, tour=3, elite=2):
    d = ev.d; p_mut = p_mut or 1.0 / d
    P = np.array([_rand_mask(rng, d) for _ in range(pop)]); F = ev.many(P)
    while True:
        order = np.argsort(-F); new = [P[i].copy() for i in order[:elite]]
        while len(new) < pop:
            a = P[max(rng.integers(0, pop, tour), key=lambda i: F[i])]
            b = P[max(rng.integers(0, pop, tour), key=lambda i: F[i])]
            c = np.where(rng.random(d) < 0.5, a, b) if rng.random() < p_cross else a.copy()
            c = c ^ (rng.random(d) < p_mut)
            new.append(_repair(rng, c))
        P = np.array(new); F = ev.many(P)


# ----------------------------------------------------------------------------- PSO
def pso(ev, rng, pop=30, w=0.72, c1=1.49, c2=1.49, vmax=4.0):
    """Binary PSO with a V-shaped transfer function (Mirjalili & Lewis, 2013): a bit is FLIPPED with
    probability |tanh(v)|, so a particle that has reached its attractors stops moving. The classic
    sigmoid transfer keeps re-sampling bits at random and does not converge on this problem."""
    d = ev.d
    X = np.array([_rand_mask(rng, d) for _ in range(pop)]); V = rng.uniform(-1, 1, (pop, d))
    F = ev.many(X); PB, PF = X.copy(), F.copy()
    g = PB[PF.argmax()].copy()
    while True:
        r1, r2 = rng.random((pop, d)), rng.random((pop, d))
        Xf = X.astype(float)
        V = np.clip(w * V + c1 * r1 * (PB.astype(float) - Xf) + c2 * r2 * (g.astype(float) - Xf), -vmax, vmax)
        flip = rng.random((pop, d)) < np.abs(np.tanh(V))
        X = np.array([_repair(rng, x) for x in (X ^ flip)]); F = ev.many(X)
        imp = F > PF; PB[imp], PF[imp] = X[imp], F[imp]
        g = PB[PF.argmax()].copy()


# ----------------------------------------------------------------------------- DE
def de(ev, rng, pop=30, Fw=0.5, CR=0.9):
    d = ev.d
    X = rng.random((pop, d)); B = np.array([_repair(rng, x > 0.5) for x in X]); F = ev.many(B)
    while True:
        for i in range(pop):
            a, b, c = rng.choice([j for j in range(pop) if j != i], 3, replace=False)
            v = np.clip(X[a] + Fw * (X[b] - X[c]), 0, 1)
            cross = rng.random(d) < CR; cross[rng.integers(d)] = True
            u = np.where(cross, v, X[i]); fu = ev(_repair(rng, u > 0.5))
            if fu >= F[i]:
                X[i], F[i] = u, fu


# ----------------------------------------------------------------------------- GWO
def gwo(ev, rng, pop=30):
    d = ev.d
    X = np.array([_rand_mask(rng, d) for _ in range(pop)]).astype(float); F = ev.many(X > 0.5)
    while True:
        a = 2.0 * (1.0 - min(ev.evals / ev.budget, 1.0))
        lead = X[np.argsort(-F)[:3]].copy()
        new = np.zeros_like(X)
        for L in lead:
            A = 2 * a * rng.random((pop, d)) - a; C = 2 * rng.random((pop, d))
            new += L - A * np.abs(C * L - X)
        new /= 3.0
        Xb = rng.random((pop, d)) < 1.0 / (1.0 + np.exp(-10 * (new - 0.5)))
        Xb = np.array([_repair(rng, x) for x in Xb]); Fn = ev.many(Xb)
        X, F = Xb.astype(float), Fn


# ----------------------------------------------------------------------------- WOA
def woa(ev, rng, pop=30, b=1.0):
    d = ev.d
    X = np.array([_rand_mask(rng, d) for _ in range(pop)]).astype(float); F = ev.many(X > 0.5)
    best, bf = X[F.argmax()].copy(), F.max()
    while True:
        a = 2.0 * (1.0 - min(ev.evals / ev.budget, 1.0))
        new = np.empty_like(X)
        for i in range(pop):
            A = 2 * a * rng.random(d) - a; C = 2 * rng.random(d)
            if rng.random() < 0.5:
                ref = best if np.abs(A).mean() < 1 else X[rng.integers(pop)]
                new[i] = ref - A * np.abs(C * ref - X[i])
            else:
                l = rng.uniform(-1, 1)
                new[i] = np.abs(best - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + best
        Xb = rng.random((pop, d)) < 1.0 / (1.0 + np.exp(-10 * (new - 0.5)))
        Xb = np.array([_repair(rng, x) for x in Xb]); F = ev.many(Xb); X = Xb.astype(float)
        if F.max() > bf:
            best, bf = X[F.argmax()].copy(), F.max()


# ----------------------------------------------------------------------------- NSGA-II
def _nd_sort(obj):
    """obj: [n, m], all minimised. Returns list of fronts (index arrays)."""
    n = len(obj); S = [[] for _ in range(n)]; cnt = np.zeros(n, int); fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(obj[p] <= obj[q]) and np.any(obj[p] < obj[q]):
                S[p].append(q)
            elif np.all(obj[q] <= obj[p]) and np.any(obj[q] < obj[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        fronts.append(nxt); i += 1
    return [np.array(f) for f in fronts[:-1]]


def _crowding(obj):
    n, m = obj.shape; cd = np.zeros(n)
    for k in range(m):
        o = np.argsort(obj[:, k]); cd[o[0]] = cd[o[-1]] = np.inf
        span = obj[o[-1], k] - obj[o[0], k]
        if span > 0:
            cd[o[1:-1]] += (obj[o[2:], k] - obj[o[:-2], k]) / span
    return cd


def nsga2(ev, rng, pop=30, p_cross=0.9, p_mut=None):
    d = ev.d; p_mut = p_mut or 1.0 / d
    P = np.array([_rand_mask(rng, d, rng.uniform(0.1, 0.9)) for _ in range(pop)]); F = ev.many(P)
    while True:
        obj = np.c_[-F, P.sum(1)]
        fronts = _nd_sort(obj); rank = np.empty(pop, int); cd = np.empty(pop)
        for r, fr in enumerate(fronts):
            rank[fr] = r; cd[fr] = _crowding(obj[fr])

        def pick():
            i, j = rng.integers(0, pop, 2)
            return i if (rank[i], -cd[i]) < (rank[j], -cd[j]) else j
        Q = []
        while len(Q) < pop:
            a, b = P[pick()], P[pick()]
            c = np.where(rng.random(d) < 0.5, a, b) if rng.random() < p_cross else a.copy()
            Q.append(_repair(rng, c ^ (rng.random(d) < p_mut)))
        Q = np.array(Q); FQ = ev.many(Q)
        R = np.vstack([P, Q]); FR = np.r_[F, FQ]; objR = np.c_[-FR, R.sum(1)]
        keep = []
        for fr in _nd_sort(objR):
            if len(keep) + len(fr) <= pop:
                keep.extend(fr.tolist())
            else:
                c = _crowding(objR[fr]); keep.extend(fr[np.argsort(-c)][:pop - len(keep)].tolist()); break
        P, F = R[keep], FR[keep]


ALGOS = {"rs": rs, "sfs": sfs, "sa": sa, "ga": ga, "pso": pso, "de": de, "gwo": gwo, "woa": woa, "nsga2": nsga2}


def optimise(algo, fitness, d, budget, seed, **kw):
    """Run one optimiser. Returns dict(best_mask, best_fit, history[budget], evals, evaluated)."""
    ev = Evaluator(fitness, d, budget)
    rng = np.random.default_rng(seed)
    try:
        ALGOS[algo](ev, rng, **kw)
    except BudgetExhausted:
        pass
    return ev.result()


def pareto_front(evaluated, d):
    """Non-dominated (fitness up, size down) subsets among everything an optimiser evaluated."""
    masks = np.array([np.frombuffer(k, dtype=bool) for k in evaluated]); fit = np.array(list(evaluated.values()))
    size = masks.sum(1); out = []
    for k in np.unique(size):
        i = np.flatnonzero(size == k); j = i[fit[i].argmax()]
        out.append((int(k), float(fit[j]), masks[j]))
    front, best = [], -np.inf
    for k, f, m in sorted(out, key=lambda t: t[0]):
        if f > best:
            front.append((k, f, m)); best = f
    return front
