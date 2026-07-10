"""Compression-clustered Naive Bayes — handling wide, correlated feature
spaces without dropping features.

Flat NB assumes feature independence; correlated features get multiplied as
independent votes, so probabilities blow out to the extremes (the demo's
+19pt held-out overconfidence). Instead of trimming features, this groups
them and votes once per group:

  1. CLUSTER features by an MDL / compression criterion — merge X and Y when
     coding them jointly saves more bits than the joint table costs
     (N·I(X;Y) > ½·(|X|-1)(|Y|-1)·log2 N). Correlated features collapse into
     a theme; independent ones stay singletons.

  2. Reduce each multi-feature cluster to one THEME variable (a coarse,
     outcome-aligned summary), and treat the members as CONDITIONAL nuance on
     that theme rather than independent outcome-votes.

  3. PREDICT with a hierarchical NB: the themes (decorrelated, few) vote on
     the outcome; an optional tempered within-cluster residual lets a member
     refine the theme's vote (`leadership:high | {quality}`) without
     double-counting.

This is a Python proof-of-concept on the demo data. The real Aito engine is a
separate service; this validates the method and the calibration gain before
porting. Evaluated leakage-free (k-fold) — see book/test_03.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

COMPANIES_CSV = Path("data/companies.csv")
CLASSES = ["disaster", "poor", "market", "good", "great"]
RANK = {c: i for i, c in enumerate(CLASSES)}  # ordinal severity, for theme encoding
UPSIDE = {"good", "great"}
MISSING = "(missing)"
ALPHA = 1.0
DEFAULT_FEATURES = [
    "market_position", "moat_type", "moat_strength", "market_quality",
    "leadership_quality", "capital_allocation", "strategic_clarity",
    "execution_track_record", "founder_still_ceo", "sector",
    "valuation_bucket", "growth_bucket", "leverage_bucket",
    "profitability_bucket", "momentum_bucket", "volatility_bucket",
]


# ── data ────────────────────────────────────────────────────────

def load_dataset(features: list[str] | None = None, csv: Path = COMPANIES_CSV):
    features = features or DEFAULT_FEATURES
    df = pd.read_csv(csv, low_memory=False)
    df = df[df["outcome_bucket"].isin(CLASSES)].copy()
    for f in features:
        if f not in df.columns:
            df[f] = MISSING
        df[f] = df[f].map(lambda v: MISSING if (pd.isna(v) or v == "") else str(v))
    return df.reset_index(drop=True), features


def stratified_folds(df: pd.DataFrame, k: int) -> np.ndarray:
    fold = np.zeros(len(df), dtype=int)
    key = (df["ticker"].astype(str) + "_" + df["vintage_year"].astype(str)).tolist()
    for c in CLASSES:
        idx = [i for i in range(len(df)) if df["outcome_bucket"].iloc[i] == c]
        idx.sort(key=lambda i: key[i])
        for j, i in enumerate(idx):
            fold[i] = j % k
    return fold


# ── information measures ────────────────────────────────────────

def _mi_nats(a: pd.Series, b: pd.Series) -> float:
    n = len(a)
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    joint = pd.crosstab(a, b) / n
    mi = 0.0
    for va in joint.index:
        for vb in joint.columns:
            p = joint.loc[va, vb]
            if p > 0:
                mi += p * math.log(p / (pa[va] * pb[vb]))
    return mi


def nmi(a: pd.Series, b: pd.Series) -> float:
    mi = _mi_nats(a, b)
    ha = -sum(p * math.log(p) for p in a.value_counts(normalize=True) if p > 0)
    hb = -sum(p * math.log(p) for p in b.value_counts(normalize=True) if p > 0)
    d = math.sqrt(ha * hb)
    return mi / d if d > 0 else 0.0


def _mdl_merge_gain(a: pd.Series, b: pd.Series) -> float:
    """Bits saved by coding a,b jointly minus the joint table's cost (BIC).
    Positive ⇒ the pair compresses ⇒ worth merging."""
    n = len(a)
    bits_saved = n * _mi_nats(a, b) / math.log(2)
    param_cost = 0.5 * (a.nunique() - 1) * (b.nunique() - 1) * math.log2(n)
    return bits_saved - param_cost


# ── compression clustering (agglomerative, MDL stop) ────────────

def cluster_features(df: pd.DataFrame, features: list[str],
                     linkage_mode: str = "complete", min_nmi: float = 0.30) -> list[list[str]]:
    """Greedily merge feature clusters by a compression criterion.

    A merge requires BOTH: it still saves bits (MDL gain > 0) AND the weakest
    cross-pair shares at least `min_nmi` of its entropy. The NMI floor is the
    resolution knob: pure BIC-MDL under-penalises at N~10^3 (real but tiny
    mutual information pervades these features), so it cascades into one mega
    theme. `min_nmi` keeps clusters to genuinely substitutable features.

      min_nmi ≈ 0.30 → tight themes ({leadership,execution,strategy,capital},
                       {market_position,moat_strength})
      min_nmi = 0    → pure MDL (looser).

    linkage_mode "complete" links by the WEAKEST cross-pair (every member-pair
    must compress); "average" by the mean.
    """
    clusters = [[f] for f in features]
    gain, nmiv = {}, {}
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            key = tuple(sorted((features[i], features[j])))
            gain[key] = _mdl_merge_gain(df[features[i]], df[features[j]])
            nmiv[key] = nmi(df[features[i]], df[features[j]])

    def agg(d, ca, cb):
        vals = [d[tuple(sorted((x, y)))] for x in ca for y in cb]
        return min(vals) if linkage_mode == "complete" else sum(vals) / len(vals)

    while len(clusters) > 1:
        best, bi, bj = -math.inf, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if agg(gain, clusters[i], clusters[j]) <= 0:
                    continue
                lk = agg(nmiv, clusters[i], clusters[j])
                if lk > best:
                    best, bi, bj = lk, i, j
        if bi < 0 or best < min_nmi:  # nothing left compresses enough
            break
        clusters[bi] = clusters[bi] + clusters[bj]
        clusters.pop(bj)
    return sorted(clusters, key=lambda c: (-len(c), c[0]))


# ── theme encoder: cluster → one outcome-aligned summary variable ──

class ThemeEncoder:
    """Per cluster, reduce members to a single coarse theme variable.

    The theme is the cluster's mean *outcome-rank tendency*: each value is
    scored by the mean outcome severity it co-occurs with (TRAIN ONLY — this
    is target encoding, validated leakage-free by the outer k-fold), members
    are averaged, and the result is quantised into `n_levels` ordinal bins.
    Singletons pass through unchanged (already independent — keep full info).
    """

    def __init__(self, clusters: list[list[str]], n_levels: int = 4):
        self.clusters = clusters
        self.n_levels = n_levels
        self.value_score: dict[str, dict[str, float]] = {}
        self.global_score: float = 0.0
        self.edges: dict[int, list[float]] = {}
        self.theme_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ThemeEncoder":
        y = df["outcome_bucket"].map(RANK)
        self.global_score = float(y.mean())
        for f in {f for c in self.clusters for f in c}:
            self.value_score[f] = y.groupby(df[f]).mean().to_dict()
        self.theme_cols = []
        for ci, cl in enumerate(self.clusters):
            if len(cl) == 1:
                self.theme_cols.append(cl[0])  # singleton: raw feature
                continue
            name = f"theme_{ci}:" + "+".join(cl)
            self.theme_cols.append(name)
            raw = self._raw_theme(df, cl)
            qs = np.quantile(raw, [k / self.n_levels for k in range(1, self.n_levels)])
            self.edges[ci] = list(qs)
        return self

    def _raw_theme(self, df: pd.DataFrame, cl: list[str]) -> np.ndarray:
        cols = [df[f].map(lambda v, ff=f: self.value_score[ff].get(v, self.global_score)) for f in cl]
        return np.mean(np.vstack([c.to_numpy(dtype=float) for c in cols]), axis=0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for ci, cl in enumerate(self.clusters):
            if len(cl) == 1:
                out[cl[0]] = df[cl[0]].to_numpy()
                continue
            raw = self._raw_theme(df, cl)
            lvl = np.digitize(raw, self.edges[ci])
            out[self.theme_cols[ci]] = np.array([f"L{int(x)}" for x in lvl])
        return pd.DataFrame(out, index=df.index)


# ── categorical Naive Bayes (generic over any column set) ───────

class NB:
    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NB":
        self.cols = list(X.columns)
        n = len(y)
        self.prior = {c: math.log((int((y == c).sum()) + self.alpha) / (n + self.alpha * len(CLASSES)))
                      for c in CLASSES}
        self.cond, self.default = {}, {}
        for f in self.cols:
            vals = X[f].unique().tolist()
            vf = len(vals) + 1
            self.cond[f], self.default[f] = {}, {}
            for c in CLASSES:
                m = y == c
                nc = int(m.sum())
                vc = X.loc[m, f].value_counts().to_dict()
                self.cond[f][c] = {v: math.log((vc.get(v, 0) + self.alpha) / (nc + self.alpha * vf)) for v in vals}
                self.default[f][c] = math.log(self.alpha / (nc + self.alpha * vf))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        P = np.zeros((len(X), len(CLASSES)))
        cols = self.cols
        for r in range(len(X)):
            row = X.iloc[r]
            lp = [self.prior[c] + sum(self.cond[f][c].get(row[f], self.default[f][c]) for f in cols)
                  for c in CLASSES]
            m = max(lp)
            ex = [math.exp(v - m) for v in lp]
            z = sum(ex)
            P[r] = [e / z for e in ex]
        return P


class ClusteredNB:
    """Cluster → theme-reduce → NB over themes, with an optional tempered
    within-cluster residual (`beta`) that lets members refine the theme vote.

    beta=0 → pure theme NB (one decorrelated vote per cluster).
    beta>0 → add β·log[P(f|theme,outcome)/P(f|theme)] per multi-cluster member.
    """

    def __init__(self, n_levels: int = 4, beta: float = 0.0, clusters: list[list[str]] | None = None):
        self.n_levels, self.beta, self.clusters = n_levels, beta, clusters

    def fit(self, df: pd.DataFrame, features: list[str]) -> "ClusteredNB":
        self.features = features
        self.clusters_ = self.clusters or cluster_features(df, features)
        self.enc = ThemeEncoder(self.clusters_, self.n_levels).fit(df)
        T = self.enc.transform(df)
        y = df["outcome_bucket"]
        self.nb = NB().fit(T, y)
        # residual tables: P(f | theme, outcome) and P(f | theme)
        self.resid = {}
        if self.beta > 0:
            for ci, cl in enumerate(self.clusters_):
                if len(cl) == 1:
                    continue
                theme = T[self.enc.theme_cols[ci]].to_numpy()
                for f in cl:
                    self.resid[(ci, f)] = self._resid_tables(df[f].to_numpy(), theme, y.to_numpy())
        return self

    @staticmethod
    def _resid_tables(fv, theme, y):
        a = ALPHA
        vals = sorted(set(fv))
        vf = len(vals) + 1
        joint, marg, n_t, n_ty = {}, {}, {}, {}
        for i in range(len(fv)):
            t, c, v = theme[i], y[i], fv[i]
            n_t[t] = n_t.get(t, 0) + 1
            n_ty[(t, c)] = n_ty.get((t, c), 0) + 1
            marg[(t, v)] = marg.get((t, v), 0) + 1
            joint[(t, c, v)] = joint.get((t, c, v), 0) + 1
        return {"vals": vals, "vf": vf, "n_t": n_t, "n_ty": n_ty, "marg": marg, "joint": joint}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        T = self.enc.transform(df)
        P = self.nb.predict_proba(T)
        if self.beta <= 0:
            return P
        logP = np.log(np.clip(P, 1e-12, 1))
        for ci, cl in enumerate(self.clusters_):
            if len(cl) == 1:
                continue
            theme = T[self.enc.theme_cols[ci]].to_numpy()
            for f in cl:
                tab = self.resid[(ci, f)]
                fv = df[f].to_numpy()
                for r in range(len(df)):
                    t, v = theme[r], fv[r]
                    for k, c in enumerate(CLASSES):
                        num = (tab["joint"].get((t, c, v), 0) + ALPHA) / (tab["n_ty"].get((t, c), 0) + ALPHA * tab["vf"])
                        den = (tab["marg"].get((t, v), 0) + ALPHA) / (tab["n_t"].get(t, 0) + ALPHA * tab["vf"])
                        logP[r, k] += self.beta * math.log(num / den)
        # renormalise
        m = logP.max(1, keepdims=True)
        ex = np.exp(logP - m)
        return ex / ex.sum(1, keepdims=True)


# ── metrics (shared) ────────────────────────────────────────────

def onehot(df: pd.DataFrame) -> np.ndarray:
    Y = np.zeros((len(df), len(CLASSES)))
    for i, c in enumerate(df["outcome_bucket"].tolist()):
        Y[i, RANK[c]] = 1
    return Y


def metrics(P: np.ndarray, Y: np.ndarray) -> dict:
    eps = 1e-12
    ti, pi = Y.argmax(1), P.argmax(1)
    conf = P.max(1)
    correct = (pi == ti).astype(float)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / len(P)) * abs(correct[m].mean() - conf[m].mean())
    up = [RANK[c] for c in UPSIDE]
    return {
        "logloss": float(-np.mean(np.log(np.clip(P[np.arange(len(P)), ti], eps, 1)))),
        "ece": float(ece),
        "acc": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "overconf": float(conf.mean() - correct.mean()),
        "optimism_upside": float(P[:, up].sum(1).mean() - Y[:, up].sum(1).mean()),
    }


def crossval_proba(df, features, fold, build) -> np.ndarray:
    """build(train_df) -> fitted model with .predict_proba(test_df)."""
    P = np.zeros((len(df), len(CLASSES)))
    for k in sorted(set(fold)):
        tr, te = df[fold != k], df[fold == k]
        model = build(tr)
        P[np.where(fold == k)[0]] = model.predict_proba(te)
    return P
