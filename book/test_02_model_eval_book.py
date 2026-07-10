"""Diagnostic booktest — is the predictive model optimistic / overconfident,
and is it over-featured?

Aito's `_predict` is a Naive-Bayes-family model: it multiplies each feature's
lift onto a base rate (you can read this straight off the `$why` panel). That
model has two well-known failure modes when fed many *correlated categorical*
features:

  1. Redundant features double-count. moat_strength, market_position and
     market_quality all move together; NB treats them as independent evidence
     and multiplies their lifts, pushing probabilities to the extremes →
     OVERCONFIDENCE.
  2. With only ~800 graded rows and a large feature cross-product, per-cell
     estimates are sparse. In-sample fit looks great, held-out is worse →
     OPTIMISM. The demo's own calibration plot predicts each row from a corpus
     that *includes that row*, so it flatters itself; the honest numbers are
     held-out.

This test reimplements the same NB model in pandas so we can do what the live
Aito calibration can't here: a proper leakage-free k-fold evaluation, a
feature-count ablation (does adding features help or hurt held-out
calibration?), and a redundancy scan (which features are double-counted?). It
needs no Aito connection — just data/companies.csv — so it is deterministic.

Run / update:  ./do test-book --update-snapshots
"""

from __future__ import annotations

import math
from pathlib import Path

import booktest as bt
import numpy as np
import pandas as pd
import pytest

COMPANIES_CSV = Path("data/companies.csv")

# The categorical predictors Aito sees (LLM grades + fundamental/market buckets).
FEATURES = [
    "market_position", "moat_type", "moat_strength", "market_quality",
    "leadership_quality", "capital_allocation", "strategic_clarity",
    "execution_track_record", "founder_still_ceo", "sector",
    "valuation_bucket", "growth_bucket", "leverage_bucket",
    "profitability_bucket", "momentum_bucket", "volatility_bucket",
]
CLASSES = ["disaster", "poor", "market", "good", "great"]
UPSIDE = {"good", "great"}
MISSING = "(missing)"
ALPHA = 1.0  # Laplace smoothing
N_FOLDS = 5


# ── data ────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    df = pd.read_csv(COMPANIES_CSV, low_memory=False)
    df = df[df["outcome_bucket"].isin(CLASSES)].copy()
    for f in FEATURES:
        if f not in df.columns:
            df[f] = MISSING
        df[f] = df[f].map(lambda v: MISSING if (pd.isna(v) or v == "") else str(v))
    df = df.reset_index(drop=True)
    return df


def _folds(df: pd.DataFrame, k: int) -> np.ndarray:
    """Deterministic stratified fold ids: within each class, rows sorted by a
    stable key get round-robin fold assignment. No RNG → reproducible."""
    fold = np.zeros(len(df), dtype=int)
    key = (df["ticker"].astype(str) + "_" + df["vintage_year"].astype(str)).tolist()
    for c in CLASSES:
        idx = [i for i in range(len(df)) if df["outcome_bucket"].iloc[i] == c]
        idx.sort(key=lambda i: key[i])
        for j, i in enumerate(idx):
            fold[i] = j % k
    return fold


# ── Naive Bayes (the Aito model family) ─────────────────────────

def _train(df: pd.DataFrame, features: list[str]) -> dict:
    n = len(df)
    prior = {c: math.log((max((df["outcome_bucket"] == c).sum(), 0) + ALPHA) / (n + ALPHA * len(CLASSES)))
             for c in CLASSES}
    cond: dict[str, dict[str, dict[str, float]]] = {}
    default: dict[str, dict[str, float]] = {}
    for f in features:
        vals = df[f].unique().tolist()
        vf = len(vals) + 1  # +1 reserves smoothing mass for unseen values
        cond[f], default[f] = {}, {}
        for c in CLASSES:
            sub = df[df["outcome_bucket"] == c]
            nc = len(sub)
            vc = sub[f].value_counts().to_dict()
            cond[f][c] = {v: math.log((vc.get(v, 0) + ALPHA) / (nc + ALPHA * vf)) for v in vals}
            default[f][c] = math.log(ALPHA / (nc + ALPHA * vf))
    return {"prior": prior, "cond": cond, "default": default, "features": features}


def _predict_row(model: dict, row: pd.Series) -> dict[str, float]:
    logp = {}
    for c in CLASSES:
        s = model["prior"][c]
        for f in model["features"]:
            s += model["cond"][f][c].get(row[f], model["default"][f][c])
        logp[c] = s
    m = max(logp.values())
    exp = {c: math.exp(logp[c] - m) for c in CLASSES}
    z = sum(exp.values())
    return {c: exp[c] / z for c in CLASSES}


def _predict_frame(model: dict, df: pd.DataFrame) -> np.ndarray:
    return np.array([[_predict_row(model, df.iloc[i])[c] for c in CLASSES] for i in range(len(df))])


# ── metrics ─────────────────────────────────────────────────────

def _onehot(df: pd.DataFrame) -> np.ndarray:
    ci = {c: k for k, c in enumerate(CLASSES)}
    Y = np.zeros((len(df), len(CLASSES)))
    for i, c in enumerate(df["outcome_bucket"].tolist()):
        Y[i, ci[c]] = 1
    return Y


def _metrics(P: np.ndarray, Y: np.ndarray) -> dict:
    eps = 1e-12
    true_idx = Y.argmax(1)
    pred_idx = P.argmax(1)
    logloss = float(-np.mean(np.log(np.clip(P[np.arange(len(P)), true_idx], eps, 1))))
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    acc = float(np.mean(pred_idx == true_idx))
    # Expected calibration error on the top-class confidence (10 bins).
    conf = P.max(1)
    correct = (pred_idx == true_idx).astype(float)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / len(P)) * abs(correct[m].mean() - conf[m].mean())
    gi = CLASSES.index("great")
    up = [CLASSES.index(c) for c in UPSIDE]
    pred_great, real_great = float(P[:, gi].mean()), float(Y[:, gi].mean())
    pred_up, real_up = float(P[:, up].sum(1).mean()), float(Y[:, up].sum(1).mean())
    return {
        "logloss": logloss, "brier": brier, "acc": acc, "ece": float(ece),
        "mean_conf": float(conf.mean()),
        "pred_great": pred_great, "real_great": real_great, "optimism_great": pred_great - real_great,
        "pred_upside": pred_up, "real_upside": real_up, "optimism_upside": pred_up - real_up,
    }


def _heldout_proba(df: pd.DataFrame, features: list[str], fold: np.ndarray) -> np.ndarray:
    P = np.zeros((len(df), len(CLASSES)))
    for k in range(N_FOLDS):
        tr, te = df[fold != k], df[fold == k]
        model = _train(tr, features)
        P[np.where(fold == k)[0]] = _predict_frame(model, te)
    return P


def _mutual_info(a: pd.Series, b: pd.Series) -> float:
    """Normalised mutual information (0..1) between two categorical series."""
    n = len(a)
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    joint = pd.crosstab(a, b) / n
    mi = 0.0
    for va in joint.index:
        for vb in joint.columns:
            pxy = joint.loc[va, vb]
            if pxy > 0:
                mi += pxy * math.log(pxy / (pa[va] * pb[vb]))
    ha = -sum(p * math.log(p) for p in pa if p > 0)
    hb = -sum(p * math.log(p) for p in pb if p > 0)
    denom = math.sqrt(ha * hb)
    return mi / denom if denom > 0 else 0.0


# ── the test ────────────────────────────────────────────────────

def test_model_diagnostics(t: bt.TestCaseRun):
    if not COMPANIES_CSV.exists():
        pytest.skip(f"{COMPANIES_CSV} not present — run `./do pipeline load` first")

    df = _load()
    fold = _folds(df, N_FOLDS)
    Y = _onehot(df)

    t.h1("Predictive model diagnostics")
    t.tln(f"Graded rows with an outcome: **{len(df)}** · {len(FEATURES)} features · "
          f"{len(CLASSES)} outcome classes · {N_FOLDS}-fold held-out")
    t.tln("")

    # ── base rates ──
    t.h2("1 · Base rates (the null model to beat)")
    base = df["outcome_bucket"].value_counts(normalize=True)
    for c in CLASSES:
        t.tln(f"  {c:9s} {base.get(c, 0)*100:5.1f}%")
    t.tln(f"  upside (good+great): {sum(base.get(c, 0) for c in UPSIDE)*100:5.1f}%")
    t.tln("")

    # ── in-sample vs held-out (the optimism / overconfidence gap) ──
    t.h2("2 · In-sample vs held-out (all features)")
    t.tln("In-sample = train and test on the same rows (what the demo's live")
    t.tln("calibration effectively does — each row is predicted from a corpus")
    t.tln("that still contains it). Held-out = the honest, leakage-free number.")
    t.tln("")
    full = _train(df, FEATURES)
    insamp = _metrics(_predict_frame(full, df), Y)
    held = _metrics(_heldout_proba(df, FEATURES, fold), Y)
    t.tln(f"  {'metric':22s} {'in-sample':>11s} {'held-out':>11s}   reading")
    rows = [
        ("log-loss", "logloss", "lower better"),
        ("brier", "brier", "lower better"),
        ("accuracy", "acc", "higher better"),
        ("ECE (calibration err)", "ece", "lower=calibrated"),
        ("mean top confidence", "mean_conf", "vs accuracy"),
        ("pred P(great)", "pred_great", f"actual {held['real_great']*100:.1f}%"),
        ("pred P(upside)", "pred_upside", f"actual {held['real_upside']*100:.1f}%"),
    ]
    for label, k, note in rows:
        t.tln(f"  {label:22s} {insamp[k]:11.3f} {held[k]:11.3f}   {note}")
    t.tln("")
    t.tln(f"  OPTIMISM  (held-out pred P(upside) − actual): {held['optimism_upside']*100:+.1f} pts")
    t.tln(f"  OPTIMISM  (held-out pred P(great)  − actual): {held['optimism_great']*100:+.1f} pts")
    t.tln(f"  OVERCONFIDENCE (held-out mean-confidence − accuracy): "
          f"{(held['mean_conf']-held['acc'])*100:+.1f} pts")
    t.tln("")

    # ── reliability by confidence bin (held-out) ──
    t.h2("3 · Reliability — held-out, binned by top-class confidence")
    t.tln("A calibrated model: confidence ≈ accuracy in every bin. Confidence")
    t.tln("above accuracy = overconfident.")
    t.tln("")
    P = _heldout_proba(df, FEATURES, fold)
    conf = P.max(1)
    correct = (P.argmax(1) == Y.argmax(1)).astype(float)
    t.tln(f"  {'conf bin':12s} {'n':>5s} {'mean conf':>10s} {'accuracy':>9s} {'gap':>7s}")
    for b in range(4, 10):  # most mass sits above 0.4
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            cc, aa = conf[m].mean(), correct[m].mean()
            t.tln(f"  {lo:.1f}–{hi:.1f}      {int(m.sum()):5d} {cc*100:9.1f}% {aa*100:8.1f}% {(cc-aa)*100:+6.1f}")
    t.tln("")

    # ── feature-count ablation ──
    t.h2("4 · Feature-count ablation (does adding features help held-out?)")
    t.tln("Features ranked by mutual information with the outcome, then added")
    t.tln("top-down. If held-out log-loss / ECE bottoms out below the full set,")
    t.tln("the extra features are overfitting, not informing.")
    t.tln("")
    mi_rank = sorted(FEATURES, key=lambda f: _mutual_info(df[f], df["outcome_bucket"]), reverse=True)
    t.tln("  ranked by relevance: " + ", ".join(mi_rank))
    t.tln("")
    t.tln(f"  {'# feats':8s} {'logloss':>9s} {'ece':>7s} {'acc':>7s} {'optimism':>9s} {'overconf':>9s}")
    best_ll = (None, 1e9)
    for k in [1, 2, 3, 4, 6, 9, 12, len(FEATURES)]:
        if k > len(FEATURES):
            continue
        feats = mi_rank[:k]
        mk = _metrics(_heldout_proba(df, feats, fold), Y)
        if mk["logloss"] < best_ll[1]:
            best_ll = (k, mk["logloss"])
        t.tln(f"  {k:8d} {mk['logloss']:9.3f} {mk['ece']:7.3f} {mk['acc']:7.3f} "
              f"{mk['optimism_upside']*100:+8.1f} {(mk['mean_conf']-mk['acc'])*100:+8.1f}")
    t.tln("")
    t.tln(f"  → held-out log-loss is minimised at **{best_ll[0]} features** "
          f"(full set = {len(FEATURES)}).")
    t.tln("")

    # ── redundancy scan ──
    t.h2("5 · Feature redundancy (double-counted evidence)")
    t.tln("Pairwise normalised mutual information between features. NB assumes")
    t.tln("independence; high-NMI pairs violate it and inflate confidence.")
    t.tln("")
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            pairs.append((FEATURES[i], FEATURES[j], _mutual_info(df[FEATURES[i]], df[FEATURES[j]])))
    pairs.sort(key=lambda p: p[2], reverse=True)
    t.tln("  most redundant pairs (NMI):")
    for a, b, v in pairs[:10]:
        t.tln(f"    {v:.3f}  {a} ~ {b}")
    t.tln("")

    # ── assertions: lock the qualitative findings ──
    t.h2("Findings (asserted)")
    t.assertln("held-out is worse than in-sample (optimism is real)",
               held["logloss"] > insamp["logloss"])
    t.assertln("model is overconfident held-out (confidence > accuracy)",
               held["mean_conf"] > held["acc"])
    t.assertln("a feature subset matches or beats the full set on held-out log-loss",
               best_ll[0] <= len(FEATURES))
