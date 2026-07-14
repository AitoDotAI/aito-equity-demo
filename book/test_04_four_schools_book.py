"""The Four Schools — do Graham, Buffett, Fisher, or the data-driven composite
win as *funds*?

The blog "Value, quality, or growth: who was right?" answers the question with a
single `relate` query (which factors lift a great outcome). This book asks the
harder, money-weighted version: build a **fund per school** with the *same
held-out methodology as the shipped "Beat the Market?" view* — rank every
company by expected outcome, buy the top-N, measure realized CAGR — and change
**only which features the model is allowed to see**:

  - Value (Graham)   : valuation, P/E, leverage, profitability      — price & balance sheet
  - Quality (Buffett): moat, market position, leadership, capital-  — the business
                       allocation, strategic clarity, execution
  - Growth (Fisher)  : sector, industry, market quality             — the megatrend (sector-led)
  - Composite        : all 16 features                              — let the data weight the lenses

Why a book and not just the live demo: the live "Beat the Market?" numbers come
from Aito's `ai:high` model over the network — non-deterministic and creds-gated.
Here we reimplement the same Naive-Bayes model family in pandas so the whole
comparison is **deterministic and leakage-free** (companies grouped by ticker via
the shipped `eval_fold`, so no company is ever trained on its own outcome). The
composite fund lands near the live-Aito held-out top-20 (≈20.6%/yr), which is the
cross-check that this mirror is faithful.

What to read for: not "which school wins the top-20 sprint" (that's a
concentrated, sector-timing-inflated bet), but **how each school decays as the
fund grows from 20 → 100 names**. The robust one is the story.

Run / update:  ./do test-book -u   (or -a to accept a changed snapshot)
"""
from __future__ import annotations

import math
from pathlib import Path

import booktest as bt
import numpy as np
import pandas as pd
import pytest

COMPANIES_CSV = Path("data/companies.csv")

CLASSES = ["disaster", "poor", "market", "good", "great"]
RANK = {c: i for i, c in enumerate(CLASSES)}  # ordinal rank for the expected-outcome score
MISSING = "(missing)"
ALPHA = 1.0        # Laplace smoothing
N_FOLDS = 5
PE_BINS = 10       # Aito auto-bins Decimals; the NB mirror needs pe_ratio as categories

# Each school = the feature family that philosophy owns. The ONLY thing that
# changes between funds. Composite = the full model (same 16 as the live view).
SCHOOLS: dict[str, list[str]] = {
    "Value (Graham)": ["valuation_bucket", "pe_ratio", "leverage_bucket", "profitability_bucket"],
    "Quality (Buffett)": ["market_position", "moat_type", "moat_strength", "leadership_quality",
                          "capital_allocation", "strategic_clarity", "execution_track_record"],
    "Growth (Fisher)": ["sector", "industry", "market_quality"],
    "Composite (data-driven)": ["market_position", "moat_type", "moat_strength", "market_quality",
                                "leadership_quality", "capital_allocation", "strategic_clarity",
                                "execution_track_record", "founder_still_ceo", "sector",
                                "valuation_bucket", "growth_bucket", "leverage_bucket",
                                "profitability_bucket", "momentum_bucket", "volatility_bucket"],
}
ALL_FEATURES = sorted({f for fs in SCHOOLS.values() for f in fs})
FUND_SIZES = [20, 50, 100]


# ── data ────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    df = pd.read_csv(COMPANIES_CSV, low_memory=False)
    df = df[df["outcome_bucket"].isin(CLASSES) & df["total_return_pct_local"].notna()].reset_index(drop=True)
    df["eval_fold"] = df["eval_fold"].astype(int)  # ticker-grouped folds, same as the shipped view
    df["cagr"] = [((1 + r / 100) ** (1 / max(w, 0.5)) - 1) * 100
                  for r, w in zip(df["total_return_pct_local"], df["window_years"].fillna(10))]
    # P/E → deterministic deciles so it's a usable categorical feature (its bucket
    # sibling valuation_bucket is already categorical).
    if "pe_ratio" in df.columns:
        pe = pd.to_numeric(df["pe_ratio"], errors="coerce")
        df["pe_ratio"] = pd.qcut(pe, PE_BINS, labels=[f"pe_d{i}" for i in range(PE_BINS)],
                                 duplicates="drop").astype("object")
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = MISSING
        df[f] = df[f].map(lambda v: MISSING if (pd.isna(v) or v == "") else str(v))
    return df


# ── Naive Bayes (the Aito model family), deterministic ──────────

def _train(df: pd.DataFrame, features: list[str]) -> dict:
    n = len(df)
    prior = {c: math.log((max(int((df["outcome_bucket"] == c).sum()), 0) + ALPHA) / (n + ALPHA * len(CLASSES)))
             for c in CLASSES}
    cond: dict[str, dict[str, dict[str, float]]] = {}
    default: dict[str, dict[str, float]] = {}
    for f in features:
        vals = df[f].unique().tolist()
        vf = len(vals) + 1  # reserve smoothing mass for unseen values
        cond[f], default[f] = {}, {}
        for c in CLASSES:
            sub = df[df["outcome_bucket"] == c]
            nc = len(sub)
            vc = sub[f].value_counts().to_dict()
            cond[f][c] = {v: math.log((vc.get(v, 0) + ALPHA) / (nc + ALPHA * vf)) for v in vals}
            default[f][c] = math.log(ALPHA / (nc + ALPHA * vf))
    return {"prior": prior, "cond": cond, "default": default, "features": features}


def _predict_proba(model: dict, row: pd.Series) -> list[float]:
    logp = {}
    for c in CLASSES:
        s = model["prior"][c]
        for f in model["features"]:
            s += model["cond"][f][c].get(row[f], model["default"][f][c])
        logp[c] = s
    m = max(logp.values())
    exp = {c: math.exp(logp[c] - m) for c in CLASSES}
    z = sum(exp.values())
    return [exp[c] / z for c in CLASSES]


def _heldout_scores(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Expected-outcome score (Σ p·rank) for every row, predicted from the 4 folds
    that exclude it. Identical ranking metric to the shipped Beat-the-Market view."""
    rankvec = np.array([RANK[c] for c in CLASSES])
    score = np.zeros(len(df))
    for k in range(N_FOLDS):
        tr = df[df["eval_fold"] != k]
        model = _train(tr, features)
        te_idx = np.where(df["eval_fold"].values == k)[0]
        for i in te_idx:
            score[i] = float(np.dot(_predict_proba(model, df.iloc[i]), rankvec))
    return score


# ── alternative scorings (does the discrete-bin / rank realm distort it?) ──
#
# Three ways to turn the same held-out signal into a ranking:
#   rank   : Σ p(bucket)·rank(bucket)        — ordinal index, evenly spaced (the shipped view)
#   expcagr: Σ p(bucket)·E[cagr | bucket]    — the expectation-value realm, in the fund's own units
#   reg    : additive regression on log-return, NO bins at all — the numeric realm
# The bin→value map and the regression coefficients are estimated on TRAIN folds only.

REG_SHRINK = 10.0  # pull small feature-cells toward the global mean (guards sparse values)


def _heldout_all_scores(df: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    rankvec = np.array([RANK[c] for c in CLASSES])
    s_rank = np.zeros(len(df)); s_expcagr = np.zeros(len(df)); s_reg = np.zeros(len(df))
    logret = np.log1p(np.maximum(df["total_return_pct_local"].values, -99.0) / 100)  # additive numeric target
    for k in range(N_FOLDS):
        tr = df[df["eval_fold"] != k]
        model = _train(tr, features)
        cagrvec = np.array([float(tr.loc[tr["outcome_bucket"] == c, "cagr"].mean()) for c in CLASSES])
        # additive log-return regression: E[y|x] = global + Σ_f shrunk(mean_f(value) − global)
        g = float(logret[tr.index].mean())
        eff: dict[str, dict[str, float]] = {}
        for f in features:
            sub = pd.DataFrame({f: tr[f].values, "y": logret[tr.index]})
            gp = sub.groupby(f)["y"]; cnt, mean = gp.count(), gp.mean()
            eff[f] = {v: (cnt[v] / (cnt[v] + REG_SHRINK)) * (mean[v] - g) for v in mean.index if pd.notna(mean[v])}
        for i in np.where(df["eval_fold"].values == k)[0]:
            row = df.iloc[i]
            p = np.array(_predict_proba(model, row))
            s_rank[i] = float(np.dot(p, rankvec))
            s_expcagr[i] = float(np.dot(p, cagrvec))
            s_reg[i] = g + sum(eff[f].get(row[f], 0.0) for f in features)
    return {"rank": s_rank, "expcagr": s_expcagr, "reg": s_reg}


# ── the test ────────────────────────────────────────────────────

def test_four_schools(t: bt.TestCaseRun):
    if not COMPANIES_CSV.exists():
        pytest.skip(f"{COMPANIES_CSV} not present — run `./do pipeline load` first")

    df = _load()
    mkt_cagr = float(df["cagr"].mean())
    mkt_total = float(df["total_return_pct_local"].mean())
    vintage_mkt = df.groupby("vintage_year")["total_return_pct_local"].mean().to_dict()

    t.h1("The Four Schools — one fund per philosophy, held-out")
    t.tln(f"Held-out universe: **{len(df)}** graded observations across "
          f"{df['vintage_year'].nunique()} vintages · {N_FOLDS}-fold, grouped by ticker.")
    t.tln(f"Equal-weight market: **{mkt_cagr:.1f}%/yr** CAGR (mean total {mkt_total:.0f}%). "
          f"Every fund is measured against this.")
    t.tln("Only the visible feature set changes between funds; model, folds, and")
    t.tln("ranking (buy the top-N by expected outcome) are identical throughout.")
    t.tln("")

    # ── compute every school ──
    results = {}
    for name, feats in SCHOOLS.items():
        score = _heldout_scores(df, feats)
        ranked = df.assign(score=score).sort_values("score", ascending=False).reset_index(drop=True)
        row = {"features": feats, "funds": {}}
        for nsz in FUND_SIZES:
            top = ranked.head(nsz)
            beat = float((top["total_return_pct_local"] > top["vintage_year"].map(vintage_mkt)).mean())
            row["funds"][nsz] = {"cagr": float(top["cagr"].mean()),
                                 "total": float(top["total_return_pct_local"].mean()), "beat": beat}
        row["decay"] = row["funds"][100]["cagr"] - row["funds"][20]["cagr"]  # top-100 minus top-20
        row["top_names"] = ranked.head(8)["ticker"].tolist()
        results[name] = row

    # ── the comparison table (the blog chart, as text) ──
    t.h2("1 · The funds — realized CAGR by size, and % that beat the market")
    t.tln("% beat = share of the fund's holdings whose total return exceeded the")
    t.tln("equal-weight market of the same vintage. Decay = top-100 CAGR − top-20")
    t.tln("CAGR: how much the edge survives when you stop cherry-picking.")
    t.tln("")
    t.tln(f"  {'School':26s} {'top-20':>8s} {'top-50':>8s} {'top-100':>8s} {'decay':>7s} {'beat@20':>8s}")
    t.tln(f"  {'Market (equal-weight)':26s} {mkt_cagr:7.1f}% {'—':>8s} {'—':>8s} {'—':>7s} {'—':>8s}")
    for name in SCHOOLS:
        r = results[name]; f = r["funds"]
        t.tln(f"  {name:26s} {f[20]['cagr']:7.1f}% {f[50]['cagr']:7.1f}% {f[100]['cagr']:7.1f}% "
              f"{r['decay']:+6.1f} {f[20]['beat']*100:7.0f}%")
    t.tln("")

    # ── who wins on which axis ──
    top20_winner = max(SCHOOLS, key=lambda s: results[s]["funds"][20]["cagr"])
    top100_winner = max(SCHOOLS, key=lambda s: results[s]["funds"][100]["cagr"])
    most_robust = min(SCHOOLS, key=lambda s: abs(results[s]["decay"]))
    t.h2("2 · Reading it")
    t.tln(f"  Top-20 sprint winner : {top20_winner} "
          f"({results[top20_winner]['funds'][20]['cagr']:.1f}%/yr)")
    t.tln(f"  Top-100 winner       : {top100_winner} "
          f"({results[top100_winner]['funds'][100]['cagr']:.1f}%/yr)")
    t.tln(f"  Most robust (smallest decay): {most_robust} "
          f"({results[most_robust]['decay']:+.1f} pts from 20→100)")
    t.tln("")
    t.tln("  The top-20 number flatters concentrated single-factor bets — a handful")
    t.tln("  of names in the sector the era happened to reward. The decay column is")
    t.tln("  the honest one: it shows which edge is breadth (survives dilution) and")
    t.tln("  which is a lucky top slice. That is the argument for the data-driven")
    t.tln("  composite — not that it wins the sprint, but that it barely decays.")
    t.tln("")

    # ── what each school actually bought ──
    t.h2("3 · What each school bought (top 8 by rank)")
    for name in SCHOOLS:
        t.tln(f"  {name:26s} {', '.join(results[name]['top_names'])}")
    t.tln("")
    t.tln("  Growth's picks are the semiconductor/IT megatrend showing up as a")
    t.tln("  portfolio — the win the blog flags as visible only in hindsight.")
    t.tln("")

    # ── does the discrete-bin / rank realm distort the ranking? ──
    t.h2("4 · Scoring realm — rank vs expectation-value vs numeric (no bins)")
    t.tln("The default score is Σ p(bucket)·rank(bucket): the bins are treated as")
    t.tln("evenly spaced. In *total-return* space they are wildly convex (great ≫")
    t.tln("good), so that ordinal score under-weights the tail. Two fixes: score by")
    t.tln("expected CAGR (Σ p·E[cagr|bucket]) — the expectation-value realm in the")
    t.tln("fund's own units — or drop bins entirely and regress numeric log-return.")
    t.tln("All three ranked on the same held-out signal; funds still measured in CAGR.")
    t.tln("")
    alt = {name: _heldout_all_scores(df, SCHOOLS[name]) for name in SCHOOLS}

    def top(scores: np.ndarray, n: int) -> float:
        return float(df.assign(s=scores).sort_values("s", ascending=False).head(n)["cagr"].mean())

    for realm, key, note in [("rank  Σp·rank", "rank", "ordinal, evenly spaced (shipped)"),
                             ("expcagr Σp·E[cagr]", "expcagr", "expectation-value, in CAGR units"),
                             ("numeric log-ret reg", "reg", "no bins — regress the number")]:
        t.tln(f"  {realm}  — {note}")
        t.tln(f"    {'School':26s} {'top-20':>8s} {'top-50':>8s} {'top-100':>8s}")
        for name in SCHOOLS:
            s = alt[name][key]
            t.tln(f"    {name:26s} {top(s,20):7.1f}% {top(s,50):7.1f}% {top(s,100):7.1f}%")
        t.tln("")
    t.tln("  Reading it:")
    t.tln("  • The composite barely moves across realms (~19% top-20 everywhere):")
    t.tln("    CAGR is already a log-compression of total return, so in the fund's")
    t.tln("    own units the bins are ≈evenly spaced and the rank score is a fine")
    t.tln("    proxy. Discretization is not why the composite fails to dominate.")
    t.tln("  • Value is the one that collapses (top-20 24.8% → ~15-17%): its lead")
    t.tln("    over the composite WAS a rank-score artifact that rewarded a")
    t.tln("    concentrated, overconfident cheap-stock sort. In the expectation")
    t.tln("    realm the composite edges every school except Growth.")
    t.tln("  • Growth wins under every realm — the tell that its edge is a genuine")
    t.tln("    (if hindsight-only) signal, not an artifact of how we score.")
    t.tln("")

    # ── invariants worth guarding against drift ──
    t.h2("5 · Sanity checks")
    all_beat = all(results[s]["funds"][20]["cagr"] > mkt_cagr for s in SCHOOLS)
    comp_robust = abs(results["Composite (data-driven)"]["decay"]) <= min(
        abs(results[s]["decay"]) for s in SCHOOLS if s != "Composite (data-driven)")
    t.tln(f"  every school's top-20 beat the market : {all_beat}")
    t.tln(f"  composite is the most robust (least decay): {comp_robust}")
    t.tln(f"  composite top-20 near live-Aito ~20.6%   : "
          f"{results['Composite (data-driven)']['funds'][20]['cagr']:.1f}%")

    assert all_beat, "a school's top-20 failed to beat the market — investigate the pipeline"
    assert results["Growth (Fisher)"]["funds"][20]["cagr"] > mkt_cagr


if __name__ == "__main__":  # allow `python -m book.test_04_...` style spot runs
    pytest.main([__file__, "-s"])
