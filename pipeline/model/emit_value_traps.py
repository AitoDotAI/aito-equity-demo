"""Emit site/data/value_traps.json — "Mr. Market, or a value trap?"

A mid-term value trader buys cheap stocks expecting the gap to close within
3 months – 2 years. Two kinds of company look cheap:

  - mispriced by a moody Mr. Market → re-rates (outperforms SPY) in 6–12 m
  - a value trap: cheap because the business is shrinking toward distress →
    keeps lagging, ends `poor` / `disaster` / bankrupt over the long horizon

The hypothesis is that growth, debt, moat, market position and the quality of
the market the company sells into separate the two. This view tests it on
the cheap cohort (valuation_bucket = cheap, i.e. low P/E at the vintage date).

Three layers:
  1. splits   — exact pandas cross-tabs: for each feature value inside the
                cheap cohort, mean 6/12-month excess return and the long-run
                trap rate. Reconcilable by hand.
  2. relate   — Aito `_relate` on a cheap-only table (the companies table's
                `where` keeps a global base rate, so the cohort gets its own
                table): which feature values lift "re-rated within 12 m" and
                which lift "long-run trap".
  3. quadrant — walk-forward (2017 ← 2014, 2020 ← 2014+2017, 2026 ← all;
                trained on the whole companies table) Aito `_predict` of
                P(outperformed in 6 m / 12 m) and P(long-run poor ∪ disaster)
                for every cheap company. Split at the cohort medians into four
                quadrants and score each on what actually happened.

    uv run python -m pipeline.model.emit_value_traps            # → site/data/value_traps.json
    uv run python -m pipeline.model.emit_value_traps --refresh  # re-query Aito (ignore cache)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from pipeline.aito.client import AitoClient, api_root, hit_value
from pipeline.aito.load import df_to_aito_rows, load_schema
from pipeline.aito.queries import parse_why
from pipeline.model import featureclust as fc

OUT = Path("site/data/value_traps.json")
CACHE = Path("data/value_trap_records.csv")  # held-out preds — re-emit JSON without re-querying Aito
TRAP = ("poor", "disaster")
INT = {"leadership_quality", "capital_allocation", "strategic_clarity", "execution_track_record"}
FEATS = [f for f in fc.DEFAULT_FEATURES if f != "valuation_bucket"]  # constant inside the cohort
SPLIT_FEATS = ["market_quality", "leverage_bucket", "growth_bucket", "moat_strength", "market_position",
               "profitability_bucket", "momentum_bucket", "capital_allocation"]
MIN_N = 8


def _wof(row, feats):
    w = {}
    for f in feats:
        v = row.get(f)
        if pd.isna(v) or v == "":
            continue
        w[f] = int(float(v)) if f in INT else (True if str(v) == "True" else (False if str(v) == "False" else str(v)))
    return w


def _r(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _cell(sub: pd.DataFrame) -> dict:
    lr = sub[sub.outcome_bucket.notna() & (sub.outcome_bucket != "")]
    return {
        "n": int(len(sub)),
        "n_6m": int(sub.fwd_6m_excess.notna().sum()),
        "n_12m": int(sub.fwd_12m_excess.notna().sum()),
        "n_longrun": int(len(lr)),
        "excess_6m": _r(sub.fwd_6m_excess.mean()),
        "excess_12m": _r(sub.fwd_12m_excess.mean()),
        "spy_6m": _r(sub.fwd_6m_excess_spy.mean()) if "fwd_6m_excess_spy" in sub else None,
        "spy_12m": _r(sub.fwd_12m_excess_spy.mean()) if "fwd_12m_excess_spy" in sub else None,
        "median_12m": _r(sub.fwd_12m_excess.median()),
        "outperf_12m": _r((sub.fwd_12m_bucket == "outperformed").sum() / max(sub.fwd_12m_bucket.notna().sum(), 1), 3),
        "trap_rate": _r(lr.outcome_bucket.isin(TRAP).mean(), 3) if len(lr) else None,
    }


def _heldout(df: pd.DataFrame, cheap: pd.DataFrame, url: str, h: dict) -> pd.DataFrame:
    targets = {"p6": ("fwd_6m_bucket", "outperformed"), "p12": ("fwd_12m_bucket", "outperformed"),
               "ptrap": ("outcome_bucket", None)}
    coltypes = load_schema()["schema"]["companies"]
    schema_cols = [c for c in df.columns if c in coltypes["columns"]]
    recs = []
    # Walk-forward by vintage. Ticker-grouped random folds would train on OTHER
    # companies' same-window returns — i.e. tell the model which sectors won the
    # next 12 months. Mid-term labels resolve within a year, so training only on
    # earlier vintages is strictly point-in-time for p6/p12. (The long-run trap
    # label still resolves in ~2026 for every vintage; see caveats.)
    vintages = sorted(int(v) for v in df.vintage_year.unique())
    for test_v in vintages[1:]:
        train, test = df[df.vintage_year < test_v], cheap[cheap.vintage_year == test_v]
        with AitoClient() as c:
            c.delete_table("vt_tmp"); c.put_schema({"schema": {"vt_tmp": coltypes}})
            c.upload_batch("vt_tmp", df_to_aito_rows(train[schema_cols]))
        with httpx.Client(timeout=60) as cl:
            for _, r in test.iterrows():
                w = _wof(r, fc.DEFAULT_FEATURES)
                if not w:
                    continue
                rec = {"ticker": r.ticker, "vintage": int(r.vintage_year), "company_name": r.company_name}
                for key, (col, val) in targets.items():
                    rr = cl.post(f"{url}/_predict", json={"from": "vt_tmp", "where": w, "predict": col,
                                 "select": ["$value", "$p", "$why"], "config": {"ai": "high"}}, headers=h)
                    if rr.status_code >= 400:
                        continue
                    hits = rr.json()["hits"]
                    d = {hit_value(x): x["$p"] for x in hits}
                    if val is None:  # trap = poor ∪ disaster
                        rec[key] = sum(d.get(c, 0) for c in TRAP)
                        why_of = max(TRAP, key=lambda c: d.get(c, 0))
                    else:
                        rec[key] = d.get(val, 0)
                        why_of = val
                    why = next((parse_why(x.get("$why")) for x in hits if hit_value(x) == why_of), None)
                    rec[f"{key}_why"] = json.dumps({"label": why_of, "prob": round(float(d.get(why_of, 0)), 4), "why": why})
                recs.append(rec)
        print(f"  test {test_v} ← train {sorted(train.vintage_year.unique().tolist())}: {len(test)} cheap")
    with AitoClient() as c:
        c.delete_table("vt_tmp")
    return pd.DataFrame(recs)


def _relate(cheap: pd.DataFrame, url: str, h: dict) -> dict:
    """Lifts inside the cheap cohort, on a cohort-only table."""
    coltypes = load_schema()["schema"]["companies"]["columns"]
    rows = []
    for _, r in cheap.iterrows():
        row = _wof(r, FEATS)
        row["rerated_12m"] = (r.fwd_12m_bucket == "outperformed") if pd.notna(r.fwd_12m_bucket) else None
        row["longrun_trap"] = (r.outcome_bucket in TRAP) if pd.notna(r.outcome_bucket) and r.outcome_bucket != "" else None
        rows.append({k: v for k, v in row.items() if v is not None})
    cols = {f: coltypes[f] for f in FEATS if f in coltypes}
    cols.update({"rerated_12m": {"type": "Boolean", "nullable": True}, "longrun_trap": {"type": "Boolean", "nullable": True}})
    with AitoClient() as c:
        c.delete_table("vt_cheap"); c.put_schema({"schema": {"vt_cheap": {"type": "table", "columns": cols}}})
        c.upload_batch("vt_cheap", rows)
    out = {}
    for target in ("rerated_12m", "longrun_trap"):
        rr = httpx.post(f"{url}/_relate", json={"from": "vt_cheap", "where": {"$on": [{target: True}, {target: {"$defined": True}}]},
                              "relate": FEATS, "limit": 40},
                        headers=h, timeout=90)
        hits = []
        if rr.status_code < 400:
            for hit in rr.json().get("hits", []):
                rel = hit.get("related", {})
                field = next((k for k in rel if not k.startswith("$")), None)
                if not field:
                    continue
                val = rel[field].get("$has") if isinstance(rel[field], dict) else rel[field]
                n = int(hit.get("fs", {}).get("f", 0))
                if n < MIN_N:
                    continue
                hits.append({"feature": field, "value": str(val), "lift": round(hit.get("lift") or 0, 2), "n": n})
        hits.sort(key=lambda x: -x["lift"])
        out[target] = {"up": hits[:6], "down": [x for x in hits[::-1] if x["lift"] < 1][:6]}
    with AitoClient() as c:
        c.delete_table("vt_cheap")
    return out


def main() -> None:
    load_dotenv(override=True)
    url = api_root(); h = {"x-api-key": os.environ["AITO_API_KEY"]}
    df = pd.read_csv("data/companies.csv", low_memory=False)
    df["eval_fold"] = df["eval_fold"].astype(str)
    has_outcome = df.outcome_bucket.notna() | df.fwd_6m_bucket.notna()
    gone = df[df.terminal_event.isin(["delisted", "bankrupt"])]
    df = df[has_outcome].reset_index(drop=True)
    # Excess vs SPY still carries each vintage's regime (equal-weight S&P lagged
    # cap-weighted SPY by ~7pp in 2020). Demean within vintage so every number
    # in the view reads "vs the equal-weight universe, same vintage"; keep the
    # SPY-relative figure alongside for the cohort table.
    for c in ("fwd_6m_excess", "fwd_12m_excess"):
        df[f"{c}_spy"] = df[c]
        df[c] = df[c] - df.groupby("vintage_year")[c].transform("mean")
    cheap = df[df.valuation_bucket == "cheap"].reset_index(drop=True)
    print(f"→ cheap cohort: {len(cheap)} rows ({cheap.fwd_12m_excess.notna().sum()} priced at 12 m)")

    # ── 1. cohort headline + exact splits ──────────────────────────
    cohorts = {name: _cell(df[df.valuation_bucket == name]) for name in ["cheap", "fair", "expensive", "negative_earnings"]}
    cohorts["all"] = _cell(df)
    # Delisted tickers have no price at the vintage date either, so no P/E and no
    # valuation bucket: they never enter the cheap cohort in the first place.
    survivorship = {
        "cheap_total": int(len(cheap)), "universe_total": int(len(df)) + int((~gone.index.isin(df.index)).sum()),
        "delisted_excluded": int(len(gone)), "delisted_with_valuation": int(gone.valuation_bucket.notna().sum()),
        "bankrupt_excluded": int((gone.terminal_event == "bankrupt").sum()),
    }
    splits = []
    for f in SPLIT_FEATS:
        vals = []
        for v, sub in cheap.groupby(cheap[f].astype(str)):
            if v in ("nan", "") or len(sub) < 3:
                continue
            c = _cell(sub); c["value"] = v.replace(".0", ""); c["thin"] = len(sub) < MIN_N
            vals.append(c)
        vals.sort(key=lambda c: -(c["excess_12m"] if c["excess_12m"] is not None else -999))
        splits.append({"feature": f, "values": vals})

    # market quality × leverage — the "cheap for a reason" grid
    mq_order = ["secular_growth", "stable", "cyclical", "disrupted", "declining"]
    lev_order = ["low", "moderate", "high", "negative_equity"]
    grid = [{"market_quality": mq, "cells": [dict(_cell(s := cheap[(cheap.market_quality == mq) & (cheap.leverage_bucket == lv)]),
                                                     leverage=lv) for lv in lev_order]} for mq in mq_order]

    # ── 2 + 3. Aito: relate inside the cohort, held-out quadrant ───
    relate = _relate(cheap, url, h)
    if CACHE.exists() and "--refresh" not in sys.argv:
        b = pd.read_csv(CACHE)
        print(f"→ reusing {CACHE} ({len(b)} held-out preds) — pass --refresh to re-query Aito")
    else:
        b = _heldout(df, cheap, url, h)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        b.to_csv(CACHE, index=False)
    real = cheap[["ticker", "vintage_year", "sector", "market_quality", "leverage_bucket", "moat_strength", "growth_bucket",
                  "pe_ratio", "debt_to_equity", "revenue_cagr_3y", "fwd_6m_excess", "fwd_12m_excess",
                  "outcome_bucket", "terminal_event", "total_return_pct_local"]].rename(columns={"vintage_year": "vintage"})
    b = b.merge(real, on=["ticker", "vintage"], how="left")

    def quadrants(pcol: str, ecol: str) -> dict:
        m_p, m_t = b[pcol].median(), b.ptrap.median()
        qs = {"rerate": (b[pcol] >= m_p) & (b.ptrap < m_t), "trap": (b[pcol] < m_p) & (b.ptrap >= m_t),
              "volatile": (b[pcol] >= m_p) & (b.ptrap >= m_t), "dull": (b[pcol] < m_p) & (b.ptrap < m_t)}
        out = {}
        for k, mask in qs.items():
            sub = b[mask]
            lr = sub[sub.outcome_bucket.notna() & (sub.outcome_bucket != "")]
            out[k] = {"n": int(len(sub)), "n_priced": int(sub[ecol].notna().sum()),
                      "excess": _r(sub[ecol].mean()), "median_excess": _r(sub[ecol].median()),
                      "hit": _r((sub[ecol] > 0).sum() / max(sub[ecol].notna().sum(), 1), 3),
                      "trap_rate": _r(lr.outcome_bucket.isin(TRAP).mean(), 3) if len(lr) else None,
                      "bankrupt": int((sub.terminal_event == "bankrupt").sum())}
        return {"median_p": _r(m_p, 4), "median_trap": _r(m_t, 4), "quadrants": out}

    points = [{"t": r.ticker, "v": int(r.vintage), "name": r.company_name if isinstance(r.company_name, str) else r.ticker,
               "p6": _r(r.p6, 4), "p12": _r(r.p12, 4), "pt": _r(r.ptrap, 4),
               "x6": _r(r.fwd_6m_excess), "x12": _r(r.fwd_12m_excess),
               "lr": r.outcome_bucket if isinstance(r.outcome_bucket, str) else None,
               "ev": r.terminal_event if isinstance(r.terminal_event, str) else None,
               "mq": r.market_quality if isinstance(r.market_quality, str) else None,
               "lev": r.leverage_bucket if isinstance(r.leverage_bucket, str) else None,
               "moat": r.moat_strength if isinstance(r.moat_strength, str) else None,
               "pe": _r(r.pe_ratio), "de": _r(r.debt_to_equity, 2),
               "g": _r(r.revenue_cagr_3y * 100 if pd.notna(r.revenue_cagr_3y) else None)}
              for r in b.itertuples(index=False)]
    whys = {f"{r.ticker}|{int(r.vintage)}": {k: json.loads(getattr(r, f"{k}_why")) for k in ("p12", "ptrap")
                                             if isinstance(getattr(r, f"{k}_why", None), str)}
            for r in b.itertuples(index=False)}

    # rank correlation of each predicted probability with what happened
    def spearman(a, c):
        s = b[[a, c]].dropna()
        return _r(s[a].rank().corr(s[c].rank()), 3) if len(s) > 10 else None
    b["is_trap"] = b.outcome_bucket.isin(TRAP).where(b.outcome_bucket.notna() & (b.outcome_bucket != ""))
    # Naive baseline: score = mean demeaned excess of the same sector in EARLIER vintages.
    for hz in ("6m", "12m"):
        col = f"fwd_{hz}_excess"
        b[f"base_{hz}"] = [df[(df.vintage_year < r.vintage) & (df.sector == r.sector)][col].mean() for r in b.itertuples()]

    def within_sector(a, c):
        t = b[[a, c, "sector"]].dropna()
        if len(t) <= 10:
            return None
        return _r(t.groupby("sector")[a].rank(pct=True).corr(t.groupby("sector")[c].rank(pct=True)), 3)

    def ic(g, a, c):
        return _r(g[a].rank().corr(g[c].rank()), 3) if g[c].notna().sum() > 10 else None

    skill = {"p6_vs_6m": spearman("p6", "fwd_6m_excess"), "p12_vs_12m": spearman("p12", "fwd_12m_excess"),
             "baseline_6m": spearman("base_6m", "fwd_6m_excess"), "baseline_12m": spearman("base_12m", "fwd_12m_excess"),
             "p6_within_sector": within_sector("p6", "fwd_6m_excess"),
             "p12_within_sector": within_sector("p12", "fwd_12m_excess"),
             "ptrap_vs_trap": spearman("ptrap", "is_trap"), "ptrap_vs_12m": spearman("ptrap", "fwd_12m_excess"),
             "by_vintage": [{"vintage": int(v), "n": int(len(g)),
                             "p6": ic(g, "p6", "fwd_6m_excess"), "baseline_6m": ic(g, "base_6m", "fwd_6m_excess"),
                             "p12": ic(g, "p12", "fwd_12m_excess"), "baseline_12m": ic(g, "base_12m", "fwd_12m_excess")}
                            for v, g in b.groupby("vintage")]}

    # Tercile tables per vintage: rank the cheap cohort by a walk-forward score,
    # cut into thirds, read off what actually happened. The cleanest test of
    # "does this score sort value traps from re-raters", with no cut-point tuning.
    def terciles(score: str) -> list[dict]:
        out = []
        for v, g in b.groupby("vintage"):
            g = g.dropna(subset=[score])
            if len(g) < 30:
                continue
            q = pd.qcut(g[score].rank(method="first"), 3, labels=["low", "mid", "high"])
            rows = []
            for lab in ["low", "mid", "high"]:
                sub = g[q == lab]
                lr = sub[sub.is_trap.notna()]
                rows.append({"tercile": lab, "n": int(len(sub)), "x6": _r(sub.fwd_6m_excess.mean()),
                             "x12": _r(sub.fwd_12m_excess.mean()),
                             "trap_rate": _r(lr.is_trap.astype(float).mean(), 3) if len(lr) else None})
            out.append({"vintage": int(v), "rows": rows})
        return out
    tercile_tables = {"ptrap": terciles("ptrap"), "p12": terciles("p12"), "baseline_12m": terciles("base_12m")}

    by_vintage = [dict(_cell(cheap[cheap.vintage_year == v]), vintage=int(v),
                       spy_note="COVID crash inside the 6 m window" if v == 2020 else None)
                  for v in sorted(cheap.vintage_year.unique())]

    payload = {
        "note": ("Cheap = P/E in the bottom band at the vintage date (valuation_bucket = cheap). "
                 "Mid-term outcome = total return over the next 126 / 252 trading days minus SPY over the same days, then demeaned "
                 "within vintage (= vs the equal-weight universe). "
                 "Long-run trap = outcome_bucket poor or disaster over the full 6–12-year window."),
        "bands": {"6m": 5, "12m": 8},
        "cohorts": cohorts, "survivorship": survivorship, "by_vintage": by_vintage,
        "splits": splits, "grid": {"leverage": lev_order, "rows": grid},
        "relate": relate,
        "heldout": {"n": int(len(b)), "skill": skill, "terciles": tercile_tables, "q6": quadrants("p6", "fwd_6m_excess"),
                    "q12": quadrants("p12", "fwd_12m_excess"), "points": points},
        "whys": whys,
        "caveats": [
            f"Survivorship: {survivorship['delisted_excluded']} universe rows (incl. {survivorship['bankrupt_excluded']} documented "
            "bankruptcies) delisted before today. Our price source (yfinance) drops delisted tickers, so they have no vintage price, "
            "no P/E, and never enter the cheap cohort. Those are disproportionately the value traps — every number here is "
            "measured on cheap stocks that survived to be priced, and flatters 'buy cheap'.",
            "Universe: S&P 500 members only. A company in a declining market rarely makes the index, so the "
            "'cheap · indebted · declining market' cell is nearly empty here — the hypothesis needs small caps to be tested properly.",
            "Three vintages (2014, 2017, 2020). Windows of different companies in the same vintage overlap in calendar time, "
            "so they share one market regime. Returns are shown vs the equal-weight universe of the same vintage (demeaned), which removes "
            "each vintage's level — e.g. the 2020 COVID crash — but not factor rotations within it (2014–2021 rewarded growth over value).",
            "Walk-forward: each vintage is scored by a table holding only earlier vintages, so no same-window returns leak in. "
            "The long-run trap label of a 2014 row still resolves in ~2026, so the trap axis is not strictly point-in-time. "
            "Quadrant cut points are the medians of the walk-forward probabilities.",
            "Baseline: 'which sector did well last time' (earlier-vintage sector mean) is reported next to the engine. "
            "Sector rotation dominates 12-month cross-sectional returns; the within-sector number is the stock-picking part.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    q = payload["heldout"]["q12"]["quadrants"]
    print(f"→ {OUT}: skill {skill} | 12m rerate-quadrant {q['rerate']['excess']}pp (trap {q['rerate']['trap_rate']}) "
          f"vs trap-quadrant {q['trap']['excess']}pp (trap {q['trap']['trap_rate']})")


if __name__ == "__main__":
    main()
