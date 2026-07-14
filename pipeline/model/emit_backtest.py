"""Emit site/data/backtest.json — held-out fund backtest + confusion matrix.

Ranks every held-out company by the model's predicted expected outcome
(5-fold, ai:high, folds grouped by ticker so no company is in both train and
test), builds top-N funds, and compares their ACTUAL returns to the
equal-weight universe. Plus the ordinal confusion matrix, because top-1
accuracy is the wrong lens for a 5-bucket ordinal target.

Two explanation layers ride along:
  - per pick: the predicted bucket's `$why` (base rate × per-feature lifts),
    the same tree Company Lab renders — so each holding is auditable.
  - per basket: a `_relate` over basket membership — "what kind of company
    got selected into this group" — the same query the Error Analysis view uses.

    uv run python -m pipeline.model.emit_backtest            # → site/data/backtest.json
    uv run python -m pipeline.model.emit_backtest --refresh  # re-query Aito (ignore cache)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from pipeline.aito.client import AitoClient
from pipeline.aito.load import df_to_aito_rows, load_schema
from pipeline.aito.queries import parse_why
from pipeline.model import featureclust as fc

OUT = Path("site/data/backtest.json")
CACHE = Path("data/backtest_records.csv")  # held-out preds (incl. $why) — re-emit JSON without re-querying Aito
CL = fc.CLASSES; RANK = {c: i for i, c in enumerate(CL)}
INT = {"leadership_quality", "capital_allocation", "strategic_clarity", "execution_track_record"}


def _wof(row, feats):
    w = {}
    for f in feats:
        v = row[f]
        if pd.isna(v) or v == "":
            continue
        w[f] = int(float(v)) if f in INT else (True if str(v) == "True" else (False if str(v) == "False" else str(v)))
    return w


def main() -> None:
    load_dotenv(override=True)
    url = os.environ["AITO_API_URL"].rstrip("/"); key = os.environ["AITO_API_KEY"]; h = {"x-api-key": key}
    df = pd.read_csv("data/companies.csv", low_memory=False)
    df = df[df.outcome_bucket.isin(CL) & df.total_return_pct_local.notna()].reset_index(drop=True)
    df["eval_fold"] = df["eval_fold"].astype(str)
    df["cagr"] = [((1 + r / 100) ** (1 / max(w, 0.5)) - 1) * 100 for r, w in zip(df.total_return_pct_local, df.window_years.fillna(10))]
    feats = [f for f in fc.DEFAULT_FEATURES if f in df.columns]
    coltypes = load_schema()["schema"]["companies"]["columns"]

    # The held-out predictions are the only expensive part (~1300 Aito predicts over
    # 5 folds). Cache them — incl. each pick's $why — so re-emitting the JSON is instant.
    if CACHE.exists() and "--refresh" not in sys.argv:
        b = pd.read_csv(CACHE)
        print(f"→ reusing {CACHE} ({len(b)} held-out preds) — pass --refresh to re-query Aito")
    else:
        recs = []
        for fold in ["0", "1", "2", "3", "4"]:
            train, test = df[df.eval_fold != fold], df[df.eval_fold == fold]
            with AitoClient() as c:
                c.delete_table("bt_tmp"); c.put_schema({"schema": {"bt_tmp": load_schema()["schema"]["companies"]}}); c.upload_batch("bt_tmp", df_to_aito_rows(train))
            with httpx.Client(timeout=60) as cl:
                for _, r in test.iterrows():
                    w = _wof(r, feats)
                    if not w:
                        continue
                    rr = cl.post(f"{url}/api/v1/_predict", json={"from": "bt_tmp", "where": w, "predict": "outcome_bucket",
                                 "select": ["feature", "$p", "$why"], "config": {"ai": "high"}}, headers=h)
                    if rr.status_code >= 400:
                        continue
                    hits = rr.json()["hits"]
                    d = {x["feature"]: x["$p"] for x in hits}
                    pred = max(d, key=d.get)
                    why = next((parse_why(x.get("$why")) for x in hits if x["feature"] == pred), None)
                    recs.append({"true": r.outcome_bucket, "pred": pred, "pred_p": round(float(d[pred]), 4),
                                 "score": sum(d.get(c, 0) * RANK[c] for c in CL),
                                 "ret": float(r.total_return_pct_local), "cagr": float(r.cagr),
                                 "vintage": int(r.vintage_year), "ticker": r.ticker,
                                 "company_name": r.company_name, "why_json": json.dumps(why)})
            with AitoClient() as c:
                c.delete_table("bt_tmp")
        b = pd.DataFrame(recs)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        b.to_csv(CACHE, index=False)
    n = len(b)

    # confusion / adjacency
    cm = [[int(((b.true == a) & (b.pred == p)).sum()) for p in CL] for a in CL]
    dd = (b.pred.map(RANK) - b.true.map(RANK))
    conf = {"matrix": cm, "labels": CL, "exact": round((dd == 0).mean(), 3),
            "within_one": round((dd.abs() <= 1).mean(), 3), "off_by_3plus": round((dd.abs() >= 3).mean(), 3),
            "optimistic": round((dd > 0).mean(), 3), "pessimistic": round((dd < 0).mean(), 3)}

    # backtest
    mk = {"mean_total": round(b.ret.mean()), "median_total": round(b.ret.median()), "cagr": round(b.cagr.mean(), 1)}
    bs = b.sort_values("score", ascending=False).reset_index(drop=True)

    whys: dict[str, dict] = {}  # "TICKER|vintage" → {label, prob, why} — shared so overlapping baskets don't duplicate

    def hkey(r):
        return f"{r.ticker}|{int(r.vintage)}"

    def holding(r):
        k = hkey(r)
        if k not in whys:
            why = json.loads(r.why_json) if isinstance(r.get("why_json"), str) and r.why_json else None
            whys[k] = {"label": r.pred, "prob": round(float(r.get("pred_p", 0) or 0), 4), "why": why}
        return {"key": k, "ticker": r.ticker, "company": r.get("company_name", r.ticker), "vintage": int(r.vintage),
                "ret": round(float(r.ret)), "cagr": round(float(r.cagr), 1), "bucket": r.true, "pred": r.pred}

    # per-basket relate — "what kind of company got selected here?" One membership table,
    # a boolean flag per basket, relate(flag=true) → the feature values that lift membership.
    feat_lookup = {(row.ticker, int(row.vintage_year)): row for _, row in df.iterrows()}
    basket_defs = [("top20", bs.head(20)), ("top50", bs.head(50)), ("top100", bs.head(100)), ("bottom20", bs.tail(20))]
    members = {name: set(hkey(r) for _, r in sub.iterrows()) for name, sub in basket_defs}
    bb_rows = []
    for _, r in bs.iterrows():
        src = feat_lookup.get((r.ticker, int(r.vintage)))
        if src is None:
            continue
        row = _wof(src, feats)
        k = hkey(r)
        for name in members:
            row[f"in_{name}"] = k in members[name]
        bb_rows.append(row)

    relate_by_basket: dict[str, list] = {name: [] for name, _ in basket_defs}
    if bb_rows:
        cols = {f: coltypes[f] for f in feats}
        cols.update({f"in_{name}": {"type": "Boolean", "nullable": True} for name, _ in basket_defs})
        with AitoClient() as c:
            c.delete_table("bt_baskets"); c.put_schema({"schema": {"bt_baskets": {"type": "table", "columns": cols}}})
            c.upload_batch("bt_baskets", bb_rows)
        for name, _ in basket_defs:
            rr = httpx.post(f"{url}/api/v1/_relate", json={"from": "bt_baskets", "where": {f"in_{name}": True},
                            "relate": feats, "limit": 10}, headers=h, timeout=90)
            if rr.status_code < 400:
                out = []
                for hit in rr.json().get("hits", []):
                    rel = hit.get("related", {})
                    field = next((k for k in rel if not k.startswith("$")), None)
                    if not field:
                        continue
                    val = rel[field].get("$has") if isinstance(rel[field], dict) else rel[field]
                    out.append({"feature": field, "value": str(val), "lift": round(hit.get("lift") or 0, 2),
                                "n": int(hit.get("fs", {}).get("f", 0))})
                out.sort(key=lambda x: -x["lift"])
                relate_by_basket[name] = out
        with AitoClient() as c:
            c.delete_table("bt_baskets")

    def fund(sub, label, rel_key):
        return {"label": label, "n": len(sub), "mean_total": round(sub.ret.mean()),
                "median_total": round(sub.ret.median()), "cagr": round(sub.cagr.mean(), 1),
                "excess_cagr": round(sub.cagr.mean() - b.cagr.mean(), 1),
                "upside_hit": round(sub.true.isin(["good", "great"]).mean(), 2),
                "holdings": [holding(r) for _, r in sub.iterrows()],
                "relate": relate_by_basket.get(rel_key, [])}
    funds = [fund(bs.head(20), "Top 20", "top20"), fund(bs.head(50), "Top 50", "top50"),
             fund(bs.head(100), "Top 100", "top100"), fund(bs.tail(20), "Bottom 20", "bottom20")]

    per_vintage = []
    for v in sorted(b.vintage.unique()):
        bv = b[b.vintage == v].sort_values("score", ascending=False); t = bv.head(20)
        per_vintage.append({"vintage": int(v), "window_years": round(float(df[df.vintage_year == v].window_years.median()), 1),
                            "market_mean": round(bv.ret.mean()), "top20_mean": round(t.ret.mean()),
                            "excess": round(t.ret.mean() - bv.ret.mean()), "names": t.ticker.head(10).tolist()})

    payload = {
        "note": ("Held-out backtest: every company is scored by the model's predicted expected "
                 "outcome (5-fold CV, ai:high grouping), then top-N funds are formed and measured "
                 "on ACTUAL returns. Folds are grouped by ticker, so no company appears in both "
                 "train and test."),
        "n_heldout": n, "accuracy": round((b.true == b.pred).mean(), 3),
        "confusion": conf, "market": mk, "funds": funds, "per_vintage": per_vintage, "whys": whys,
        "caveats": [
            "Folds are grouped by ticker — no company predicts its own other-vintage outcome.",
            "But every outcome window ends at the same date (~2026), so different companies' overlapping-period returns share one market regime — this is not a pure out-of-time backtest.",
            "The graded universe under-represents companies that delisted early (survivorship); the disaster bucket is thin.",
            "'Market' = equal-weight of the graded universe, not the cap-weighted S&P 500 (~13%/yr over this window).",
            "Top picks tilt mega-cap tech — a factor exposure that paid off in 2014–2020, not necessarily repeatable stock-picking skill.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"→ {OUT}: top-20 CAGR {funds[0]['cagr']}%/yr vs market {mk['cagr']}%/yr | "
          f"{len(whys)} whys · relate top20 {len(funds[0]['relate'])} feats")


if __name__ == "__main__":
    main()
