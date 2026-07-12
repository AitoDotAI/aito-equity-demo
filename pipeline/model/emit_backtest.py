"""Emit site/data/backtest.json — held-out fund backtest + confusion matrix.

Ranks every held-out company by the model's predicted expected outcome
(5-fold, ai:high, folds grouped by ticker so no company is in both train and
test), builds top-N funds, and compares their ACTUAL returns to the
equal-weight universe. Plus the ordinal confusion matrix, because top-1
accuracy is the wrong lens for a 5-bucket ordinal target.

    uv run python -m pipeline.model.emit_backtest      # → site/data/backtest.json
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from pipeline.aito.client import AitoClient
from pipeline.aito.load import df_to_aito_rows, load_schema
from pipeline.model import featureclust as fc

OUT = Path("site/data/backtest.json")
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
                rr = cl.post(f"{url}/api/v1/_predict", json={"from": "bt_tmp", "where": w, "predict": "outcome_bucket", "config": {"ai": "high"}}, headers=h)
                if rr.status_code >= 400:
                    continue
                d = {x["feature"]: x["$p"] for x in rr.json()["hits"]}
                recs.append({"true": r.outcome_bucket, "pred": max(d, key=d.get),
                             "score": sum(d.get(c, 0) * RANK[c] for c in CL),
                             "ret": float(r.total_return_pct_local), "cagr": float(r.cagr),
                             "vintage": int(r.vintage_year), "ticker": r.ticker})
        with AitoClient() as c:
            c.delete_table("bt_tmp")
    b = pd.DataFrame(recs); n = len(b)

    # confusion / adjacency
    cm = [[int(((b.true == a) & (b.pred == p)).sum()) for p in CL] for a in CL]
    dd = (b.pred.map(RANK) - b.true.map(RANK))
    conf = {"matrix": cm, "labels": CL, "exact": round((dd == 0).mean(), 3),
            "within_one": round((dd.abs() <= 1).mean(), 3), "off_by_3plus": round((dd.abs() >= 3).mean(), 3),
            "optimistic": round((dd > 0).mean(), 3), "pessimistic": round((dd < 0).mean(), 3)}

    # backtest
    mk = {"mean_total": round(b.ret.mean()), "median_total": round(b.ret.median()), "cagr": round(b.cagr.mean(), 1)}
    bs = b.sort_values("score", ascending=False)

    def fund(sub, label):
        return {"label": label, "n": len(sub), "mean_total": round(sub.ret.mean()),
                "median_total": round(sub.ret.median()), "cagr": round(sub.cagr.mean(), 1),
                "excess_cagr": round(sub.cagr.mean() - b.cagr.mean(), 1),
                "upside_hit": round(sub.true.isin(["good", "great"]).mean(), 2)}
    funds = [fund(bs.head(20), "Top 20"), fund(bs.head(50), "Top 50"), fund(bs.head(100), "Top 100"), fund(bs.tail(20), "Bottom 20")]

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
        "confusion": conf, "market": mk, "funds": funds, "per_vintage": per_vintage,
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
    print(f"→ {OUT}: top-20 CAGR {funds[0]['cagr']}%/yr vs market {mk['cagr']}%/yr | within-1-bin {conf['within_one']}")


if __name__ == "__main__":
    main()
