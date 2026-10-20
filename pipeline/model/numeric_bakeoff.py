"""Held-out bake-off: bucketed vs numeric (quantity) evidence on rep2.

aito-core ticket td-20261020092559142882 (inference lane): on this data v2's default
Decimal quantity binning and `$numeric` extract almost no evidence, and both lose to
hand-made buckets. Re-run this unchanged as the before/after for a core fix:

    ./do pipeline numeric-bakeoff              # → data/numeric_bakeoff.csv + summary table
    AITO_ENV=<env> ./do pipeline numeric-bakeoff   # against a branched env

Baseline (core 2.11.7, 2026-10-20), accuracy / log-loss / rank-IC / top-100 CAGR:
  buckets 0.346 / 1.494 / 0.295 / 17.3 %   bare 0.342 / 1.515 / 0.276 / 14.1 %
  numeric 0.335 / 1.549 / 0.285 / 14.5 %
Writes a temp collection `nb_tmp` per fold (dropped at the end); everything else is read-only.

5-fold, folds grouped by ticker (eval_fold), training table = the other folds,
uploaded as a temp collection on master and dropped afterwards. Same rows,
same LLM grades + sector in every arm; only the fundamentals' encoding varies:

  buckets   valuation/growth/leverage/profitability/momentum/volatility buckets + valuation_vs_own
  bare      the raw numbers (pe_ratio, pe_vs_own, revenue_cagr_3y, debt_to_equity, return_on_equity,
            momentum_12m, volatility_12m) as plain where values
  numeric   the same numbers wrapped in {"$numeric": v}
"""
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from pipeline.aito.client import AitoClient, AitoConfig, hit_value
from pipeline.aito.load import df_to_aito_rows, load_schema

CL = ["disaster", "poor", "market", "good", "great"]
RANK = {c: i for i, c in enumerate(CL)}
LLM = ["market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality",
       "capital_allocation", "strategic_clarity", "execution_track_record", "sector"]
INT = {"leadership_quality", "capital_allocation", "strategic_clarity", "execution_track_record"}
BUCKETS = ["valuation_bucket", "growth_bucket", "leverage_bucket", "profitability_bucket",
           "momentum_bucket", "volatility_bucket", "valuation_vs_own"]
NUMS = ["pe_ratio", "pe_vs_own", "revenue_cagr_3y", "debt_to_equity", "return_on_equity", "momentum_12m", "volatility_12m"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/numeric_bakeoff.csv"


def where_for(r, arm):
    w = {}
    for f in LLM:
        v = r[f]
        if pd.notna(v) and v != "":
            w[f] = int(float(v)) if f in INT else str(v)
    if arm == "buckets":
        for f in BUCKETS:
            if pd.notna(r[f]) and r[f] != "":
                w[f] = str(r[f])
    else:
        for f in NUMS:
            v = r[f]
            if pd.notna(v) and math.isfinite(float(v)):
                w[f] = float(v) if arm == "bare" else {"$numeric": float(v)}
    return w


def main():
    cfg = AitoConfig.from_env()
    df = pd.read_csv("data/companies.csv", low_memory=False)
    df = df[df.outcome_bucket.isin(CL)].reset_index(drop=True)
    df["eval_fold"] = df.eval_fold.astype(str)
    df["cagr"] = [((1 + r / 100) ** (1 / max(w, 0.5)) - 1) * 100 for r, w in zip(df.total_return_pct_local, df.window_years.fillna(10))]
    tdef = load_schema()["schema"]["companies"]
    cols = [c for c in df.columns if c in tdef["columns"]]
    recs = []
    with httpx.Client(base_url=cfg.api_root, headers={"x-api-key": cfg.api_key}, timeout=60) as cl:
        for fold in ["0", "1", "2", "3", "4"]:
            train, test = df[df.eval_fold != fold], df[df.eval_fold == fold]
            with AitoClient() as c:
                c.delete_table("nb_tmp"); c.put_schema({"schema": {"nb_tmp": tdef}}); c.upload_batch("nb_tmp", df_to_aito_rows(train[cols]))

            def one(args):
                r, arm = args
                w = where_for(r, arm)
                rr = cl.post("/_predict", json={"from": "nb_tmp", "where": w, "predict": "outcome_bucket", "config": {"ai": "high"}})
                if rr.status_code >= 400:
                    return {"arm": arm, "err": rr.text[:200]}
                d = {hit_value(h): h["$p"] for h in rr.json()["hits"]}
                return {"arm": arm, "ticker": r.ticker, "vintage": int(r.vintage_year), "true": r.outcome_bucket,
                        "p_true": d.get(r.outcome_bucket, 1e-6), "pred": max(d, key=d.get),
                        "score": sum(d.get(k, 0) * RANK[k] for k in CL), "ret": r.total_return_pct_local, "cagr": r.cagr}
            jobs = [(r, arm) for _, r in test.iterrows() for arm in ("buckets", "bare", "numeric")]
            with ThreadPoolExecutor(8) as ex:
                recs += list(ex.map(one, jobs))
            print(f"fold {fold}: {len(test)} rows", flush=True)
    with AitoClient() as c:
        c.delete_table("nb_tmp")
    b = pd.DataFrame(recs)
    errs = b[b.get("err").notna()] if "err" in b else pd.DataFrame()
    if len(errs):
        print("errors:", len(errs), errs.err.iloc[0])
    b = b[b.ticker.notna()]
    b.to_csv(OUT, index=False)
    mk = b[b.arm == "buckets"].cagr.mean()
    print(f"\nheld-out n={b.ticker.nunique()} companies × vintages per arm; market CAGR {mk:.1f}%")
    for arm, g in b.groupby("arm"):
        g = g.sort_values("score", ascending=False)
        print(f"{arm:8} acc {(g.true == g.pred).mean():.3f}  logloss {-np.log(g.p_true.clip(1e-6)).mean():.3f}  "
              f"rank-IC {g.score.rank().corr(g.ret.rank()):.3f}  top20 {g.head(20).cagr.mean():5.1f}%  "
              f"top50 {g.head(50).cagr.mean():5.1f}%  top100 {g.head(100).cagr.mean():5.1f}%  bottom20 {g.tail(20).cagr.mean():5.1f}%")


if __name__ == "__main__":
    main()
