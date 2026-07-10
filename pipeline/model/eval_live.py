"""Evaluate the LIVE Aito `_predict` engine on the graded companies, so we can
see whether an aito-core change moved calibration.

The booktests (test_02/03) are a *pandas* NB reimplementation — Aito-independent.
This one hits the real engine: for each graded row it predicts outcome_bucket
from its features and scores the returned distribution against the true outcome,
with the same metrics. Compare the printed row to the baselines below.

NOTE: this is IN-SAMPLE (Aito's corpus contains the row it predicts), so it is
the optimistic number — compare it to the flat-NB *in-sample* baseline, not the
held-out one. Needs AITO_API_URL / AITO_API_KEY.

    uv run python -m pipeline.model.eval_live            # all 16 features, all rows
    uv run python -m pipeline.model.eval_live --sample 600 --features calib
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pipeline.aito.client import AitoClient
from pipeline.model import featureclust as fc

CALIB_FEATURES = ["market_position", "moat_type", "moat_strength", "market_quality",
                  "leadership_quality", "founder_still_ceo", "sector"]
# Aito column types — the where-clause must send the right JSON type, not the
# stringified value featureclust uses internally.
INT_FEATURES = {"leadership_quality", "capital_allocation", "strategic_clarity", "execution_track_record"}
BOOL_FEATURES = {"founder_still_ceo"}


def _to_aito(f: str, v):
    if f in INT_FEATURES:
        return int(float(v))
    if f in BOOL_FEATURES:
        return str(v).lower() in ("true", "1", "1.0")
    return str(v)

# Reference baselines from book/test_02 (pandas NB, same data):
BASELINES = {
    "flat NB in-sample (16f)":  {"logloss": 1.655, "ece": 0.170, "overconf": 0.171},
    "flat NB held-out (16f)":   {"logloss": 1.823, "ece": 0.194, "overconf": 0.194},
    "clustered held-out (fix)": {"logloss": 1.358, "ece": 0.071, "overconf": 0.067},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["all", "calib"], default="all")
    ap.add_argument("--sample", type=int, default=None, help="evaluate a stratified subset for speed")
    args = ap.parse_args()

    df, all_feats = fc.load_dataset()
    feats = all_feats if args.features == "all" else CALIB_FEATURES
    if args.sample and args.sample < len(df):
        fold = fc.stratified_folds(df, max(2, round(len(df) / args.sample)))
        df = df[fold == 0].reset_index(drop=True)
    print(f"→ live-Aito eval · {len(df)} rows · {len(feats)} features ({args.features}) · IN-SAMPLE")

    P = np.zeros((len(df), len(fc.CLASSES)))
    lat = []
    with AitoClient() as c:
        for i in range(len(df)):
            row = df.iloc[i]
            where = {}
            for f in feats:
                v = row[f]
                if v in (None, "", fc.MISSING) or (isinstance(v, float) and v != v):
                    continue
                where[f] = _to_aito(f, v)
            if not where:
                P[i] = [df["outcome_bucket"].eq(c).mean() for c in fc.CLASSES]  # base rate
                continue
            t0 = time.perf_counter()
            r = c.predict({"from": "companies", "where": where, "predict": "outcome_bucket",
                           "select": ["feature", "$p"]})
            lat.append((time.perf_counter() - t0) * 1000)
            dist = {h.get("feature"): float(h.get("$p", 0)) for h in r.get("hits", [])}
            v = np.array([dist.get(cl, 0.0) for cl in fc.CLASSES])
            P[i] = v / v.sum() if v.sum() > 0 else [1 / len(fc.CLASSES)] * len(fc.CLASSES)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(df)}")

    Y = fc.onehot(df)
    m = fc.metrics(P, Y)
    p50 = sorted(lat)[len(lat) // 2] if lat else 0

    print("\n=== LIVE AITO (this server build) ===")
    print(f"  log-loss {m['logloss']:.3f} · ECE {m['ece']:.3f} · acc {m['acc']:.3f} · "
          f"overconf {m['overconf']*100:+.1f}pt · optimism {m['optimism_upside']*100:+.1f}pt · "
          f"p50 {p50:.0f}ms")
    print("\n=== reference baselines (pandas NB, book/test_02) ===")
    for name, b in BASELINES.items():
        print(f"  {name:26s} log-loss {b['logloss']:.3f} · ECE {b['ece']:.3f} · "
              f"overconf {b['overconf']*100:+.1f}pt")
    print("\nReading: live ECE/overconf near 'flat in-sample' → engine unchanged (still flat NB).")
    print("         near 'clustered held-out' → the wide-feature fix is live and working.")

    # reliability, live
    conf, correct = P.max(1), (P.argmax(1) == Y.argmax(1)).astype(float)
    print("\n  conf bin     n   mean-conf  accuracy   gap")
    for bb in range(4, 10):
        lo, hi = bb / 10, (bb + 1) / 10
        mm = (conf > lo) & (conf <= hi)
        if mm.sum():
            print(f"  {lo:.1f}-{hi:.1f}   {int(mm.sum()):4d}   {conf[mm].mean()*100:7.1f}%  "
                  f"{correct[mm].mean()*100:6.1f}%  {(conf[mm].mean()-correct[mm].mean())*100:+5.1f}")


if __name__ == "__main__":
    main()
