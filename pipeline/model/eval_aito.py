"""Honest held-out evaluation of the LIVE Aito engine via `/api/v1/_evaluate`,
which MASKS each test row so the model can't see its own answer.

Use this — not an in-sample predict loop — to judge whether an aito-core change
improved calibration. In-sample numbers flatter the model and hide exactly the
overfitting we care about (this tool exists because an in-sample pass once told
us the wide-feature problem was "fixed"; _evaluate showed it was not).

It sweeps vintage holdouts × feature width and reports Aito's own metrics:

  accuracyGain   accuracy − baseAccuracy        (model picks the mode better)
  informationGain  baseEntropy − crossEntropy   (bits; >0 = probabilities beat
                   base rate, <0 = WORSE than base — the key calibration signal)
  geomMeanLift   geometric-mean P(true)/base    (>1 better)

Reading for the wide-feature question: if 16 features score LOWER infoGain than
7, the engine is still double-counting correlated features. The fix (see
docs/wide-feature-calibration.md) should make wide ≥ narrow.

NOTE: vintages share companies (NVDA 2014/2017/2020), so a vintage holdout has
mild cross-vintage leakage that *helps* the model — findings of degradation
despite that are conservative.

    uv run python -m pipeline.model.eval_aito
"""

from __future__ import annotations

import argparse

import httpx

from pipeline.aito.client import AitoClient

F16 = ["market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality",
       "founder_still_ceo", "sector", "valuation_bucket", "growth_bucket", "leverage_bucket",
       "profitability_bucket", "momentum_bucket", "volatility_bucket", "capital_allocation",
       "strategic_clarity", "execution_track_record"]
F7 = ["market_position", "moat_type", "moat_strength", "market_quality",
      "leadership_quality", "founder_still_ceo", "sector"]
KEYS = ["n", "accuracy", "baseAccuracy", "accuracyGain", "informationGain", "mxe", "h", "geomMeanLift"]


def evaluate(client: AitoClient, feats: list[str], test: dict) -> dict:
    where = {f: {"$get": f} for f in feats}
    body = {"test": test, "evaluate": {"from": "companies", "where": where, "predict": "outcome_bucket"}}
    return client._post_json("/api/v1/_evaluate", body)


CALIBRATED = ["sector", "momentum_bucket", "volatility_bucket", "market_quality",
              "valuation_bucket", "moat_strength"]


def _wmean(rows: list[dict], key: str) -> float:
    N = sum(r["n"] for r in rows)
    return sum(r["n"] * r[key] for r in rows) / N if N else float("nan")


def random_cv() -> None:
    """5-fold random cross-year CV via the eval_fold column (mixes vintages, so
    it isolates the redundancy/calibration effect from the yearly regime trend).
    Requires `./do pipeline load` to have written eval_fold."""
    print("→ live Aito _evaluate · 5-fold RANDOM cross-year CV (eval_fold)\n")
    print(f"  {'feature set':14s} {'infoGain':>9} {'gmLift':>7} {'accGain':>8}   per-fold infoGain")
    with AitoClient() as c:
        for label, feats in [("7", F7), ("16", F16), ("calibrated-6", CALIBRATED)]:
            rows = []
            for k in range(5):
                try:
                    rows.append(evaluate(c, feats, {"eval_fold": str(k)}))
                except httpx.HTTPStatusError as e:
                    print(f"  {label:14s} ERROR {e.response.text[:80]}")
                    rows = []
                    break
            if not rows:
                continue
            pf = ", ".join(f"{r['informationGain']:+.2f}" for r in rows)
            print(f"  {label:14s} {_wmean(rows,'informationGain'):>+9.3f} {_wmean(rows,'geomMeanLift'):>7.3f} "
                  f"{_wmean(rows,'accuracyGain'):>+8.3f}   [{pf}]")
    print("\ninfoGain>0 = probabilities beat base rate. 16f<7f = wide-feature penalty;")
    print("calibrated-6 recovers it → the residual tail is noise/thin-data, not the year trend.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", type=int, nargs="*", default=[2014, 2017, 2020])
    ap.add_argument("--random", action="store_true", help="5-fold random cross-year CV (needs eval_fold column)")
    args = ap.parse_args()

    if args.random:
        random_cv()
        return

    print("→ live Aito _evaluate · masked held-out · vintage holdouts × feature width\n")
    hdr = f"{'holdout':>8} {'feats':>5} {'n':>4} {'acc':>6} {'baseAcc':>7} {'accGain':>7} {'infoGain':>8} {'mxe':>6} {'h':>6} {'gmLift':>6}"
    print(hdr)
    with AitoClient() as c:
        for vy in args.vintages:
            for label, feats in [("7", F7), ("16", F16)]:
                try:
                    r = evaluate(c, feats, {"vintage_year": vy})
                except httpx.HTTPStatusError as e:
                    print(f"{vy:>8} {label:>5}  ERROR {e.response.text[:80]}")
                    continue
                print(f"{vy:>8} {label:>5} {int(r['n']):>4} {r['accuracy']:>6.3f} {r['baseAccuracy']:>7.3f} "
                      f"{r['accuracyGain']:>+7.3f} {r['informationGain']:>+8.3f} {r['mxe']:>6.3f} "
                      f"{r['h']:>6.3f} {r['geomMeanLift']:>6.3f}")
    print("\ninfoGain<0 → held-out probabilities worse than base rate.")
    print("16-feature infoGain < 7-feature → wide-feature double-counting NOT fixed.")


if __name__ == "__main__":
    main()
