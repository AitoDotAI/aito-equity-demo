"""Emit site/data/eval.json — honest held-out evaluation metrics for the demo's
Evaluation view, straight from Aito's masked `_evaluate` (it hides each test row
from the model). Uses the eval_fold column for a random cross-year 5-fold CV so
the numbers aren't inflated by the yearly regime trend or in-sample leakage.

Slow (~30 server-side _evaluate calls); run once and commit the JSON:

    uv run python -m pipeline.model.emit_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.aito.client import AitoClient
from pipeline.model import eval_aito as ea
from pipeline.model import featureclust as fc

OUT = Path("site/data/eval.json")
FOLDS = ["0", "1", "2", "3", "4"]
SWEEP_K = [1, 2, 3, 4, 6, 9, 12, 16]


def _wmean(rows: list[dict], key: str) -> float:
    n = sum(r["n"] for r in rows)
    return sum(r["n"] * r[key] for r in rows) / n if n else 0.0


def _cv(c: AitoClient, feats: list[str]) -> dict:
    rows = [ea.evaluate(c, feats, {"eval_fold": k}) for k in FOLDS]
    return {
        "accuracy": round(_wmean(rows, "accuracy"), 3),
        "base_accuracy": round(_wmean(rows, "baseAccuracy"), 3),
        "accuracy_gain": round(_wmean(rows, "accuracyGain"), 3),
        "information_gain": round(_wmean(rows, "informationGain"), 3),
        "geom_mean_lift": round(_wmean(rows, "geomMeanLift"), 3),
        "mean_rank": round(_wmean(rows, "meanRank"), 3),
        "base_mean_rank": round(_wmean(rows, "baseMeanRank"), 3),
    }


def main() -> None:
    df, _ = fc.load_dataset()
    n_graded = len(df)

    with AitoClient() as c:
        print("→ 5-fold CV: calibrated feature set")
        cal = _cv(c, ea.CALIBRATED)
        print("→ 5-fold CV: all 16 features")
        allf = _cv(c, ea.F16)

        print("→ single-feature ranking (fold 0)")
        singles = {f: ea.evaluate(c, [f], {"eval_fold": "0"})["informationGain"] for f in ea.F16}
        rank = sorted(ea.F16, key=lambda f: singles[f], reverse=True)

        print("→ feature-count sweep (fold 0)")
        sweep = []
        for k in SWEEP_K:
            r = ea.evaluate(c, rank[:k], {"eval_fold": "0"})
            sweep.append({"k": k, "information_gain": round(r["informationGain"], 3),
                          "accuracy": round(r["accuracy"], 3), "added": rank[k - 1]})
        peak_k = max(sweep, key=lambda s: s["information_gain"])["k"]

    payload = {
        "note": ("Held-out via Aito _evaluate, which masks each test row from the "
                 "model. Random cross-year 5-fold CV (companies grouped by ticker so "
                 "no company appears in both train and test). These are the honest, "
                 "leakage-free numbers — not in-sample."),
        "n_graded": n_graded,
        "classes": fc.CLASSES,
        "metrics_glossary": {
            "accuracy": "share of held-out companies whose top-predicted bucket was correct",
            "accuracy_gain": "accuracy above always guessing the most common bucket",
            "information_gain": "bits the predicted probabilities beat the base rate (>0 good, <0 worse than base)",
            "geom_mean_lift": "geometric-mean probability assigned to the true bucket vs base (>1 good)",
        },
        "headline": {"feature_set": "calibrated", "features": ea.CALIBRATED, **cal},
        "comparison": [
            {"label": "base rate", "sub": "guess the most common bucket",
             "accuracy": cal["base_accuracy"], "information_gain": 0.0},
            {"label": "calibrated features", "sub": f"{len(ea.CALIBRATED)} decorrelated features",
             "accuracy": cal["accuracy"], "information_gain": cal["information_gain"]},
            {"label": "all 16 features", "sub": "everything, incl. redundant/noisy",
             "accuracy": allf["accuracy"], "information_gain": allf["information_gain"]},
        ],
        "feature_sweep": sweep,
        "peak_k": peak_k,
        "single_feature_ranking": [
            {"feature": f, "information_gain": round(singles[f], 3)} for f in rank
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"→ {OUT}: headline acc {cal['accuracy']} (base {cal['base_accuracy']}), "
          f"infoGain {cal['information_gain']} | all-16 infoGain {allf['information_gain']} | peak k={peak_k}")


if __name__ == "__main__":
    main()
