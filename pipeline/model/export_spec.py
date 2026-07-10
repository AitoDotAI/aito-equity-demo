"""Export a self-contained replication spec (JSON) for the wide-feature
calibration fix, for the aito-core team to reproduce against.

The JSON captures: the data source + feature schema, the redundancy (NMI)
that motivates the fix, the compression-clustering result at two resolutions,
the held-out metric table (flat NB vs clustered), and a handful of reference
predictions (out-of-fold) so a re-implementation can be checked row-for-row.

    uv run python -m pipeline.model.export_spec        # → docs/wide-feature-calibration.spec.json
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd

from pipeline.model import featureclust as fc

OUT = Path("docs/wide-feature-calibration.spec.json")
SCHEMA = Path("pipeline/aito/schema.json")
REFERENCE_ROWS = [  # (ticker, vintage) — span the outcome spectrum
    ("NVDA", 2017), ("COST", 2017), ("AAL", 2017),
    ("MMM", 2017), ("CHK", 2014), ("WIN", 2014),
]


def _round(o, n=4):
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: _round(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, n) for v in o]
    return o


def main() -> None:
    df, features = fc.load_dataset()
    fold = fc.stratified_folds(df, 5)
    Y = fc.onehot(df)
    types = json.loads(SCHEMA.read_text())["schema"]["companies"]["columns"]

    # ── feature schema (Aito-style + observed stats) ──
    feature_schema = {}
    for f in features:
        vals = sorted(df[f].unique().tolist())
        feature_schema[f] = {
            "aito_type": types.get(f, {}).get("type", "String"),
            "nullable": True,
            "cardinality": len(vals),
            "values": vals if len(vals) <= 12 else f"{len(vals)} distinct",
        }

    # ── redundancy (NMI) ──
    nmi_pairs = sorted(
        ({"a": a, "b": b, "nmi": fc.nmi(df[a], df[b])} for a, b in combinations(features, 2)),
        key=lambda p: p["nmi"], reverse=True,
    )

    # ── clusters at two resolutions ──
    clusters = {
        "0.30_tight": fc.cluster_features(df, features, min_nmi=0.30),
        "0.08_aggressive": fc.cluster_features(df, features, min_nmi=0.08),
    }

    # ── models: out-of-fold predictions + metrics ──
    builders = {
        "flat_nb": lambda tr: _Flat(fc.NB().fit(tr[features], tr["outcome_bucket"]), features),
        "clustered_tight_b0": lambda tr: fc.ClusteredNB(beta=0.0, clusters=clusters["0.30_tight"]).fit(tr, features),
        "clustered_tight_b0.5": lambda tr: fc.ClusteredNB(beta=0.5, clusters=clusters["0.30_tight"]).fit(tr, features),
        "clustered_aggressive_b0": lambda tr: fc.ClusteredNB(beta=0.0, clusters=clusters["0.08_aggressive"]).fit(tr, features),
    }
    oof, results = {}, {}
    for name, b in builders.items():
        P = fc.crossval_proba(df, features, fold, b)
        oof[name] = P
        results[name] = fc.metrics(P, Y)

    # ── reference predictions (out-of-fold) ──
    idx = {(r.ticker, int(r.vintage_year)): i for i, r in df.iterrows()}
    refs = []
    for tk, vy in REFERENCE_ROWS:
        i = idx.get((tk, vy))
        if i is None:
            continue
        refs.append({
            "ticker": tk, "vintage_year": vy,
            "true_outcome": df["outcome_bucket"].iloc[i],
            "features": {f: df[f].iloc[i] for f in features},
            "flat_nb_proba": dict(zip(fc.CLASSES, oof["flat_nb"][i].tolist())),
            "clustered_aggressive_proba": dict(zip(fc.CLASSES, oof["clustered_aggressive_b0"][i].tolist())),
        })

    spec = {
        "artifact": "wide-feature-calibration-spec",
        "version": "1.0",
        "summary": (
            "Naive-Bayes-family _predict over wide, correlated categorical feature "
            "spaces is overconfident: correlated features are multiplied as independent "
            "evidence, pushing probabilities to extremes. Fix = compression-cluster the "
            "features into themes and vote ONCE per theme. Validated held-out on the "
            "aito-equity-demo data; see results."
        ),
        "source": {
            "aito_db": "https://shared.aito.ai/db/aito-equity-demo",
            "table": "companies",
            "target": "outcome_bucket",
            "classes": fc.CLASSES,
            "class_is_ordinal": True,
            "ordinal_rank": fc.RANK,
            "n_rows_graded": int(len(df)),
            "note": "graded = rows with both LLM features and a realised outcome_bucket",
        },
        "feature_schema": feature_schema,
        "evaluation_protocol": {
            "scheme": "stratified 5-fold, deterministic",
            "fold_rule": "within each class, rows sorted by f'{ticker}_{vintage_year}', assigned fold = position % 5 (no RNG)",
            "metrics": {
                "logloss": "multiclass negative log-likelihood of the true class (lower better)",
                "ece": "expected calibration error on top-class confidence, 10 equal-width bins (lower better)",
                "accuracy": "argmax == true",
                "overconfidence": "mean top-class confidence − accuracy (0 = calibrated, + = overconfident)",
                "optimism_upside": "mean predicted P(good)+P(great) − realised frequency",
            },
        },
        "redundancy_nmi": {
            "definition": "NMI(X,Y) = I(X;Y) / sqrt(H(X)*H(Y)), natural-log entropies",
            "top_pairs": _round(nmi_pairs[:12]),
        },
        "clustering_algorithm": {
            "type": "agglomerative, complete-linkage, MDL + NMI-floor stop",
            "merge_rule": (
                "merge the cluster pair with the highest min-cross-pair NMI, provided "
                "(a) min-cross-pair MDL gain > 0 AND (b) min-cross-pair NMI >= min_nmi; "
                "stop when no pair qualifies"
            ),
            "mdl_gain": "N*I(X;Y)/ln(2) - 0.5*(|X|-1)*(|Y|-1)*log2(N)   # bits saved − BIC table cost",
            "min_nmi_is_resolution_knob": True,
            "note_bic_too_weak": (
                "At N~1300 pure BIC-MDL under-penalises (tiny MI pervades all features) and "
                "cascades into one mega-cluster; the NMI floor controls resolution. aito-core "
                "may prefer a predictor-aware description length instead."
            ),
            "clusters": {k: v for k, v in clusters.items()},
        },
        "model": {
            "flat_nb": {"alpha_laplace": fc.ALPHA, "structure": "P(y) * prod_f P(f|y)"},
            "clustered_nb": {
                "alpha_laplace": fc.ALPHA,
                "theme_encoder": (
                    "per multi-feature cluster: score each value by mean outcome-rank on the "
                    "TRAIN fold (target encoding), average member scores per row, quantise into "
                    "n_levels bins by train quantiles → one ordinal theme variable. Singletons "
                    "pass through unchanged."
                ),
                "n_levels": 4,
                "predict_beta0": "P(y) * prod_themes P(theme|y)   # one decorrelated vote per cluster",
                "predict_beta_residual": (
                    "+ beta * sum_{f in multi-cluster} log[P(f|theme,y)/P(f|theme)]   "
                    "# WARNING: this formulation re-injects an outcome vote and WORSENS "
                    "calibration. The intended nuance must refine the THEME assignment, not add "
                    "a parallel outcome vote. Open design question (see open_questions)."
                ),
            },
        },
        "results_heldout": _round(results),
        "headline": {
            "flat_overconfidence_pt": _round(results["flat_nb"]["overconf"] * 100, 1),
            "clustered_overconfidence_pt": _round(results["clustered_aggressive_b0"]["overconf"] * 100, 1),
            "flat_ece": _round(results["flat_nb"]["ece"]),
            "clustered_ece": _round(results["clustered_aggressive_b0"]["ece"]),
            "flat_logloss": _round(results["flat_nb"]["logloss"]),
            "clustered_logloss": _round(results["clustered_aggressive_b0"]["logloss"]),
        },
        "reference_predictions_oof": _round(refs),
        "open_questions": [
            "Nuance reformulation: encode within-cluster members so they refine the theme "
            "VALUE / disambiguate it, without adding a parallel outcome vote (the naive "
            "residual backfires).",
            "Interpretability vs calibration: tight themes maximise accuracy, aggressive "
            "merging maximises calibration. Capture residual cross-singleton correlation "
            "(e.g. a Chow-Liu forest over leftovers, or multi-resolution themes) for the "
            "Pareto sweet spot.",
            "Clustering criterion: NMI-floor is a pragmatic knob; consider a predictor-aware "
            "MDL (penalise the joint table's contribution to the PREDICTOR, not just the "
            "feature codebook) so the resolution is chosen, not tuned.",
            "Semantic disambiguation: a value's meaning is cluster-context-dependent "
            "(`C` in {car,license,driver} vs {Java,C++,programming}); conditioning likelihoods "
            "on the cluster gives context-sensitive encoding — verify on a feature with "
            "polysemous values.",
        ],
        "recommendation_for_aito_core": (
            "Implement compression-based feature grouping in the predict path so correlated "
            "columns vote once per group. The clean theme-vote (beta=0) already cuts ECE "
            f"{results['flat_nb']['ece']:.3f}→{results['clustered_aggressive_b0']['ece']:.3f} and "
            f"overconfidence {results['flat_nb']['overconf']*100:+.0f}→"
            f"{results['clustered_aggressive_b0']['overconf']*100:+.0f}pt held-out. Prioritise "
            "getting the grouping + single-vote right; treat conditional nuance as a v2 once the "
            "refine-not-revote formulation is settled."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"→ {OUT} ({OUT.stat().st_size} bytes)")
    print(f"  flat ECE {results['flat_nb']['ece']:.3f} overconf {results['flat_nb']['overconf']*100:+.1f}pt")
    print(f"  clustered ECE {results['clustered_aggressive_b0']['ece']:.3f} "
          f"overconf {results['clustered_aggressive_b0']['overconf']*100:+.1f}pt")


class _Flat:
    def __init__(self, nb, features):
        self.nb, self.features = nb, features

    def predict_proba(self, df):
        return self.nb.predict_proba(df[self.features])


if __name__ == "__main__":
    main()
