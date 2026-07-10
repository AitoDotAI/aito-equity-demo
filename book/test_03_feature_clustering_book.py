"""Booktest — does compression-based feature clustering fix the wide-feature
overconfidence that test_02 measured?

test_02 showed flat Naive Bayes over 16 correlated features is badly
overconfident (+19pt) and that held-out log-loss is minimised at ~2 features
— i.e. the extra features are double-counted, not informative. The fix tested
here keeps ALL features but stops them double-voting:

  • MDL clustering groups features that compress together into themes,
  • each theme votes once on the outcome (decorrelated),
  • members refine their theme conditionally (tempered residual β).

The scoreboard is the same held-out k-fold calibration. Win = lower held-out
log-loss / ECE / overconfidence than flat NB, WITHOUT dropping features.

Run / update:  ./do test-book -a
"""

from __future__ import annotations

import booktest as bt
import pytest

from pipeline.model import featureclust as fc


def test_feature_clustering(t: bt.TestCaseRun):
    if not fc.COMPANIES_CSV.exists():
        pytest.skip(f"{fc.COMPANIES_CSV} not present — run `./do pipeline load` first")

    df, features = fc.load_dataset()
    fold = fc.stratified_folds(df, 5)
    Y = fc.onehot(df)

    t.h1("Compression-clustered NB vs flat NB")
    t.tln(f"{len(df)} graded rows · {len(features)} features · 5-fold held-out")
    t.tln("")

    # two clustering resolutions: tight/interpretable vs aggressive/calibration
    clusters_tight = fc.cluster_features(df, features, min_nmi=0.30)
    clusters_loose = fc.cluster_features(df, features, min_nmi=0.08)

    # ── discovered clusters ──
    t.h2("1 · MDL feature clusters (the compression knob)")
    t.tln("min_nmi is the resolution: high → tight, interpretable themes;")
    t.tln("low → fold more correlation into fewer votes.")
    t.tln("")
    for label, cl in [("min_nmi=0.30 (tight / interpretable)", clusters_tight),
                      ("min_nmi=0.08 (aggressive / calibration)", clusters_loose)]:
        multi = [c for c in cl if len(c) > 1]
        t.tln(f"  {label} — {len(cl)} groups, {len(multi)} themes:")
        for c in multi:
            t.tln(f"     theme: {', '.join(c)}")
        t.tln("")

    # ── held-out comparison ──
    t.h2("2 · Held-out calibration: flat vs clustered (resolution × nuance)")
    t.tln("Same metrics as test_02. 'overconf' = mean confidence − accuracy")
    t.tln("(0 = calibrated, + = overconfident). Every model sees all 16 features.")
    t.tln("")

    models = [
        ("flat NB (16 indep)", lambda tr: _FlatWrap(fc.NB().fit(tr[features], tr["outcome_bucket"]), features)),
        ("clustered tight, β=0", lambda tr: fc.ClusteredNB(beta=0.0, clusters=clusters_tight).fit(tr, features)),
        ("clustered tight, β=0.5", lambda tr: fc.ClusteredNB(beta=0.5, clusters=clusters_tight).fit(tr, features)),
        ("clustered aggressive, β=0", lambda tr: fc.ClusteredNB(beta=0.0, clusters=clusters_loose).fit(tr, features)),
    ]
    t.tln(f"  {'model':28s} {'logloss':>8s} {'ece':>6s} {'acc':>6s} {'overconf':>9s} {'optimism':>9s}")
    results = {}
    for name, build in models:
        m = fc.metrics(fc.crossval_proba(df, features, fold, build), Y)
        results[name] = m
        t.tln(f"  {name:28s} {m['logloss']:8.3f} {m['ece']:6.3f} {m['acc']:6.3f} "
              f"{m['overconf']*100:+8.1f} {m['optimism_upside']*100:+8.1f}")
    t.tln("")
    flat = results["flat NB (16 indep)"]
    agg = results["clustered aggressive, β=0"]
    t.tln(f"  best calibration (aggressive β=0) vs flat:  log-loss "
          f"{agg['logloss']-flat['logloss']:+.3f} · ECE {agg['ece']-flat['ece']:+.3f} · "
          f"overconf {(agg['overconf']-flat['overconf'])*100:+.1f}pt")
    t.tln("  note: the β>0 within-cluster nuance term re-injects an outcome vote")
    t.tln("  and worsens calibration — the clean theme vote (β=0) wins. The")
    t.tln("  nuance must refine the theme, not add a parallel outcome vote.")
    t.tln("")

    # ── reliability of the best calibrated model ──
    t.h2("3 · Reliability — clustered aggressive β=0, by confidence bin")
    P = fc.crossval_proba(df, features, fold,
                          lambda tr: fc.ClusteredNB(beta=0.0, clusters=clusters_loose).fit(tr, features))
    conf, correct = P.max(1), (P.argmax(1) == Y.argmax(1)).astype(float)
    t.tln(f"  {'conf bin':12s} {'n':>5s} {'mean conf':>10s} {'accuracy':>9s} {'gap':>7s}")
    for b in range(4, 10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            t.tln(f"  {lo:.1f}–{hi:.1f}      {int(m.sum()):5d} {conf[m].mean()*100:9.1f}% "
                  f"{correct[m].mean()*100:8.1f}% {(conf[m].mean()-correct[m].mean())*100:+6.1f}")
    t.tln("")

    # ── findings ──
    t.h2("Findings (asserted)")
    t.assertln("clustering (aggressive, β=0) cuts held-out overconfidence vs flat NB",
               agg["overconf"] < flat["overconf"])
    t.assertln("clustering (aggressive, β=0) improves held-out log-loss vs flat NB",
               agg["logloss"] < flat["logloss"])
    t.assertln("tight clustering recovers an interpretable management theme",
               any(set(["leadership_quality", "execution_track_record"]).issubset(set(c)) for c in clusters_tight))


class _FlatWrap:
    """Adapts the column-restricted flat NB to the crossval_proba interface."""
    def __init__(self, nb, features):
        self.nb, self.features = nb, features

    def predict_proba(self, df):
        return self.nb.predict_proba(df[self.features])
