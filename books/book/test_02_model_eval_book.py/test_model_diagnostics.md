# Predictive model diagnostics

Graded rows with an outcome: **1294** · 16 features · 5 outcome classes · 5-fold held-out


## 1 · Base rates (the null model to beat)

  disaster    6.1%
  poor       11.8%
  market     20.6%
  good       31.9%
  great      29.5%
  upside (good+great):  61.4%


## 2 · In-sample vs held-out (all features)

In-sample = train and test on the same rows (what the demo's live
calibration effectively does — each row is predicted from a corpus
that still contains it). Held-out = the honest, leakage-free number.

  metric                   in-sample    held-out   reading
  log-loss                     1.655       1.823   lower better
  brier                        0.740       0.788   lower better
  accuracy                     0.423       0.399   higher better
  ECE (calibration err)        0.170       0.194   lower=calibrated
  mean top confidence          0.594       0.593   vs accuracy
  pred P(great)                0.277       0.278   actual 29.5%
  pred P(upside)               0.596       0.599   actual 61.4%

  OPTIMISM  (held-out pred P(upside) − actual): -1.6 pts
  OPTIMISM  (held-out pred P(great)  − actual): -1.7 pts
  OVERCONFIDENCE (held-out mean-confidence − accuracy): +19.4 pts


## 3 · Reliability — held-out, binned by top-class confidence

A calibrated model: confidence ≈ accuracy in every bin. Confidence
above accuracy = overconfident.

  conf bin         n  mean conf  accuracy     gap
  0.4–0.5        345      45.1%     36.5%   +8.6
  0.5–0.6        288      54.7%     36.5%  +18.3
  0.6–0.7        156      64.7%     42.9%  +21.8
  0.7–0.8        112      74.8%     50.9%  +23.9
  0.8–0.9         92      84.8%     58.7%  +26.1
  0.9–1.0        136      96.9%     41.9%  +55.0


## 4 · Feature-count ablation (does adding features help held-out?)

Features ranked by mutual information with the outcome, then added
top-down. If held-out log-loss / ECE bottoms out below the full set,
the extra features are overfitting, not informing.

  ranked by relevance: sector, volatility_bucket, momentum_bucket, moat_type, leadership_quality, valuation_bucket, capital_allocation, moat_strength, market_quality, execution_track_record, strategic_clarity, market_position, profitability_bucket, growth_bucket, leverage_bucket, founder_still_ceo

  # feats    logloss     ece     acc  optimism  overconf
         1     1.303   0.023   0.398     -0.3     +1.5
         2     1.291   0.033   0.416     -0.6     +3.0
         3     1.298   0.041   0.420     -0.2     +4.0
         4     1.315   0.064   0.418     -0.4     +6.3
         6     1.351   0.094   0.418     -0.7     +9.4
         9     1.508   0.145   0.408     -1.6    +14.5
        12     1.718   0.170   0.403     -2.3    +17.0
        16     1.823   0.194   0.399     -1.6    +19.4

  → held-out log-loss is minimised at **2 features** (full set = 16).


## 5 · Feature redundancy (double-counted evidence)

Pairwise normalised mutual information between features. NB assumes
independence; high-NMI pairs violate it and inflate confidence.

  most redundant pairs (NMI):
    0.681  leadership_quality ~ execution_track_record
    0.622  leadership_quality ~ capital_allocation
    0.564  market_position ~ moat_strength
    0.484  leadership_quality ~ strategic_clarity
    0.422  strategic_clarity ~ execution_track_record
    0.410  capital_allocation ~ strategic_clarity
    0.396  capital_allocation ~ execution_track_record
    0.363  moat_strength ~ strategic_clarity
    0.343  moat_strength ~ leadership_quality
    0.342  moat_strength ~ capital_allocation


## Findings (asserted)

ok
ok
ok
