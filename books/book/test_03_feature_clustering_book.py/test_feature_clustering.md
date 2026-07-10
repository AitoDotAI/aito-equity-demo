# Compression-clustered NB vs flat NB

1294 graded rows · 16 features · 5-fold held-out


## 1 · MDL feature clusters (the compression knob)

min_nmi is the resolution: high → tight, interpretable themes;
low → fold more correlation into fewer votes.

  min_nmi=0.30 (tight / interpretable) — 12 groups, 2 themes:
     theme: leadership_quality, execution_track_record, strategic_clarity, capital_allocation
     theme: market_position, moat_strength

  min_nmi=0.08 (aggressive / calibration) — 6 groups, 2 themes:
     theme: market_position, moat_strength, leadership_quality, execution_track_record, strategic_clarity, capital_allocation, moat_type, market_quality, valuation_bucket, profitability_bucket
     theme: sector, volatility_bucket


## 2 · Held-out calibration: flat vs clustered (resolution × nuance)

Same metrics as test_02. 'overconf' = mean confidence − accuracy
(0 = calibrated, + = overconfident). Every model sees all 16 features.

  model                         logloss    ece    acc  overconf  optimism
  flat NB (16 indep)              1.823  0.194  0.399    +19.4     -1.6
  clustered tight, β=0            1.543  0.149  0.415    +14.9     -0.6
  clustered tight, β=0.5          1.567  0.178  0.399    +17.8     +0.7
  clustered aggressive, β=0       1.358  0.071  0.375     +6.7     -0.3

  best calibration (aggressive β=0) vs flat:  log-loss -0.466 · ECE -0.123 · overconf -12.7pt
  note: the β>0 within-cluster nuance term re-injects an outcome vote
  and worsens calibration — the clean theme vote (β=0) wins. The
  nuance must refine the theme, not add a parallel outcome vote.


## 3 · Reliability — clustered aggressive β=0, by confidence bin

  conf bin         n  mean conf  accuracy     gap
  0.4–0.5        454      44.5%     35.0%   +9.5
  0.5–0.6        238      54.6%     53.8%   +0.8
  0.6–0.7         72      65.3%     44.4%  +20.8
  0.7–0.8         19      72.6%     47.4%  +25.3
  0.8–0.9          3      88.2%    100.0%  -11.8
  0.9–1.0          6      92.1%    100.0%   -7.9


## Findings (asserted)

ok
ok
ok
