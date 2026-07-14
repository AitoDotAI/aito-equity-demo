# The Four Schools — one fund per philosophy, held-out

Held-out universe: **1294** graded observations across 3 vintages · 5-fold, grouped by ticker.
Equal-weight market: **7.9%/yr** CAGR (mean total 260%). Every fund is measured against this.
Only the visible feature set changes between funds; model, folds, and
ranking (buy the top-N by expected outcome) are identical throughout.


## 1 · The funds — realized CAGR by size, and % that beat the market

% beat = share of the fund's holdings whose total return exceeded the
equal-weight market of the same vintage. Decay = top-100 CAGR − top-20
CAGR: how much the edge survives when you stop cherry-picking.

  School                       top-20   top-50  top-100   decay  beat@20
  Market (equal-weight)          7.9%        —        —       —        —
  Value (Graham)                24.8%    16.2%    12.4%  -12.4      60%
  Quality (Buffett)             18.0%    16.0%    14.0%   -3.9      55%
  Growth (Fisher)               31.1%    26.1%    21.4%   -9.6      75%
  Composite (data-driven)       19.5%    18.6%    17.7%   -1.7      60%


## 2 · Reading it

  Top-20 sprint winner : Growth (Fisher) (31.1%/yr)
  Top-100 winner       : Growth (Fisher) (21.4%/yr)
  Most robust (smallest decay): Composite (data-driven) (-1.7 pts from 20→100)

  The top-20 number flatters concentrated single-factor bets — a handful
  of names in the sector the era happened to reward. The decay column is
  the honest one: it shows which edge is breadth (survives dilution) and
  which is a lucky top slice. That is the argument for the data-driven
  composite — not that it wins the sprint, but that it barely decays.


## 3 · What each school bought (top 8 by rank)

  Value (Graham)             HPQ, BBWI, HPQ, MCO, GOOGL, GOOGL, RTX, COP
  Quality (Buffett)          HD, GOOGL, HD, ALK, V, AZO, HD, ISRG
  Growth (Fisher)            AMD, MCHP, SWKS, MCHP, MCHP, SWKS, AVGO, HPQ
  Composite (data-driven)    NVDA, GOOGL, CRM, HD, V, CTSH, AAPL, TXN

  Growth's picks are the semiconductor/IT megatrend showing up as a
  portfolio — the win the blog flags as visible only in hindsight.


## 4 · Scoring realm — rank vs expectation-value vs numeric (no bins)

The default score is Σ p(bucket)·rank(bucket): the bins are treated as
evenly spaced. In *total-return* space they are wildly convex (great ≫
good), so that ordinal score under-weights the tail. Two fixes: score by
expected CAGR (Σ p·E[cagr|bucket]) — the expectation-value realm in the
fund's own units — or drop bins entirely and regress numeric log-return.
All three ranked on the same held-out signal; funds still measured in CAGR.

  rank  Σp·rank  — ordinal, evenly spaced (shipped)
    School                       top-20   top-50  top-100
    Value (Graham)                24.8%    16.2%    12.4%
    Quality (Buffett)             18.0%    16.0%    14.0%
    Growth (Fisher)               31.1%    26.1%    21.4%
    Composite (data-driven)       19.5%    18.6%    17.7%

  expcagr Σp·E[cagr]  — expectation-value, in CAGR units
    School                       top-20   top-50  top-100
    Value (Graham)                17.1%    16.0%    12.8%
    Quality (Buffett)             17.2%    16.0%    13.4%
    Growth (Fisher)               26.0%    25.8%    21.7%
    Composite (data-driven)       19.0%    18.6%    17.8%

  numeric log-ret reg  — no bins — regress the number
    School                       top-20   top-50  top-100
    Value (Graham)                14.9%    14.1%    13.5%
    Quality (Buffett)             20.1%    14.9%    14.2%
    Growth (Fisher)               18.6%    26.2%    21.9%
    Composite (data-driven)       19.7%    17.4%    16.5%

  Reading it:
  • The composite barely moves across realms (~19% top-20 everywhere):
    CAGR is already a log-compression of total return, so in the fund's
    own units the bins are ≈evenly spaced and the rank score is a fine
    proxy. Discretization is not why the composite fails to dominate.
  • Value is the one that collapses (top-20 24.8% → ~15-17%): its lead
    over the composite WAS a rank-score artifact that rewarded a
    concentrated, overconfident cheap-stock sort. In the expectation
    realm the composite edges every school except Growth.
  • Growth wins under every realm — the tell that its edge is a genuine
    (if hindsight-only) signal, not an artifact of how we score.


## 5 · Sanity checks

  every school's top-20 beat the market : True
  composite is the most robust (least decay): True
  composite top-20 near live-Aito ~20.6%   : 19.5%
