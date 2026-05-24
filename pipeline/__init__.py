"""Data pipeline for the aito-equity-demo.

Stages:
  1. universe/   — point-in-time index constituents per market
  2. filings/    — fetch 10-K + proxy filings before the vintage date
  3. extraction/ — LLM grading of qualitative features
  4. prices/     — forward returns and survival to today
  5. outcomes.py — bucketing (window-aware annualised thresholds)
  6. aito/       — schema + load + precomputed-query JSON output

Output: site/data/*.json (consumed directly by site/index.html at runtime).

US-only concrete implementations ship in v1; the abstract protocols
(UniverseSource, FilingFetcher, PriceSource) allow Nordic/EU sources
to slot in without touching downstream stages.
"""
