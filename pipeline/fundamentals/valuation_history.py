"""Cheap versus *what*? A company's P/E against its own history.

A stock that has traded at a low P/E for years (a utility, a bank, a tobacco
company) is cheap by habit: there is no "normal" for it to revert to. A stock
whose P/E has fallen well below its own usual level is cheap *relative to
itself* — the case a mean-reversion trader is betting on. Absolute P/E can't
tell the two apart; this stage adds the company's own yardstick.

Per (ticker, vintage):
  pe_hist_<k>y      P/E at the start of each of the 5 years before the vintage,
                    built exactly like the vintage P/E: as-traded price on the
                    first trading day of that year × shares / net income, both
                    from the latest 10-K FILED before that date (point-in-time)
  pe_own_median_5y  median of the positive ones (needs ≥ 3)
  market_cap_pit    point-in-time market cap at the vintage (as-traded price ×
                    split-consistent share count) — the loader derives P/E, P/B
                    and earnings yield from this
  (pe_vs_own = vintage P/E / pe_own_median_5y is derived in the loader;
   < 1 = cheaper than the company usually is)

    uv run python -m pipeline.fundamentals.valuation_history   # → data/valuation_history.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from pipeline.fundamentals.sec_xbrl import (
    CONCEPTS, _concept_points, _latest_before, _session, compute_fundamentals, fetch_companyfacts, load_cik_index,
)

YEARS_BACK = 5
MIN_POINTS = 3


def raw_closes(ticker: str, start: date, end: date) -> tuple[pd.Series, pd.Series]:
    """Split-adjusted closes (no dividend adjustment) + the split series, to undo later splits."""
    t = yf.Ticker(ticker)
    h, splits = pd.DataFrame(), None
    for attempt in range(3):  # Yahoo intermittently answers a live ticker with no rows
        try:
            h = t.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False, actions=False)
            splits = t.splits
        except Exception:
            h = pd.DataFrame()
        if not h.empty:
            break
        time.sleep(2 * (attempt + 1))
    if h.empty or "Close" not in h:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    c = h["Close"].dropna()
    c.index = (c.index.tz_localize(None) if c.index.tz is not None else c.index).normalize()
    if splits is None or not len(splits):
        return c, pd.Series(dtype=float)
    splits.index = (splits.index.tz_localize(None) if splits.index.tz is not None else splits.index).normalize()
    return c, splits


def _split_factor(splits: pd.Series, after: pd.Timestamp, upto: pd.Timestamp | None = None) -> float:
    if not len(splits):
        return 1.0
    m = splits.index > after
    if upto is not None:
        m &= splits.index <= upto
    return float(np.prod(splits.values[m])) if m.any() else 1.0


def as_traded(closes: pd.Series, splits: pd.Series, on: date) -> tuple[float, pd.Timestamp] | None:
    """The price that traded on the first trading day at/after `on` (same rule as vintage_price)."""
    after = closes[closes.index >= pd.Timestamp(on)]
    if after.empty or (after.index[0] - pd.Timestamp(on)).days > 10:
        return None
    d0 = after.index[0]
    return float(after.iloc[0]) * _split_factor(splits, d0), d0


def _shares_asof(facts: dict, on: date) -> tuple[float, pd.Timestamp] | None:
    """Common shares outstanding known before `on`, and the date they are stated at.

    Order of trust:
      1. dei:EntityCommonStockSharesOutstanding — the cover-page count of COMMON
         shares, on every 10-K and 10-Q; summed across classes in one filing
         (GOOGL reports A, B and C separately).
      2. us-gaap WeightedAverageNumberOfDilutedSharesOutstanding (annual).
      3. us-gaap CommonStockSharesOutstanding — last, because some filers tag a
         preferred class with it (MRK 3,076; SPG 8,000), which gave P/E ≈ 0.0001.
    """
    pts = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    eligible = [p for p in pts if p.get("filed") and p.get("end") and p.get("val") and date.fromisoformat(p["filed"]) < on]
    if eligible:
        last = max(eligible, key=lambda p: (p["filed"], p["end"]))
        same = [p for p in eligible if p.get("accn") == last.get("accn") and p["end"] == last["end"]]
        return float(sum(p["val"] for p in same)), pd.Timestamp(last["end"])
    for key in ("WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding"):
        hit = _latest_before(_concept_points(facts, key), on)
        if hit and hit.get("val"):
            return float(hit["val"]), pd.Timestamp(hit["end"])
    return None


MIN_MARKET_CAP = 1e9  # an S&P 500 member below $1B means the share count is wrong, not the company tiny


def market_cap_at(facts: dict, closes: pd.Series, splits: pd.Series, on: date) -> tuple[float | None, float | None]:
    """(market cap, net income) at `on`, point-in-time.

    Shares come from the latest 10-K filed before `on` and are stated at that
    10-K's balance-sheet date. A split between that date and `on` (KO, Aug 2012)
    would otherwise pair a pre-split count with a post-split price, so the count
    is scaled by the splits in between.
    """
    ni = compute_fundamentals(facts, on).get("net_income")
    sh = _shares_asof(facts, on)
    px = as_traded(closes, splits, on)
    if sh is None or px is None:
        return None, ni
    shares, as_of = sh
    price, d0 = px
    mcap = price * shares * _split_factor(splits, as_of, d0)
    return (mcap if mcap >= MIN_MARKET_CAP else None), ni


def pe_at(facts: dict, closes: pd.Series, splits: pd.Series, on: date) -> float | None:
    mcap, ni = market_cap_at(facts, closes, splits, on)
    return mcap / ni if mcap and ni else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="data/universe.csv")
    ap.add_argument("--out", default="data/valuation_history.csv")
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.universe)
    if args.tickers:
        df = df[df.ticker.isin(args.tickers)]
    cik_index = load_cik_index()
    rows: list[dict] = []
    with _session() as client:
        for i, (ticker, g) in enumerate(df.groupby("ticker"), 1):
            cik = cik_index.get(ticker)
            facts = fetch_companyfacts(cik, client) if cik else None
            vds = [date.fromisoformat(v) for v in g.vintage_date]
            closes, splits = raw_closes(ticker, min(vds) - timedelta(days=366 * YEARS_BACK + 30), max(vds) + timedelta(days=10))
            for r, vd in zip(g.itertuples(index=False), vds):
                row = {"ticker": ticker, "vintage_year": int(r.vintage_year), "vintage_date": r.vintage_date}
                if facts is not None and not closes.empty:
                    mcap, _ = market_cap_at(facts, closes, splits, vd)
                    row["market_cap_pit"] = mcap
                    hist = []
                    for k in range(1, YEARS_BACK + 1):
                        pe = pe_at(facts, closes, splits, date(vd.year - k, vd.month, vd.day))
                        row[f"pe_hist_{k}y"] = round(pe, 2) if pe is not None else None
                        if pe is not None and pe > 0:
                            hist.append(pe)
                    if len(hist) >= MIN_POINTS:
                        row["pe_own_median_5y"] = round(float(np.median(hist)), 2)
                        row["pe_own_n"] = len(hist)
                rows.append(row)
            if i % 50 == 0:
                print(f"  {i} tickers")
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(out)} rows · {out.get('pe_own_median_5y', pd.Series()).notna().sum()} with an own 5-y P/E norm)")


if __name__ == "__main__":
    main()
