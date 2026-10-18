"""Mid-horizon outcomes: does a cheap stock re-rate within 6–12 months?

The long-horizon target asks "company at vintage → 6–12-year outcome"; the
post-filing stage asks about the next 20 trading days. This stage fills the
gap a mid-term value trader actually holds for: 6 and 12 months from the
vintage date, measured as EXCESS return over SPY so a market-wide move (the
2020 vintage's 6-month window is the COVID crash) doesn't masquerade as a
company signal.

Per (ticker, vintage):
  fwd_6m_return     raw total return, vintage → +126 trading days (%)
  fwd_6m_excess     fwd_6m_return − SPY's return over the same days (pp)
  fwd_6m_bucket     lagged / inline / outperformed  (±5pp)
  fwd_12m_*         same at +252 trading days       (±8pp)

The anchor is the vintage date (not the filing date) because that is where
the valuation features are struck: pe_ratio = vintage_price × shares /
latest-10-K net income.

Survivorship — read this before trusting the cohort numbers. yfinance has no
history for delisted tickers, and a delisted ticker may since have been
reissued to an unrelated company. So only rows whose long-horizon outcome is
`trading` are priced; delisted/bankrupt rows get no mid-term outcome. Those
rows are disproportionately the value traps this target is meant to find, so
the priced cohort is biased toward cheap stocks that worked out. The view
states the gap; closing it needs a price source with delisted history.

    uv run python -m pipeline.prices.midterm      # → data/midterm_outcomes.csv
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

HORIZONS = {"6m": (126, 5.0), "12m": (252, 8.0)}  # label → (trading days, ±band in pp)
BENCHMARK = "SPY"


def bucket_excess(excess_pp: float, band: float) -> str:
    if excess_pp <= -band:
        return "lagged"
    if excess_pp >= band:
        return "outperformed"
    return "inline"


def _closes(ticker: str, start: date, end: date) -> pd.Series:
    try:
        h = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True, actions=False)
    except Exception:
        return pd.Series(dtype=float)
    if h.empty or "Close" not in h:
        return pd.Series(dtype=float)
    c = h["Close"].dropna()
    idx = c.index.tz_localize(None) if c.index.tz is not None else c.index
    return pd.Series(c.values, index=idx.normalize())


def _fwd(closes: pd.Series, t0: date, td: int) -> tuple[float, pd.Timestamp, pd.Timestamp] | None:
    """Return over `td` trading days from the first trading day at/after t0."""
    after = closes[closes.index >= pd.Timestamp(t0)]
    if len(after) <= td or after.iloc[0] <= 0:
        return None
    return (float(after.iloc[td]) / float(after.iloc[0]) - 1) * 100, after.index[0], after.index[td]


def compute_rows(universe: pd.DataFrame, outcomes: pd.DataFrame) -> list[dict]:
    keys = ["ticker", "vintage_year", "vintage_date"]
    df = universe[keys].merge(outcomes[keys + ["terminal_event"]], on=keys, how="left")
    # 2026-vintage rows have no long-horizon outcome yet; they are current
    # constituents, so they trade. Everything else must be `trading`.
    df = df[(df.terminal_event == "trading") | (df.vintage_year >= 2026)]
    n_skipped = len(universe) - len(df)

    vdates = sorted({date.fromisoformat(v) for v in df.vintage_date})
    end = max(vdates) + timedelta(days=400)
    spy = _closes(BENCHMARK, min(vdates) - timedelta(days=7), min(end, date.today()))

    rows: list[dict] = []
    for i, (ticker, g) in enumerate(df.groupby("ticker"), 1):
        g_dates = [date.fromisoformat(v) for v in g.vintage_date]
        closes = _closes(ticker, min(g_dates) - timedelta(days=7), min(max(g_dates) + timedelta(days=400), date.today()))
        for r, vd in zip(g.itertuples(index=False), g_dates):
            row = {"ticker": ticker, "vintage_year": int(r.vintage_year), "vintage_date": r.vintage_date}
            for label, (td, band) in HORIZONS.items():
                got = _fwd(closes, vd, td) if not closes.empty else None
                if got is None:
                    continue
                ret, d0, d1 = got
                s0, s1 = spy.get(d0), spy.get(d1)
                if s0 is None or s1 is None or s0 <= 0:
                    continue
                excess = ret - (s1 / s0 - 1) * 100
                row[f"fwd_{label}_return"] = round(ret, 2)
                row[f"fwd_{label}_excess"] = round(excess, 2)
                row[f"fwd_{label}_bucket"] = bucket_excess(excess, band)
            rows.append(row)
        if i % 100 == 0:
            print(f"  {i} tickers")
    print(f"  skipped {n_skipped} rows without a `trading` long-horizon outcome (delisted/bankrupt/unpriced)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--outcomes", default="data/outcomes.csv")
    parser.add_argument("--out", default="data/midterm_outcomes.csv")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    universe = pd.read_csv(args.universe)
    if args.tickers:
        universe = universe[universe.ticker.isin(args.tickers)]
    outcomes = pd.read_csv(args.outcomes)

    res = pd.DataFrame(compute_rows(universe, outcomes))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(res)} rows)")
    for label in HORIZONS:
        col = f"fwd_{label}_bucket"
        if col in res:
            print(f"  {col}: {res[col].value_counts().to_dict()}  (n={res[col].notna().sum()})")


if __name__ == "__main__":
    main()
