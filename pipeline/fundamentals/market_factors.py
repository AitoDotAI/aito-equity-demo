"""Price-derived market factors at vintage, from yfinance (free, point-in-time).

For each (ticker, vintage) we use ONLY price history up to the vintage date:
  momentum_12m   — trailing 12-month total return before vintage
  volatility_12m — annualised stdev of daily returns, trailing 12 months
  vintage_price  — adjusted close at/just-before vintage
  market_cap     — vintage_price × shares_outstanding (shares from XBRL stage)

Combined with the XBRL fundamentals (net_income, equity), the merge step
derives the valuation factors:
  pe_ratio       = market_cap / net_income
  pb_ratio       = market_cap / stockholders_equity
  earnings_yield = net_income / market_cap   (1 / PE; the "cheapness" factor)

All continuous factors are also bucketed into interpretable bands for the
factor explorer (e.g. momentum strong/neutral/weak; valuation cheap/fair/rich).
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def _trailing_window(ticker: str, vintage: date) -> pd.DataFrame:
    """~13 months of daily history ending at the vintage date."""
    start = vintage - timedelta(days=400)
    end = vintage + timedelta(days=2)
    t = yf.Ticker(ticker)
    try:
        h = t.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True, actions=False)
    except Exception:
        return pd.DataFrame()
    return h


def market_factors(ticker: str, vintage: date) -> dict:
    h = _trailing_window(ticker, vintage)
    out: dict = {
        "vintage_price": None,
        "momentum_12m": None,
        "volatility_12m": None,
    }
    if h.empty or "Close" not in h:
        return out
    closes = h["Close"].dropna()
    if len(closes) < 30:
        return out
    price_end = float(closes.iloc[-1])
    out["vintage_price"] = round(price_end, 4)

    # 12-month momentum: price now vs ~252 trading days ago (or earliest).
    lookback = closes.iloc[-252] if len(closes) >= 252 else closes.iloc[0]
    out["momentum_12m"] = round((price_end / float(lookback) - 1) * 100, 2)

    # Annualised volatility of daily log returns over the window.
    rets = np.log(closes / closes.shift(1)).dropna()
    if len(rets) > 20:
        out["volatility_12m"] = round(float(rets.std()) * np.sqrt(252) * 100, 2)
    return out


# ── Bucketers (interpretable bands for the factor explorer) ──────


def bucket_momentum(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    if v >= 30:
        return "strong"
    if v <= -10:
        return "weak"
    return "neutral"


def bucket_volatility(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    if v >= 40:
        return "high"
    if v <= 22:
        return "low"
    return "medium"


def bucket_pe(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    if v < 0:
        return "negative_earnings"
    if v < 15:
        return "cheap"
    if v <= 30:
        return "fair"
    return "expensive"


def bucket_growth(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    if v >= 0.15:
        return "high"
    if v <= 0.0:
        return "shrinking"
    return "moderate"


def bucket_leverage(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    if v < 0:
        return "negative_equity"
    if v < 0.5:
        return "low"
    if v <= 1.5:
        return "moderate"
    return "high"


def bucket_roe(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    if v < 0:
        return "negative"
    if v >= 0.20:
        return "high"
    if v >= 0.10:
        return "solid"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default="data/market_factors.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.universe)
    if args.tickers:
        df = df[df["ticker"].isin(args.tickers)]
    elif args.limit:
        df = df.head(args.limit)

    print(f"→ Market factors for {len(df)} (ticker, vintage) rows")
    rows: list[dict] = []
    n_ok = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        mf = market_factors(row.ticker, date.fromisoformat(row.vintage_date))
        mf.update({"ticker": row.ticker, "vintage_year": row.vintage_year, "vintage_date": row.vintage_date})
        rows.append(mf)
        if mf["vintage_price"] is not None:
            n_ok += 1
        if i % 50 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  {i}/{len(df)}  ok={n_ok}")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(rows)} rows · {n_ok} with price)")


if __name__ == "__main__":
    main()
