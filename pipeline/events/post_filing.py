"""Post-10-K-filing drift: a short-horizon, tradeable reframing.

The long-horizon views ask "company at vintage → 12-year outcome". This
stage reframes to an EVENT: a 10-K hits the wire, and we measure the
forward 20-trading-day return after it. Same companies, same features —
a horizon a trading desk actually rebalances on.

Honesty note: these are 10-K *filing* dates, not 8-K earnings-announcement
dates (historical earnings dates aren't freely available for 2014-era
filings). The annual report is largely pre-priced by the earlier earnings
8-K, so this is *post-annual-report* drift — weaker than classic PEAD. The
point of the demo is the two-horizon contrast (quality predicts 12 years;
momentum/surprise carry the weak ~1-month signal), not an alpha claim.

Per (ticker, vintage):
  filing_date            latest 10-K filed before the vintage date
  fwd_20d_return         (close[filing+21td] / close[filing+1td] - 1) %
                         — enter the day AFTER the filing, exit 20 td later
  fwd_20d_bucket         sharp_down / down / flat / up / sharp_up
  pre_filing_mom_60d     return over the 60 td BEFORE the filing
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FILINGS_DIR = Path("data/10k_excerpts")
FWD_TD = 20      # forward trading days
MOM_TD = 60      # pre-filing momentum window


def latest_10k_filing_date(ticker: str, vintage_date: date) -> date | None:
    """Most recent 10-K filed strictly before vintage_date (from cache)."""
    tdir = FILINGS_DIR / ticker
    if not tdir.exists():
        return None
    best: date | None = None
    for acc in tdir.iterdir():
        meta_path = acc / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("filing_type") != "10-K":
            continue
        try:
            filed = date.fromisoformat(meta["filed_date"])
        except (KeyError, ValueError):
            continue
        if filed < vintage_date and (best is None or filed > best):
            best = filed
    return best


def bucket_20d(ret_pct: float) -> str:
    if ret_pct <= -10:
        return "sharp_down"
    if ret_pct <= -3:
        return "down"
    if ret_pct < 3:
        return "flat"
    if ret_pct < 10:
        return "up"
    return "sharp_up"


def compute_events_for_ticker(ticker: str, filings: list[tuple[int, date, date]]) -> list[dict]:
    """filings: list of (vintage_year, vintage_date, filing_date)."""
    if not filings:
        return []
    earliest = min(f[2] for f in filings) - timedelta(days=160)
    latest = max(f[2] for f in filings) + timedelta(days=90)
    try:
        h = yf.Ticker(ticker).history(
            start=earliest.isoformat(), end=latest.isoformat(), auto_adjust=True, actions=False
        )
    except Exception:
        return []
    if h.empty or "Close" not in h:
        return []
    closes = h["Close"].dropna()
    idx = closes.index.tz_localize(None) if closes.index.tz is not None else closes.index
    closes = pd.Series(closes.values, index=idx.normalize())

    out = []
    for vy, vd, fd in filings:
        fdt = pd.Timestamp(fd)
        # trading days at/after the filing date
        after = closes[closes.index >= fdt]
        before = closes[closes.index < fdt]
        row = {
            "ticker": ticker,
            "vintage_year": vy,
            "vintage_date": vd.isoformat(),
            "filing_date": fd.isoformat(),
            "fwd_20d_return": None,
            "fwd_20d_bucket": None,
            "pre_filing_mom_60d": None,
        }
        if len(after) >= FWD_TD + 1:
            entry = float(after.iloc[1])           # day after filing
            exit_ = float(after.iloc[1 + FWD_TD])  # +20 td
            if entry > 0:
                ret = (exit_ / entry - 1) * 100
                row["fwd_20d_return"] = round(ret, 2)
                row["fwd_20d_bucket"] = bucket_20d(ret)
        if len(before) >= MOM_TD:
            p0 = float(before.iloc[-MOM_TD])
            p1 = float(before.iloc[-1])
            if p0 > 0:
                row["pre_filing_mom_60d"] = round((p1 / p0 - 1) * 100, 2)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default="data/filing_events.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.universe)
    if args.tickers:
        df = df[df["ticker"].isin(args.tickers)]
    elif args.limit:
        df = df.head(args.limit)

    # Group filings by ticker so we fetch each price history once.
    by_ticker: dict[str, list[tuple[int, date, date]]] = {}
    n_nofiling = 0
    for row in df.itertuples(index=False):
        vd = date.fromisoformat(row.vintage_date)
        fd = latest_10k_filing_date(row.ticker, vd)
        if fd is None:
            n_nofiling += 1
            continue
        by_ticker.setdefault(row.ticker, []).append((int(row.vintage_year), vd, fd))

    print(f"→ Post-filing drift for {sum(len(v) for v in by_ticker.values())} events "
          f"across {len(by_ticker)} tickers ({n_nofiling} rows had no cached 10-K)")

    rows: list[dict] = []
    for i, (ticker, filings) in enumerate(by_ticker.items(), 1):
        rows.extend(compute_events_for_ticker(ticker, filings))
        if i % 50 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  {i}/{len(by_ticker)} tickers")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    n_ret = res["fwd_20d_return"].notna().sum() if len(res) else 0
    print(f"→ {args.out} ({len(res)} events · {n_ret} with forward return)")
    if n_ret:
        print("  fwd_20d_bucket:", res["fwd_20d_bucket"].value_counts().to_dict())


if __name__ == "__main__":
    main()
