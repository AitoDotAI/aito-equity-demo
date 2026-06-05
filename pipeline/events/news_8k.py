"""8-K news-reaction event study (the real trading-relevant demo).

An 8-K *is* corporate news: a material event, filed with the SEC at a
point in time, free, and tagged with Item codes that ARE high-level
themes (2.02 = earnings, 5.02 = management change, 2.01 = M&A, ...). So
we get a timestamped, themed news feed for free — exactly what a reaction
model needs, and the item codes give the sample sizes for credibility.

Per 8-K event:
  date            filing date
  theme           high-level theme from the item code(s)
  items           raw item codes
  react_1d        close[+1] / anchor - 1   (anchor = close the day before)
  react_5d        close[+5] / anchor - 1   (~1 week)
  react_20d       close[+20] / anchor - 1  (~1 month)
  react_1d_bucket / 5d / 20d   up/flat/down bands

The anchor is the close the trading day BEFORE the filing, so the day-of
reaction is captured. Source: cached EDGAR submissions (recent window,
~2020+ for active filers) + yfinance prices.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

INDEX_DIR = Path("data/10k_excerpts/_index")
TICKER_INDEX = INDEX_DIR / "company_tickers.json"

# Item code → high-level theme. First matching non-exhibit item wins.
ITEM_THEME = {
    "2.02": "earnings",
    "2.01": "M&A",
    "1.01": "material agreement",
    "1.02": "material agreement",
    "5.02": "management change",
    "7.01": "guidance / RegFD",
    "8.01": "other material news",
    "2.05": "restructuring",
    "2.06": "restructuring",
    "4.01": "auditor change",
    "4.02": "restatement",
    "3.01": "listing / compliance",
    "5.07": "shareholder vote",
    "1.03": "bankruptcy",
    "2.03": "new debt / financing",
    "2.04": "debt trigger",
    "3.02": "equity issuance",
}
# Items that are administrative/exhibit-only — never the primary theme.
SKIP_ITEMS = {"9.01", "5.03", "5.05", "8.03"}


def load_cik_index() -> dict[str, int]:
    idx = json.loads(TICKER_INDEX.read_text(encoding="utf-8"))
    return {v["ticker"]: int(v["cik_str"]) for v in idx.values()}


def primary_theme(items_str: str) -> tuple[str, str]:
    """Return (theme, cleaned_items). items_str like '2.02,9.01'."""
    items = [i.strip() for i in str(items_str).split(",") if i.strip()]
    for it in items:
        if it in SKIP_ITEMS:
            continue
        if it in ITEM_THEME:
            return ITEM_THEME[it], ",".join(items)
    # Only exhibit/admin items, or unknown
    for it in items:
        if it not in SKIP_ITEMS:
            return "other", ",".join(items)
    return "exhibits only", ",".join(items)


def eightk_events(cik: int) -> list[dict]:
    """All 8-K (date, items) from a company's cached submissions."""
    path = INDEX_DIR / f"submissions_{cik:010d}.json"
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    recent = d.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items = recent.get("items", [])
    out = []
    for i, f in enumerate(forms):
        if f != "8-K":
            continue
        it = items[i] if i < len(items) else ""
        theme, cleaned = primary_theme(it)
        if theme in ("exhibits only",):
            continue
        out.append({"date": dates[i], "items": cleaned, "theme": theme})
    return out


def bucket_react(r: float) -> str:
    if r <= -3:
        return "down"
    if r < 3:
        return "flat"
    return "up"


def compute_reactions(ticker: str, events: list[dict]) -> list[dict]:
    if not events:
        return []
    ds = sorted(date.fromisoformat(e["date"]) for e in events)
    start = ds[0] - timedelta(days=15)
    end = ds[-1] + timedelta(days=45)
    try:
        h = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True, actions=False)
    except Exception:
        return []
    if h.empty or "Close" not in h:
        return []
    closes = h["Close"].dropna()
    idx = (closes.index.tz_localize(None) if closes.index.tz is not None else closes.index).normalize()
    closes = pd.Series(closes.values, index=idx)

    out = []
    for e in events:
        ed = pd.Timestamp(date.fromisoformat(e["date"]))
        before = closes[closes.index < ed]
        after = closes[closes.index >= ed]
        if len(before) < 1 or len(after) < 1:
            continue
        anchor = float(before.iloc[-1])
        if anchor <= 0:
            continue
        row = {
            "ticker": ticker, "date": e["date"], "theme": e["theme"], "items": e["items"],
            "react_1d": None, "react_5d": None, "react_20d": None,
        }
        # k trading days after the event (after.iloc[k-1] is the k-th close at/after event)
        for label, k in [("react_1d", 1), ("react_5d", 5), ("react_20d", 20)]:
            if len(after) > k:
                row[label] = round((float(after.iloc[k]) / anchor - 1) * 100, 2)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default="data/news_events.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    uni = pd.read_csv(args.universe)
    tickers = sorted(uni["ticker"].unique())
    if args.tickers:
        tickers = [t for t in tickers if t in args.tickers]
    elif args.limit:
        tickers = tickers[: args.limit]

    cik_index = load_cik_index()
    print(f"→ 8-K news events for {len(tickers)} tickers")

    rows: list[dict] = []
    n_tickers_with_events = 0
    for i, ticker in enumerate(tickers, 1):
        cik = cik_index.get(ticker)
        if cik is None:
            continue
        events = eightk_events(cik)
        if not events:
            continue
        reacts = compute_reactions(ticker, events)
        if reacts:
            n_tickers_with_events += 1
            rows.extend(reacts)
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  {i}/{len(tickers)} tickers · {len(rows)} events")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(res)} events from {n_tickers_with_events} tickers)")
    if len(res):
        print("  themes:", res["theme"].value_counts().to_dict())


if __name__ == "__main__":
    main()
