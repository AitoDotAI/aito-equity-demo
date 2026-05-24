"""yfinance-backed PriceSource — works for US large caps and most non-US tickers.

For each (ticker, vintage_date):
  1. Find the closest trading day at-or-after vintage_date and grab the
     adjusted close (yfinance auto_adjust=True handles splits + dividends).
  2. Grab the most recent adjusted close as the end value.
  3. If the ticker has NO price history past some date well before today,
     mark it as `delisted` and use the last-known price as terminal value.

Acquisitions and bankruptcies need a separately-maintained
`data/delistings.csv` because yfinance can't distinguish them from
ordinary delistings. For v1 we record terminal_event="delisted" and
flag rows needing manual annotation; the CSV can be enriched later
(documented in ADR 0001).

Caveats:
  - yfinance quality degrades for delisted / non-US small caps.
  - Currency conversion (when we add non-US) handled in a follow-up:
    yfinance's auto_adjust=True returns prices in the security's
    listing currency; FX series come from yf.Ticker("EURUSD=X") etc.
  - Today's date used as the end date; for fixed reproducibility,
    pass `--as-of YYYY-MM-DD` to lock the snapshot.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from pipeline.prices.base import PriceSource, TerminalOutcome


class YFinancePriceSource:
    market: str

    def __init__(self, market: str = "US") -> None:
        self.market = market

    def terminal_outcome(
        self,
        ticker: str,
        vintage_date: date,
        as_of: date | None = None,
    ) -> TerminalOutcome:
        as_of = as_of or date.today()
        t = yf.Ticker(ticker)

        # Pull a window around the vintage date to handle weekend/holiday gaps.
        start = vintage_date - timedelta(days=7)
        end_window = vintage_date + timedelta(days=14)

        try:
            hist_start = t.history(
                start=start.isoformat(),
                end=end_window.isoformat(),
                auto_adjust=True,
                actions=False,
            )
        except Exception:
            hist_start = pd.DataFrame()

        if hist_start.empty:
            return TerminalOutcome(
                ticker=ticker,
                vintage_date=vintage_date,
                end_date=vintage_date,
                window_years=0.0,
                total_return_local=None,
                total_return_usd=None,
                survived_intact=False,
                terminal_event="delisted",  # may also be no_data — caller annotates
            )

        price_start = float(hist_start.iloc[0]["Close"])
        start_date = hist_start.index[0].date()

        # End price — most recent trading day before as_of.
        try:
            hist_end = t.history(
                start=(as_of - timedelta(days=14)).isoformat(),
                end=(as_of + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                actions=False,
            )
        except Exception:
            hist_end = pd.DataFrame()

        if hist_end.empty:
            # Has history at vintage but no recent — probably delisted between.
            full = t.history(
                start=vintage_date.isoformat(),
                end=as_of.isoformat(),
                auto_adjust=True,
                actions=False,
            )
            if full.empty:
                return TerminalOutcome(
                    ticker=ticker,
                    vintage_date=vintage_date,
                    end_date=start_date,
                    window_years=0.0,
                    total_return_local=0.0,
                    total_return_usd=0.0,
                    survived_intact=False,
                    terminal_event="delisted",
                )
            price_end = float(full.iloc[-1]["Close"])
            end_date = full.index[-1].date()
            terminal_event = "delisted"
            survived = False
        else:
            price_end = float(hist_end.iloc[-1]["Close"])
            end_date = hist_end.index[-1].date()
            terminal_event = "trading"
            survived = True

        total_return = (price_end / price_start - 1) * 100
        window_years = (end_date - start_date).days / 365.25

        # For US-market tickers, local == USD.
        is_us = self.market == "US"
        return TerminalOutcome(
            ticker=ticker,
            vintage_date=vintage_date,
            end_date=end_date,
            window_years=window_years,
            total_return_local=total_return,
            total_return_usd=total_return if is_us else None,  # FX in follow-up
            survived_intact=survived,
            terminal_event=terminal_event,
        )


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    """Read data/universe.csv → fetch outcomes → write data/outcomes.csv."""
    import argparse

    from pipeline.outcomes import bucket_for_return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        default="data/universe.csv",
        help="Input universe CSV (from `./do pipeline universe`)",
    )
    parser.add_argument(
        "--out",
        default="data/outcomes.csv",
        help="Output outcomes CSV",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Snapshot date (YYYY-MM-DD); default: today",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (for fast testing on a subset)",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        help="Seconds to sleep between yfinance calls (default 0)",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    universe_df = pd.read_csv(args.universe)
    if args.limit:
        universe_df = universe_df.head(args.limit)
    print(f"→ {len(universe_df)} (ticker, vintage) rows from {args.universe}")

    source = YFinancePriceSource(market="US")

    rows: list[dict] = []
    failed = 0
    for i, row in enumerate(universe_df.itertuples(index=False), 1):
        if args.throttle > 0:
            time.sleep(args.throttle)
        try:
            outcome = source.terminal_outcome(
                ticker=row.ticker,
                vintage_date=date.fromisoformat(row.vintage_date),
                as_of=as_of,
            )
        except Exception as e:
            print(f"  ✗ {row.ticker} @ {row.vintage_date}: {e}")
            failed += 1
            continue

        bucket = None
        if outcome.total_return_local is not None and outcome.window_years > 0:
            bucket = bucket_for_return(outcome.total_return_local, outcome.window_years)

        rows.append(
            {
                "ticker": row.ticker,
                "vintage_year": row.vintage_year,
                "vintage_date": row.vintage_date,
                "end_date": outcome.end_date.isoformat(),
                "window_years": round(outcome.window_years, 2),
                "total_return_pct_local": outcome.total_return_local,
                "total_return_pct_usd": outcome.total_return_usd,
                "survived_intact": outcome.survived_intact,
                "terminal_event": outcome.terminal_event,
                "outcome_bucket": bucket,
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(universe_df)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(rows)} rows, {failed} failed)")


if __name__ == "__main__":
    main()
