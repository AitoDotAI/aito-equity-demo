"""yfinance-backed PriceSource — works for US large caps and most non-US tickers.

Caveats:
  - Quality degrades for delisted / non-US small caps
  - Splits and currency conversion handled by yfinance auto_adjust=True
  - For delisted tickers, fall back to a manually-maintained delistings table
    (data/delistings.csv) for terminal event + date
"""

from __future__ import annotations

from datetime import date

from pipeline.prices.base import PriceSource, TerminalOutcome


class YFinancePriceSource:
    market: str

    def __init__(self, market: str = "US") -> None:
        self.market = market

    def terminal_outcome(self, ticker: str, vintage_date: date) -> TerminalOutcome:
        raise NotImplementedError(
            "YFinancePriceSource.terminal_outcome pending — "
            "see aito-equity-demo-TASK.md → Data Pipeline → outcomes"
        )
