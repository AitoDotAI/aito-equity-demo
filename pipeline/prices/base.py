"""Price-source protocol — forward returns and survival tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

TerminalEvent = Literal["trading", "acquired", "merged", "delisted", "bankrupt", "spun_off"]


@dataclass(frozen=True)
class TerminalOutcome:
    """The final state of a position from vintage to today (or terminal event).

    For acquired companies, total_return uses the acquisition price as the
    terminal value. For bankruptcies / liquidations, terminal value = 0
    unless residual is recoverable (rare; document if so).
    """

    ticker: str
    vintage_date: date
    end_date: date  # today, or the terminal-event date if earlier
    window_years: float
    total_return_local: float | None  # percentage in the security's currency
    total_return_usd: float | None  # percentage in USD
    survived_intact: bool  # still trading independently today
    terminal_event: TerminalEvent


class PriceSource(Protocol):
    market: str

    def terminal_outcome(self, ticker: str, vintage_date: date) -> TerminalOutcome:
        ...
