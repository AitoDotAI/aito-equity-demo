"""Universe construction protocol — point-in-time index constituents.

A UniverseSource returns the (ticker, vintage) pairs that should be graded.
v1 ships SP500WikipediaSource (US-only). Future markets implement
OMXHelsinkiSource, STOXX600Source, etc., conforming to the same protocol.

Survivorship is eliminated by reconstructing membership AT the vintage date,
not from today's index composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class UniverseEntry:
    """One (ticker, vintage) row in the universe.

    Tickers are namespaced per Yahoo convention so they stay unique across
    markets (e.g. "NOKIA.HE" for Helsinki listings, "VOLV-B.ST" for Stockholm,
    plain "NVDA" for US).
    """

    ticker: str
    vintage_year: int
    vintage_date: date
    market: str  # "US", "FI", "SE", "NO", "DK", "EU"
    exchange: str  # "NASDAQ", "NYSE", "HEL", "STO", ...
    currency: str  # "USD", "EUR", "SEK", "NOK", "DKK"
    reporting_standard: str  # "GAAP", "IFRS"
    filing_language: str  # "EN", "FI", "SV", ...


class UniverseSource(Protocol):
    """Return historical index constituents as of a vintage date."""

    market: str

    def list_constituents(self, vintage_date: date) -> list[UniverseEntry]:
        """Return the index membership as of vintage_date.

        Implementations must NOT use today's index composition — they must
        reconstruct membership at the vintage date (from edit history,
        archived snapshots, or a maintained historical constituent file).
        """
        ...
