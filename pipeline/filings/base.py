"""Filing-fetcher protocol — pull a company's regulatory filings for a date.

The qualitative-feature LLM extractor consumes the most recent 10-K (or
equivalent annual report) and DEF 14A (proxy) filed BEFORE the vintage
date. Different markets have different sources:

  US      — SEC EDGAR (10-K, DEF 14A)
  EU/UK   — ESEF-tagged annual reports, mostly via company IR pages
  Nordic  — Nasdaq Nordic disclosure system, IR pages

Implementations conform to this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class Filing:
    """One fetched regulatory filing, normalised to plain text."""

    ticker: str
    filing_type: str  # "10-K", "DEF 14A", "annual_report", ...
    filed_date: date
    period_end: date  # fiscal-period end the filing covers
    language: str  # "EN", "FI", "SV", ...
    text: str  # main-document text (HTML/PDF extracted to plain text)
    source_url: str


class FilingFetcher(Protocol):
    market: str

    def fetch_latest_before(
        self,
        ticker: str,
        cutoff_date: date,
        filing_types: tuple[str, ...] = ("10-K", "DEF 14A"),
    ) -> list[Filing]:
        """Return the most recent filing of each requested type filed before cutoff_date.

        Empty list if nothing found — important for the survivorship-bias
        handling (bankrupt companies stop filing; the caller should handle
        this rather than silently dropping the ticker).
        """
        ...
