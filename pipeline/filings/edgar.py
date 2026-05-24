"""SEC EDGAR filing fetcher — US-market FilingFetcher implementation.

EDGAR REST API:
  https://data.sec.gov/submissions/CIK{cik}.json  → filing index per company
  https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/...  → filing documents

Rate limits: 10 req/s with a proper User-Agent header. Cache filings to
local disk (data/10k_excerpts/{ticker}/{accession}/) — never re-download.

The fetcher returns plain-text excerpts of the main document
(Item 1, Item 1A, Item 7, Item 7A for 10-K; key sections of DEF 14A)
rather than the full filing — these are the sections the LLM grader needs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pipeline.filings.base import Filing


class EDGARFetcher:
    """US-market FilingFetcher — SEC EDGAR."""

    market: str = "US"

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
    ) -> None:
        # EDGAR requires a real contact in the User-Agent string.
        if "@" not in user_agent:
            raise ValueError(
                "EDGAR User-Agent must include a contact email "
                "(see https://www.sec.gov/os/accessing-edgar-data)"
            )
        self.user_agent = user_agent
        self.cache_dir = cache_dir or Path("data/10k_excerpts")

    def fetch_latest_before(
        self,
        ticker: str,
        cutoff_date: date,
        filing_types: tuple[str, ...] = ("10-K", "DEF 14A"),
    ) -> list[Filing]:
        raise NotImplementedError(
            "EDGARFetcher.fetch_latest_before pending — "
            "see aito-equity-demo-TASK.md → Data Pipeline"
        )
