"""S&P 500 historical constituents from Wikipedia.

Approach:
  1. Fetch today's constituent list (table 0 of `List_of_S&P_500_companies`).
  2. Fetch the "Selected changes" table (table 1) — historical adds/removes.
  3. Replay changes backwards from today to `vintage_date` to reconstruct
     the membership at that date.

The replay logic:
  - Start with today's set of tickers.
  - For each row in changes with date STRICTLY AFTER vintage_date:
      - If the row added a ticker after vintage_date → remove it from our set.
      - If the row removed a ticker after vintage_date → add it back (it
        existed at vintage_date, just got removed later).

Caveats documented in ADR 0001:
  - Ticker reuses (rare; same ticker bound to different securities over
    time). The "Selected changes" table rarely disambiguates these.
  - Ticker renames (same security, new ticker symbol). We track the
    *current* ticker; if a company was AAA in 2017 and got renamed to BBB
    in 2020, we'll see BBB in today's list but the 2017 row needs AAA.
    For v1 we accept this limitation; the focal companies (NVDA, SHLD,
    COST, META) don't have renames in the windows we care about.
  - Selected-changes table completeness pre-2010 is uneven. We sanity-
    check the 2017 list against an archived SlickCharts snapshot.

For the qualitative pipeline this granularity is sufficient — the
analysis is about long-horizon outcomes per company-vintage, not about
index-replication precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import ClassVar

import httpx
import pandas as pd

from pipeline.universe.base import UniverseEntry

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_UA = "aito-equity-demo/0.1 (https://github.com/AitoDotAI/aito-equity-demo; antti@aito.ai)"


@dataclass(frozen=True)
class _Change:
    when: date
    added_ticker: str | None
    removed_ticker: str | None


class SP500WikipediaSource:
    """US-market UniverseSource — S&P 500 from Wikipedia."""

    market: ClassVar[str] = "US"
    exchange_default: ClassVar[str] = "NYSE"  # overridden per-ticker when present

    def __init__(self, user_agent: str = DEFAULT_UA, timeout: float = 30.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout

    # ── Public API ────────────────────────────────────────────────

    def list_constituents(self, vintage_date: date) -> list[UniverseEntry]:
        current_df, changes_df = self._fetch_tables()
        today_tickers = self._normalise_constituent_tickers(current_df)
        changes = self._parse_changes(changes_df)
        historical = self._replay_to(today_tickers, changes, vintage_date)
        # Join back to the constituent table for sector/industry where we have it.
        meta_by_ticker = self._constituent_metadata(current_df)
        return [
            self._make_entry(t, vintage_date, meta_by_ticker.get(t))
            for t in sorted(historical)
        ]

    # ── Fetch ────────────────────────────────────────────────────

    def _fetch_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        r = httpx.get(
            WIKI_URL,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
            follow_redirects=True,
        )
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        if len(tables) < 2:
            raise RuntimeError(
                f"Expected at least 2 tables on {WIKI_URL}, got {len(tables)}"
            )
        return tables[0], tables[1]

    # ── Constituents (table 0) ───────────────────────────────────

    @staticmethod
    def _normalise_constituent_tickers(df: pd.DataFrame) -> set[str]:
        col = "Symbol"
        if col not in df.columns:
            raise RuntimeError(f"Constituent table missing '{col}' column: {list(df.columns)}")
        # Wikipedia tickers can have suffixes like "BRK.B" — keep as-is, just strip whitespace.
        return {str(t).strip() for t in df[col].dropna()}

    @staticmethod
    def _constituent_metadata(df: pd.DataFrame) -> dict[str, dict]:
        """Map ticker → {sector, industry, name} from today's table."""
        # Columns vary; tolerate either of the two common header forms.
        sector_col = next((c for c in df.columns if "Sector" in c), None)
        industry_col = next(
            (c for c in df.columns if "Sub-Industry" in c or "Industry" in c), None
        )
        name_col = next((c for c in df.columns if c == "Security"), None)
        out: dict[str, dict] = {}
        for _, row in df.iterrows():
            t = str(row.get("Symbol", "")).strip()
            if not t:
                continue
            out[t] = {
                "sector": str(row[sector_col]).strip() if sector_col else "",
                "industry": str(row[industry_col]).strip() if industry_col else "",
                "company_name": str(row[name_col]).strip() if name_col else t,
            }
        return out

    # ── Changes (table 1) ────────────────────────────────────────

    @classmethod
    def _parse_changes(cls, df: pd.DataFrame) -> list[_Change]:
        """The 'Selected changes' table is multi-header.

        Top-level header groups are typically: Date | Added | Removed | Reason
        Under Added: Ticker, Security. Under Removed: Ticker, Security.

        pandas.read_html flattens multi-header columns into tuples like
        ('Added', 'Ticker'), or into compound strings like 'Added Ticker'.
        We try both.
        """
        date_col = cls._find_col(df, "Date")
        added_col = cls._find_col(df, "Added", subkey="Ticker")
        removed_col = cls._find_col(df, "Removed", subkey="Ticker")
        if not (date_col and added_col and removed_col):
            raise RuntimeError(
                f"Could not locate Date/Added/Removed columns in changes table: {list(df.columns)}"
            )

        out: list[_Change] = []
        for _, row in df.iterrows():
            d = cls._parse_date(row[date_col])
            if d is None:
                continue
            added = cls._clean_ticker(row.get(added_col))
            removed = cls._clean_ticker(row.get(removed_col))
            if not added and not removed:
                continue
            out.append(_Change(when=d, added_ticker=added, removed_ticker=removed))
        return out

    @staticmethod
    def _find_col(df: pd.DataFrame, key: str, subkey: str | None = None) -> object | None:
        """Locate a column whose top-or-flat name contains `key` and optionally subkey."""
        for c in df.columns:
            if isinstance(c, tuple):
                top = " ".join(str(x) for x in c)
                if key in top and (subkey is None or subkey in top):
                    return c
            else:
                s = str(c)
                if key in s and (subkey is None or subkey in s):
                    return c
        return None

    @staticmethod
    def _parse_date(raw: object) -> date | None:
        if pd.isna(raw):
            return None
        s = str(raw).strip()
        from datetime import datetime
        # Wikipedia uses ISO 'YYYY-MM-DD' for "Date added" in current table,
        # and 'Month D, YYYY' in changes. Try both.
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except (ValueError, TypeError):
                continue
        # Last resort: pandas date parsing.
        try:
            return pd.Timestamp(s).date()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _clean_ticker(raw: object) -> str | None:
        if raw is None or pd.isna(raw):
            return None
        s = str(raw).strip()
        if not s or s.lower() == "nan":
            return None
        # Some rows have ticker + footnote markers like "AAA[1]"
        s = re.sub(r"\[.*?\]", "", s).strip()
        return s or None

    # ── Replay ───────────────────────────────────────────────────

    @staticmethod
    def _replay_to(
        today_tickers: set[str],
        changes: list[_Change],
        vintage_date: date,
    ) -> set[str]:
        """Reconstruct constituents at vintage_date by reversing post-vintage changes."""
        membership = set(today_tickers)
        for ch in changes:
            if ch.when <= vintage_date:
                continue  # this change is at-or-before vintage; today's state is correct
            # Change happened after vintage; reverse it.
            if ch.added_ticker and ch.added_ticker in membership:
                membership.discard(ch.added_ticker)
            if ch.removed_ticker:
                membership.add(ch.removed_ticker)
        return membership

    # ── Entry construction ──────────────────────────────────────

    def _make_entry(
        self,
        ticker: str,
        vintage_date: date,
        meta: dict | None,
    ) -> UniverseEntry:
        return UniverseEntry(
            ticker=ticker,
            vintage_year=vintage_date.year,
            vintage_date=vintage_date,
            market=self.market,
            exchange=self.exchange_default,  # refined later via EDGAR if needed
            currency="USD",
            reporting_standard="GAAP",
            filing_language="EN",
        )


# ── CLI ─────────────────────────────────────────────────────────

VINTAGES = [date(2014, 1, 1), date(2017, 1, 1), date(2020, 1, 1)]


def main() -> None:
    """Run for all three vintages and write data/universe.csv."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vintage",
        type=int,
        nargs="*",
        default=[2017],
        help="Vintage year(s) to construct (default: 2017 only)",
    )
    parser.add_argument(
        "--out",
        default="data/universe.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    source = SP500WikipediaSource()
    print(f"→ Fetching {WIKI_URL}")

    rows: list[dict] = []
    metadata = None
    for year in args.vintage:
        vd = date(year, 1, 1)
        print(f"→ Reconstructing constituents at {vd.isoformat()}")
        entries = source.list_constituents(vd)
        if metadata is None:
            # Pull sector/industry/name from the current table (one fetch was enough)
            current_df, _ = source._fetch_tables()
            metadata = source._constituent_metadata(current_df)
        for e in entries:
            meta = metadata.get(e.ticker, {})
            rows.append(
                {
                    "ticker": e.ticker,
                    "vintage_year": e.vintage_year,
                    "vintage_date": e.vintage_date.isoformat(),
                    "market": e.market,
                    "exchange": e.exchange,
                    "currency": e.currency,
                    "reporting_standard": e.reporting_standard,
                    "filing_language": e.filing_language,
                    "company_name": meta.get("company_name", ""),
                    "sector": meta.get("sector", ""),
                    "industry": meta.get("industry", ""),
                }
            )
        print(f"  {year}: {len(entries)} constituents")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"→ {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
