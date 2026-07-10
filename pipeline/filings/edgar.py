"""SEC EDGAR filing fetcher — US-market FilingFetcher implementation.

EDGAR endpoints:
  https://www.sec.gov/files/company_tickers.json
    → ticker → CIK + company name (one bulk fetch, cached locally)
  https://data.sec.gov/submissions/CIK{cik:010d}.json
    → filings list for one company (covers most-recent 1000 filings)
  https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}
    → actual filing document

Rate limits: 10 req/s with a real User-Agent (must include email). The
fetcher self-throttles and caches every download under
`data/10k_excerpts/{ticker}/{accession}/` — re-runs are free.

What we save per filing:
  - raw.html    (or .htm) — the original primary document
  - text.txt    — BeautifulSoup .get_text() (rough plain text)
  - meta.json   — accession, filed_date, period_end, source_url

Section extraction (Item 1, 1A, 7, 7A for 10-K; key sections of DEF 14A)
is the extraction stage's job — see pipeline/extraction/extract.py. This
fetcher's responsibility ends at "the filing is on disk as plain text".
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from pipeline.filings.base import Filing

CACHE_DIR_DEFAULT = Path("data/10k_excerpts")
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{file}"

# Delisted tickers drop out of SEC's current company_tickers.json, so the
# ticker→CIK index can't resolve them. These are companies that went through
# Chapter 11 (common cancelled) but were live and filing at their vintage —
# we still want to grade their pre-bankruptcy 10-K. CIKs verified against
# data.sec.gov/submissions (the entity that filed the relevant 10-K).
CIK_OVERRIDES: dict[str, int] = {
    "FTR": 20520,    # Frontier Communications
    "CHK": 895126,   # Chesapeake Energy (now Expand Energy)
    "WIN": 1282266,  # Windstream Holdings
    "DNR": 945764,   # Denbury Resources
    "DO": 949039,    # Diamond Offshore Drilling
    "NE": 1458891,   # Noble Corp (drilling — not Noble Energy)
    "ESV": 314808,   # Ensco (now Valaris)
    "RDC": 85408,    # Rowan Companies plc
    "BTU": 1064728,  # Peabody Energy
}

# SEC's documented limit is 10 req/s. We stay well under to be polite and
# avoid throttling under burst.
THROTTLE_SECONDS = 0.15


@dataclass
class _EdgarSession:
    client: httpx.Client
    last_call: float = 0.0

    def get(self, url: str) -> httpx.Response:
        elapsed = time.monotonic() - self.last_call
        if elapsed < THROTTLE_SECONDS:
            time.sleep(THROTTLE_SECONDS - elapsed)
        r = self.client.get(url, timeout=30.0, follow_redirects=True)
        self.last_call = time.monotonic()
        r.raise_for_status()
        return r


class EDGARFetcher:
    """US-market FilingFetcher — SEC EDGAR."""

    market: str = "US"

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError(
                "EDGAR User-Agent must include a contact email "
                "(see https://www.sec.gov/os/accessing-edgar-data)"
            )
        self.user_agent = user_agent
        self.cache_dir = (cache_dir or CACHE_DIR_DEFAULT).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ticker_to_cik: dict[str, tuple[int, str]] | None = None

    # ── Public API ───────────────────────────────────────────────

    def fetch_latest_before(
        self,
        ticker: str,
        cutoff_date: date,
        filing_types: tuple[str, ...] = ("10-K", "DEF 14A"),
    ) -> list[Filing]:
        cik, name = self._cik_for(ticker)
        if cik is None:
            return []

        submissions = self._submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return []

        zipped = list(
            zip(
                recent.get("form", []),
                recent.get("filingDate", []),
                recent.get("reportDate", []),
                recent.get("accessionNumber", []),
                recent.get("primaryDocument", []),
            )
        )

        out: list[Filing] = []
        for form_type in filing_types:
            best = self._latest_before(zipped, form_type, cutoff_date)
            if best is None:
                continue
            form, filed, period_end, accession, primary_doc = best
            filing = self._materialise(
                ticker=ticker,
                cik=cik,
                accession=accession,
                primary_doc=primary_doc,
                filing_type=form,
                filed_date=date.fromisoformat(filed),
                period_end=date.fromisoformat(period_end) if period_end else date.fromisoformat(filed),
            )
            if filing is not None:
                out.append(filing)
        return out

    # ── CIK lookup ───────────────────────────────────────────────

    def _cik_for(self, ticker: str) -> tuple[int | None, str]:
        # Overrides win: these tickers were either dropped from the current
        # index (delisted) or REUSED by a different company (e.g. NE → the
        # 2022 re-listed Noble plc), so the live index would resolve the wrong
        # entity. Our vintages predate every reuse, so the override is correct.
        if ticker in CIK_OVERRIDES:
            return CIK_OVERRIDES[ticker], ""
        if self._ticker_to_cik is None:
            self._ticker_to_cik = self._load_ticker_index()
        if ticker in self._ticker_to_cik:
            cik, name = self._ticker_to_cik[ticker]
            return cik, name
        return None, ""

    def _load_ticker_index(self) -> dict[str, tuple[int, str]]:
        cache_path = self.cache_dir / "_index" / "company_tickers.json"
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            session = self._session()
            r = session.get(TICKER_CIK_URL)
            cache_path.write_bytes(r.content)
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        out: dict[str, tuple[int, str]] = {}
        for v in data.values():
            t = str(v.get("ticker", "")).strip().upper()
            if t:
                out[t] = (int(v["cik_str"]), str(v.get("title", "")))
        return out

    # ── Submissions ──────────────────────────────────────────────

    def _submissions(self, cik: int) -> dict:
        """Return submissions data with `filings.recent` augmented by any
        older `filings.files` (EDGAR splits long histories into additional
        JSON files; we merge them all so the cutoff filter sees pre-2020
        filings for active companies like NVDA, META, etc.)."""
        cache_path = self.cache_dir / "_index" / f"submissions_{cik:010d}.json"
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            session = self._session()
            r = session.get(SUBMISSIONS_URL.format(cik=cik))
            cache_path.write_bytes(r.content)
        data = json.loads(cache_path.read_text(encoding="utf-8"))

        additional = data.get("filings", {}).get("files", []) or []
        if not additional:
            return data

        session = self._session()
        recent = data["filings"]["recent"]
        # The recent dict has parallel arrays keyed by field name.
        keys = list(recent.keys())
        for f in additional:
            extra_cache = self.cache_dir / "_index" / f["name"]
            if not extra_cache.exists():
                url = f"https://data.sec.gov/submissions/{f['name']}"
                r = session.get(url)
                extra_cache.write_bytes(r.content)
            extra = json.loads(extra_cache.read_text(encoding="utf-8"))
            for k in keys:
                recent[k].extend(extra.get(k, []))
        return data

    @staticmethod
    def _latest_before(
        zipped: list[tuple[str, str, str, str, str]],
        form_type: str,
        cutoff: date,
    ) -> tuple[str, str, str, str, str] | None:
        best: tuple[str, str, str, str, str] | None = None
        for entry in zipped:
            form, filed, period_end, accession, primary_doc = entry
            if form != form_type:
                continue
            try:
                fd = date.fromisoformat(filed)
            except ValueError:
                continue
            if fd >= cutoff:
                continue
            if best is None or fd > date.fromisoformat(best[1]):
                best = entry
        return best

    # ── Materialise (download + cache) ───────────────────────────

    def _materialise(
        self,
        ticker: str,
        cik: int,
        accession: str,
        primary_doc: str,
        filing_type: str,
        filed_date: date,
        period_end: date,
    ) -> Filing | None:
        accession_no_dashes = accession.replace("-", "")
        cache_subdir = self.cache_dir / ticker / accession_no_dashes
        cache_subdir.mkdir(parents=True, exist_ok=True)

        text_path = cache_subdir / "text.txt"
        meta_path = cache_subdir / "meta.json"

        if text_path.exists() and meta_path.exists():
            text = text_path.read_text(encoding="utf-8", errors="replace")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return Filing(
                ticker=ticker,
                filing_type=filing_type,
                filed_date=date.fromisoformat(meta["filed_date"]),
                period_end=date.fromisoformat(meta["period_end"]),
                language="EN",
                text=text,
                source_url=meta["source_url"],
            )

        source_url = ARCHIVES_URL.format(
            cik=cik,
            accession=accession_no_dashes,
            file=primary_doc,
        )
        session = self._session()
        try:
            r = session.get(source_url)
        except httpx.HTTPError as e:
            print(f"  ✗ {ticker} {filing_type} {accession}: {e}")
            return None

        raw_path = cache_subdir / "raw.html"
        raw_path.write_bytes(r.content)

        # Plain-text extraction. Strip script/style; collapse whitespace.
        soup = BeautifulSoup(r.content, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text_path.write_text(text, encoding="utf-8")

        meta = {
            "ticker": ticker,
            "filing_type": filing_type,
            "filed_date": filed_date.isoformat(),
            "period_end": period_end.isoformat(),
            "source_url": source_url,
            "fetched_at": datetime.utcnow().isoformat(),
            "raw_bytes": len(r.content),
            "text_chars": len(text),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return Filing(
            ticker=ticker,
            filing_type=filing_type,
            filed_date=filed_date,
            period_end=period_end,
            language="EN",
            text=text,
            source_url=source_url,
        )

    # ── HTTP session ─────────────────────────────────────────────

    def _session(self) -> _EdgarSession:
        if not hasattr(self, "_sess") or self._sess is None:
            client = httpx.Client(
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            )
            self._sess = _EdgarSession(client=client)
        return self._sess


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    """Run EDGAR fetch for the universe (or a subset) and cache filings."""
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        default="data/universe.csv",
        help="Input universe CSV",
    )
    parser.add_argument(
        "--user-agent",
        default="Aito Equity Demo antti@aito.ai",
        help="EDGAR User-Agent (must include contact email)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (for fast testing)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Only fetch these tickers (overrides --limit)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.universe)
    if args.tickers:
        df = df[df["ticker"].isin(args.tickers)]
    elif args.limit:
        df = df.head(args.limit)

    fetcher = EDGARFetcher(user_agent=args.user_agent)
    print(f"→ Fetching filings for {len(df)} (ticker, vintage) rows")

    success = 0
    failed = 0
    no_filings = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        try:
            filings = fetcher.fetch_latest_before(
                ticker=row.ticker,
                cutoff_date=date.fromisoformat(row.vintage_date),
                filing_types=("10-K", "DEF 14A"),
            )
        except Exception as e:
            print(f"  ✗ {row.ticker}: {e}")
            failed += 1
            continue
        if not filings:
            no_filings += 1
            continue
        success += 1
        if i % 25 == 0:
            print(f"  {i}/{len(df)}  ok={success} no_filings={no_filings} failed={failed}")

    print(f"→ Done. ok={success} no_filings={no_filings} failed={failed}")


if __name__ == "__main__":
    main()
