"""Fetch the press-release exhibit from earnings 8-Ks (Item 2.02).

The 8-K cover document is boilerplate; the actual earnings news lives in
an exhibit (usually EX-99.1, the press release, sometimes plus CFO
commentary). The filing index.json doesn't carry document types, so we
select the content exhibits by filename heuristic — the .htm files that
are NOT the cover primary document and NOT XBRL viewer / summary files.

Output: data/earnings_8k/{ticker}/{accession}/release.txt (+ meta.json)
        data/earnings_events.csv  (ticker, date, accession, release_chars)

Free, EDGAR rate-limited. Use --tickers / --limit to bound the run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

INDEX_DIR = Path("data/10k_excerpts/_index")
TICKER_INDEX = INDEX_DIR / "company_tickers.json"
CACHE_DIR = Path("data/earnings_8k")
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
USER_AGENT = "Aito Equity Demo antti@aito.ai"
THROTTLE = 0.15

# Files to skip when picking the press-release exhibit.
SKIP_RE = re.compile(r"(^R\d+\.htm$)|(_def|_lab|_pre|_cal)|(FilingSummary|MetaLinks)|(-index)", re.I)


def load_cik_index() -> dict[str, int]:
    idx = json.loads(TICKER_INDEX.read_text(encoding="utf-8"))
    return {v["ticker"]: int(v["cik_str"]) for v in idx.values()}


def earnings_filings(cik: int) -> list[tuple[str, str, str]]:
    """(date, accession, primary_doc) for every 2.02 earnings 8-K."""
    path = INDEX_DIR / f"submissions_{cik:010d}.json"
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    r = d.get("filings", {}).get("recent", {})
    forms, dates, items = r.get("form", []), r.get("filingDate", []), r.get("items", [])
    accs, pdocs = r.get("accessionNumber", []), r.get("primaryDocument", [])
    out = []
    for i, f in enumerate(forms):
        if f == "8-K" and "2.02" in (items[i] if i < len(items) else ""):
            out.append((dates[i], accs[i], pdocs[i] if i < len(pdocs) else ""))
    return out


class _Session:
    def __init__(self) -> None:
        self.c = httpx.Client(headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}, timeout=30.0)
        self.last = 0.0

    def get(self, url: str) -> httpx.Response | None:
        dt = time.monotonic() - self.last
        if dt < THROTTLE:
            time.sleep(THROTTLE - dt)
        try:
            r = self.c.get(url, follow_redirects=True)
            self.last = time.monotonic()
            return r if r.status_code == 200 else None
        except httpx.HTTPError:
            return None


def pick_release_docs(file_names: list[str], primary_doc: str) -> list[str]:
    """Content exhibits likely to hold the press release / commentary."""
    out = []
    for name in file_names:
        low = name.lower()
        if not low.endswith((".htm", ".html", ".txt")):
            continue
        if name == primary_doc:        # the 8-K cover — boilerplate
            continue
        if SKIP_RE.search(name):       # XBRL viewer / summary / index
            continue
        out.append(name)
    # Prefer obvious press-release names first.
    def rank(n: str) -> int:
        l = n.lower()
        if any(k in l for k in ("pr.htm", "press", "ex99", "ex-99", "earnings", "release", "results")):
            return 0
        if "commentary" in l or "cfo" in l:
            return 1
        return 2
    return sorted(out, key=rank)[:3]   # press release + maybe commentary


def fetch_release(session: _Session, cik: int, accession: str, primary_doc: str) -> str | None:
    acc_nodash = accession.replace("-", "")
    base = ARCHIVES.format(cik=cik, acc=acc_nodash)
    idx = session.get(base + "index.json")
    if idx is None:
        return None
    try:
        items = idx.json()["directory"]["item"]
    except Exception:
        return None
    names = [it["name"] for it in items]
    docs = pick_release_docs(names, primary_doc)
    if not docs:
        return None
    texts = []
    for doc in docs:
        r = session.get(base + doc)
        if r is None:
            continue
        soup = BeautifulSoup(r.content, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        t = soup.get_text(separator="\n")
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"[ \t]+", " ", t)
        texts.append(t)
    full = "\n\n".join(texts).strip()
    return full or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default="data/earnings_events.csv")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=None, help="Process at most N tickers")
    parser.add_argument("--since", default="2015-01-01", help="Only events on/after this date")
    args = parser.parse_args()

    uni = pd.read_csv(args.universe)
    tickers = sorted(uni["ticker"].unique())
    if args.tickers:
        tickers = [t for t in tickers if t in args.tickers]
    elif args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    since = date.fromisoformat(args.since)

    cik_index = load_cik_index()
    session = _Session()
    print(f"→ Earnings 8-K exhibits for {len(tickers)} tickers (since {since})")

    rows: list[dict] = []
    n_ok = n_skip = 0
    for ti, ticker in enumerate(tickers, 1):
        cik = cik_index.get(ticker)
        if cik is None:
            continue
        for dt, acc, pdoc in earnings_filings(cik):
            try:
                if date.fromisoformat(dt) < since:
                    continue
            except ValueError:
                continue
            acc_nodash = acc.replace("-", "")
            cache_sub = CACHE_DIR / ticker / acc_nodash
            rel_path = cache_sub / "release.txt"
            if rel_path.exists():
                text = rel_path.read_text(encoding="utf-8", errors="replace")
            else:
                text = fetch_release(session, cik, acc, pdoc)
                if text:
                    cache_sub.mkdir(parents=True, exist_ok=True)
                    rel_path.write_text(text, encoding="utf-8")
                    (cache_sub / "meta.json").write_text(json.dumps({"ticker": ticker, "date": dt, "accession": acc}), encoding="utf-8")
            if text:
                n_ok += 1
                rows.append({"ticker": ticker, "date": dt, "accession": acc, "release_chars": len(text)})
            else:
                n_skip += 1
        if ti % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  {ti}/{len(tickers)} tickers · {n_ok} releases ok, {n_skip} missing")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"→ {args.out} ({n_ok} releases, {n_skip} missing)")


if __name__ == "__main__":
    main()
