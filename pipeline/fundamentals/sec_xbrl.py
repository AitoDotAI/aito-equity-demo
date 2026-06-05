"""Point-in-time fundamentals from SEC EDGAR's XBRL companyfacts API.

Free, no key, natively point-in-time: every reported financial line carries
the date it was `filed`, so we can pick the latest value KNOWN as of each
vintage date and never leak future restatements.

  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

For each (ticker, vintage) we take the most recent annual (10-K / FY) value
filed strictly before the vintage date, for each concept, then derive ratios.
Coverage is intentionally spotty-tolerant — Aito columns are nullable, so a
company missing a concept just yields a null for that factor.

Derived factors (none require price — pure fundamentals):
  gross_margin, operating_margin, net_margin   (profitability)
  debt_to_equity                               (leverage)
  return_on_equity, return_on_assets           (quality)
  revenue_cagr_3y                              (growth)

Raw values carried for downstream price-based ratios (see market_factors.py):
  net_income, stockholders_equity, shares_outstanding, revenue
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CACHE_DIR = Path("data/fundamentals_cache")
TICKER_INDEX = Path("data/10k_excerpts/_index/company_tickers.json")
USER_AGENT = "Aito Equity Demo antti@aito.ai"
THROTTLE = 0.15  # SEC asks <= 10 req/s

# Concept fallbacks — XBRL tags drift across filers/years; try in order.
CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "shares": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
}


def _session() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}, timeout=30.0)


def load_cik_index() -> dict[str, int]:
    if not TICKER_INDEX.exists():
        raise RuntimeError(f"{TICKER_INDEX} missing — run `./do pipeline filings` first")
    idx = json.loads(TICKER_INDEX.read_text(encoding="utf-8"))
    return {v["ticker"]: int(v["cik_str"]) for v in idx.values()}


def fetch_companyfacts(cik: int, client: httpx.Client) -> dict | None:
    cache = CACHE_DIR / f"CIK{cik:010d}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = client.get(COMPANYFACTS_URL.format(cik=cik))
        time.sleep(THROTTLE)
        if r.status_code != 200:
            return None
        cache.write_bytes(r.content)
        return r.json()
    except Exception:
        return None


def _is_full_year(p: dict) -> bool:
    """True if a flow point spans ~one fiscal year (filters transition/partial
    periods that XBRL still tags 'FY')."""
    start, end = p.get("start"), p.get("end")
    if not start or not end:
        return True  # instantaneous (balance-sheet) point — no duration to check
    try:
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return True
    return 350 <= days <= 380


def _concept_points(facts: dict, key: str, full_year_only: bool = True) -> list[dict]:
    """Annual (10-K/FY) USD/shares points for one concept tag, or []."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    if key not in gaap:
        return []
    units = gaap[key].get("units", {})
    usd = units.get("USD") or units.get("shares") or []
    pts = [p for p in usd if p.get("form") == "10-K" and p.get("fp") == "FY"]
    if full_year_only:
        pts = [p for p in pts if _is_full_year(p)]
    return pts


def _latest_before(points: list[dict], cutoff: date) -> dict | None:
    """Most recent annual value (by period end) that was FILED before cutoff."""
    eligible = []
    for p in points:
        filed = p.get("filed")
        end = p.get("end")
        if not filed or not end:
            continue
        try:
            if date.fromisoformat(filed) < cutoff:
                eligible.append(p)
        except ValueError:
            continue
    if not eligible:
        return None
    # Prefer the latest period END (most recent fiscal year known at cutoff).
    return max(eligible, key=lambda p: p["end"])


def _value_at(facts: dict, concept_keys: list[str], cutoff: date) -> float | None:
    """First concept (in fallback order) that yields a pre-vintage value.

    Tag usage drifts over time (e.g. RevenueFromContract... only exists post
    ASC 606 / 2018), so we must try each concept and take the first that
    actually has a usable point — not the first that merely exists.
    """
    for key in concept_keys:
        hit = _latest_before(_concept_points(facts, key), cutoff)
        if hit and hit.get("val") is not None:
            return float(hit["val"])
    return None


def _revenue_points(facts: dict) -> list[dict]:
    """All annual revenue points, merged across concept tags (dedup by end)."""
    merged: dict[str, dict] = {}
    for key in CONCEPTS["revenue"]:
        for p in _concept_points(facts, key):
            end = p.get("end")
            if end and (end not in merged or p.get("filed", "") < merged[end].get("filed", "")):
                merged[end] = p
    return list(merged.values())


def _revenue_n_years_before(facts: dict, cutoff: date, years_back: int) -> tuple[float | None, str | None]:
    """Revenue from `years_back` fiscal years before the latest-known one."""
    pts = _revenue_points(facts)
    latest = _latest_before(pts, cutoff)
    if not latest:
        return None, None
    target_year = int(latest["end"][:4]) - years_back
    candidates = [
        p for p in pts
        if p.get("end", "").startswith(str(target_year))
        and p.get("filed") and date.fromisoformat(p["filed"]) < cutoff
    ]
    if not candidates:
        return None, None
    pick = max(candidates, key=lambda p: p["end"])
    return float(pick["val"]), pick["end"]


def compute_fundamentals(facts: dict, vintage_date: date) -> dict:
    rev = _value_at(facts, CONCEPTS["revenue"], vintage_date)
    ni = _value_at(facts, CONCEPTS["net_income"], vintage_date)
    gp = _value_at(facts, CONCEPTS["gross_profit"], vintage_date)
    oi = _value_at(facts, CONCEPTS["operating_income"], vintage_date)
    assets = _value_at(facts, CONCEPTS["assets"], vintage_date)
    equity = _value_at(facts, CONCEPTS["equity"], vintage_date)
    ltd = _value_at(facts, CONCEPTS["long_term_debt"], vintage_date)
    shares = _value_at(facts, CONCEPTS["shares"], vintage_date)

    def ratio(a, b):
        return round(a / b, 4) if (a is not None and b not in (None, 0)) else None

    out: dict = {
        "net_income": ni,
        "revenue": rev,
        "stockholders_equity": equity,
        "shares_outstanding": shares,
        "gross_margin": ratio(gp, rev),
        "operating_margin": ratio(oi, rev),
        "net_margin": ratio(ni, rev),
        "debt_to_equity": ratio(ltd, equity),
        "return_on_equity": ratio(ni, equity),
        "return_on_assets": ratio(ni, assets),
    }
    rev_3y, _ = _revenue_n_years_before(facts, vintage_date, 3)
    if rev is not None and rev_3y not in (None, 0) and rev > 0 and rev_3y > 0:
        out["revenue_cagr_3y"] = round((rev / rev_3y) ** (1 / 3) - 1, 4)
    else:
        out["revenue_cagr_3y"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default="data/fundamentals.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.universe)
    if args.tickers:
        df = df[df["ticker"].isin(args.tickers)]
    elif args.limit:
        df = df.head(args.limit)

    cik_index = load_cik_index()
    print(f"→ Fundamentals for {len(df)} (ticker, vintage) rows; {len(cik_index)} tickers in CIK index")

    rows: list[dict] = []
    n_ok = n_nocik = n_nofacts = 0
    facts_cache: dict[int, dict | None] = {}
    with _session() as client:
        for i, row in enumerate(df.itertuples(index=False), 1):
            cik = cik_index.get(row.ticker)
            if cik is None:
                n_nocik += 1
                continue
            if cik not in facts_cache:
                facts_cache[cik] = fetch_companyfacts(cik, client)
            facts = facts_cache[cik]
            if facts is None:
                n_nofacts += 1
                continue
            fund = compute_fundamentals(facts, date.fromisoformat(row.vintage_date))
            fund.update({"ticker": row.ticker, "vintage_year": row.vintage_year, "vintage_date": row.vintage_date})
            rows.append(fund)
            n_ok += 1
            if i % 50 == 0:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(f"  {i}/{len(df)}  ok={n_ok} no_cik={n_nocik} no_facts={n_nofacts}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(rows)} rows · ok={n_ok} no_cik={n_nocik} no_facts={n_nofacts})")


if __name__ == "__main__":
    main()
