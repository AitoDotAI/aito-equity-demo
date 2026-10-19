"""Merge pipeline outputs → companies table → push to Aito.

Inputs (all in data/):
  universe.csv        — point-in-time index constituents (1520 rows)
  outcomes.csv        — yfinance forward returns + bucket
  llm_features.csv    — LLM-graded qualitative features

Output:
  data/companies.csv   — merged table (one row per ticker × vintage)
  → uploaded to Aito as `companies`

Idempotent: drops + recreates the `companies` table on every run.
Needs AITO_API_URL + AITO_API_KEY in env (or .env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from pipeline.aito.client import AitoClient

SCHEMA_PATH = Path(__file__).parent / "schema.json"
DEFAULT_TABLE = "companies"


def merge_pipeline_outputs(
    universe_csv: Path = Path("data/universe.csv"),
    outcomes_csv: Path = Path("data/outcomes.csv"),
    features_csv: Path = Path("data/llm_features.csv"),
    out_csv: Path = Path("data/companies.csv"),
) -> pd.DataFrame:
    """Left-join universe + outcomes + llm_features into one wide table."""
    universe = pd.read_csv(universe_csv)
    print(f"  universe: {len(universe)} rows")

    if outcomes_csv.exists():
        outcomes = pd.read_csv(outcomes_csv)
        print(f"  outcomes: {len(outcomes)} rows")
        universe = universe.merge(
            outcomes,
            on=["ticker", "vintage_year", "vintage_date"],
            how="left",
            suffixes=("", "_outcome"),
        )

    if features_csv.exists():
        features = pd.read_csv(features_csv)
        print(f"  llm_features: {len(features)} rows")
        universe = universe.merge(
            features,
            on=["ticker", "vintage_year", "vintage_date"],
            how="left",
            suffixes=("", "_feat"),
        )

    # Fundamentals (SEC XBRL) + market factors (yfinance) + filing events +
    # 6/12-month excess returns (pipeline.prices.midterm).
    for extra_csv, label in [
        (Path("data/fundamentals.csv"), "fundamentals"),
        (Path("data/market_factors.csv"), "market_factors"),
        (Path("data/filing_events.csv"), "filing_events"),
        (Path("data/midterm_outcomes.csv"), "midterm_outcomes"),
        (Path("data/valuation_history.csv"), "valuation_history"),
    ]:
        if extra_csv.exists():
            extra = pd.read_csv(extra_csv)
            print(f"  {label}: {len(extra)} rows")
            # Drop overlapping non-key cols so we don't get _x/_y suffixes.
            keys = ["ticker", "vintage_year", "vintage_date"]
            dup = [c for c in extra.columns if c in universe.columns and c not in keys]
            extra = extra.drop(columns=dup)
            universe = universe.merge(extra, on=keys, how="left")

    # Derived valuation factors (need market cap = price × shares) + buckets.
    universe = _derive_valuation_and_buckets(universe)

    # Backfill company_name for tickers absent from today's S&P table (removed
    # constituents have no row in the current Wikipedia table). The EDGAR
    # ticker→title index, cached during the filings stage, covers most of them.
    universe = _backfill_company_names(universe)
    universe = _backfill_delisted_metadata(universe)
    universe["profile_text"] = universe.apply(profile_text, axis=1)

    # Deterministic company-grouped fold id (hashlib, NOT Python hash() which is
    # salted per-process) for random cross-year evaluation: each company lands in
    # one fold, so folds mix vintages without same-company train/test leakage.
    # Used by `./do pipeline model-eval --random`; harmless to the demo.
    universe["eval_fold"] = universe["ticker"].astype(str).map(
        lambda t: str(int(hashlib.md5(t.encode()).hexdigest(), 16) % 5)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(out_csv, index=False)
    print(f"→ {out_csv} ({len(universe)} rows, {len(universe.columns)} cols)")
    return universe


# Fields whose values make up a company's "profile" for analogue search.
PROFILE_FIELDS = [
    "market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality",
    "capital_allocation", "strategic_clarity", "execution_track_record", "sector",
    "valuation_bucket", "growth_bucket", "leverage_bucket", "profitability_bucket",
    "momentum_bucket", "volatility_bucket",
]


def profile_token(field: str, value) -> str:
    """One analyzer-proof token per field=value: `moat_strength=wide` → `moatstrengthxwide`.

    Alphanumerics only, so any analyzer keeps it whole. The site builds the same
    tokens (`profileToken` in site/index.html) for the Lab's analogue query.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"[^a-z0-9]", "", f"{field}x{value}".lower())


def profile_text(row: pd.Series) -> str | None:
    """Space-joined profile tokens — the Text column v2 similarity ranks on.

    On the v2 (rep2) engine `_similarity` is text BM25 only; the v1 engine's
    similarity over categorical columns has no v2 counterpart. BM25 over these
    tokens is the same idea: IDF-weighted overlap of field=value pairs, so a
    shared rare value (moat = wide) counts more than a common one.
    """
    toks = [profile_token(f, row[f]) for f in PROFILE_FIELDS if f in row.index and pd.notna(row[f]) and row[f] != ""]
    return " ".join(toks) or None


def _derive_valuation_and_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute valuation ratios from market cap + fundamentals, then bucketize
    all continuous factors into interpretable bands for the factor explorer."""
    from pipeline.fundamentals.market_factors import (
        bucket_growth, bucket_leverage, bucket_momentum, bucket_pe, bucket_roe, bucket_volatility,
    )

    have = set(df.columns)
    # Market cap must be point-in-time: as-traded price × the share count of the
    # same date (pipeline.fundamentals.valuation_history). `vintage_price` is
    # adjusted to TODAY for later splits, so vintage_price × as-reported shares
    # divided every future split out of the market cap — a look-ahead leak that
    # labelled future winners "cheap" (AAPL 2014 at P/E 0.4). No PIT cap → no
    # valuation, rather than a wrong one.
    if "market_cap_pit" in have:
        df["market_cap"] = df["market_cap_pit"]
        have.add("market_cap")
    if "market_cap" in have:
        if "net_income" in have:
            df["pe_ratio"] = (df["market_cap"] / df["net_income"]).where(df["net_income"] != 0)
            df["earnings_yield"] = (df["net_income"] / df["market_cap"]).where(df["market_cap"] != 0)
        if "stockholders_equity" in have:
            df["pb_ratio"] = (df["market_cap"] / df["stockholders_equity"]).where(df["stockholders_equity"] > 0)

    # Bucketed (categorical) versions — what the factor explorer & relate use.
    bucketers = {
        "momentum_bucket": ("momentum_12m", bucket_momentum),
        "volatility_bucket": ("volatility_12m", bucket_volatility),
        "valuation_bucket": ("pe_ratio", bucket_pe),
        "growth_bucket": ("revenue_cagr_3y", bucket_growth),
        "leverage_bucket": ("debt_to_equity", bucket_leverage),
        "profitability_bucket": ("return_on_equity", bucket_roe),
        "pre_filing_mom_bucket": ("pre_filing_mom_60d", bucket_momentum),
    }
    for new_col, (src, fn) in bucketers.items():
        if src in df.columns:
            df[new_col] = df[src].map(fn)
    return _derive_relative_valuation(df)


# Cheap against what? Absolute P/E can't separate a stock that is always cheap
# (nothing to revert to) from one that is cheap against its own usual level.
OWN_BELOW, OWN_ABOVE = 0.8, 1.25


def _derive_relative_valuation(df: pd.DataFrame) -> pd.DataFrame:
    if "pe_ratio" not in df.columns:
        return df
    pe = df["pe_ratio"].where(df["pe_ratio"] > 0)
    if "pe_own_median_5y" in df.columns:
        df["pe_vs_own"] = (pe / df["pe_own_median_5y"]).round(3)
    sec_med = pe.groupby([df["vintage_year"], df["sector"]]).transform("median")
    df["pe_vs_sector"] = (pe / sec_med).round(3)

    def own_band(r):
        if pd.isna(r):
            return None
        return "below_own_norm" if r < OWN_BELOW else ("above_own_norm" if r > OWN_ABOVE else "near_own_norm")
    df["valuation_vs_own"] = df.get("pe_vs_own", pd.Series(index=df.index, dtype=float)).map(own_band)

    def kind(row):
        own, vb = row.get("valuation_vs_own"), row.get("valuation_bucket")
        if own is None or vb in (None, "negative_earnings") or pd.isna(vb):
            return None
        if own == "below_own_norm":
            return "derated"                # cheaper than it usually is — the reversion case
        if vb == "cheap":
            return "structurally_cheap"     # low P/E, but that is its normal
        if vb == "fair" and own == "near_own_norm":
            return "normally_priced"        # the control group
        return "other"
    df["cheap_kind"] = df.apply(kind, axis=1)
    return df


def _smart_title(name: str) -> str:
    """Title-case an ALL-CAPS registrant name without mangling mixed-case ones.

    'ADVANCE AUTO PARTS INC' → 'Advance Auto Parts Inc'; leaves 'AbbVie' alone.
    """
    if not name or not name.isupper():
        return name
    small = {"and", "of", "the"}
    out = []
    for w in name.title().split():
        low = w.lower()
        out.append(low if low in small else w)
    return " ".join(out)


# Point-in-time GICS sector + name for curated delisted names absent from
# today's S&P table (they have no row in the current Wikipedia constituents,
# so universe.csv leaves sector null). These are the wipeout cohort we grade
# from their pre-bankruptcy 10-K; without sector they can't match sector-keyed
# similarity queries. Sectors are the company's GICS classification at vintage.
_DELISTED_META: dict[str, tuple[str, str]] = {
    "FTR": ("Communication Services", "Frontier Communications"),
    "WIN": ("Communication Services", "Windstream Holdings"),
    "CHK": ("Energy", "Chesapeake Energy"),
    "DNR": ("Energy", "Denbury Resources"),
    "DO": ("Energy", "Diamond Offshore Drilling"),
    "NE": ("Energy", "Noble Corporation"),
    "ESV": ("Energy", "Ensco"),
    "RDC": ("Energy", "Rowan Companies"),
    "BTU": ("Energy", "Peabody Energy"),
}


def _backfill_delisted_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Fill sector (and name) for curated delisted names the live S&P table drops."""
    filled = 0
    for tk, (sector, name) in _DELISTED_META.items():
        mask = (df["ticker"] == tk) & (df["sector"].isna() | (df["sector"].astype(str).str.strip() == ""))
        if mask.any():
            df.loc[mask, "sector"] = sector
            nm = df["company_name"].isna() | (df["company_name"].astype(str).str.strip().isin(["", tk]))
            df.loc[(df["ticker"] == tk) & nm, "company_name"] = name
            filled += int(mask.sum())
    if filled:
        print(f"  backfilled sector for {filled} curated delisted rows")
    return df


def _backfill_company_names(df: pd.DataFrame) -> pd.DataFrame:
    idx_path = Path("data/10k_excerpts/_index/company_tickers.json")
    if not idx_path.exists():
        return df
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return df
    by_ticker = {v["ticker"]: _smart_title(str(v.get("title", ""))) for v in idx.values()}

    missing_mask = df["company_name"].isna() | (df["company_name"].astype(str).str.strip() == "")
    n_before = int(missing_mask.sum())
    df.loc[missing_mask, "company_name"] = df.loc[missing_mask, "ticker"].map(by_ticker)
    # Anything still missing falls back to the ticker itself.
    still_missing = df["company_name"].isna() | (df["company_name"].astype(str).str.strip() == "")
    df.loc[still_missing, "company_name"] = df.loc[still_missing, "ticker"]
    n_filled = n_before - int(still_missing.sum())
    print(f"  backfilled {n_filled} company names from EDGAR index ({int(still_missing.sum())} fell back to ticker)")
    return df


# Columns typed Int in schema.json. Pandas promotes int columns to float64
# when any NaN is present, so 4 becomes 4.0 — Aito then rejects "Double" for
# an Integer column. Coerce these back to int after the NaN check.
INT_COLUMNS = {
    "vintage_year",
    "years_public",
    "leadership_quality",
    "capital_allocation",
    "strategic_clarity",
    "execution_track_record",
}


def df_to_aito_rows(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to Aito-friendly row dicts.

    - NaN → omitted (Aito rejects NaN; missing keys are fine for nullable columns)
    - numpy scalars → Python primitives so JSON serialises cleanly
    - Int-typed columns coerced back from float (pandas NaN promotion)
    """
    rows: list[dict] = []
    for raw in df.to_dict(orient="records"):
        clean: dict = {}
        for k, v in raw.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            if pd.isna(v):
                continue
            if hasattr(v, "item"):  # numpy scalar → python scalar
                v = v.item()
            if k in INT_COLUMNS and isinstance(v, float):
                v = int(round(v))
            clean[k] = v
        rows.append(clean)
    return rows


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def upload_to_aito(rows: list[dict], table: str = DEFAULT_TABLE) -> None:
    with AitoClient() as client:
        print(f"  drop table '{table}' (if exists)")
        client.delete_table(table)
        print(f"  put schema")
        client.put_schema(load_schema())
        print(f"  upload {len(rows)} rows")
        n = client.upload_batch(table, rows)
        print(f"→ uploaded {n} rows to {client.config.api_root}/data/{table}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-only", action="store_true", help="Build data/companies.csv only; skip Aito upload")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    args = parser.parse_args()

    df = merge_pipeline_outputs()
    if args.merge_only:
        return

    rows = df_to_aito_rows(df)
    upload_to_aito(rows, table=args.table)


if __name__ == "__main__":
    main()
