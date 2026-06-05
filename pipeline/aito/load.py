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
import json
import math
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

    # Fundamentals (SEC XBRL) + market factors (yfinance) + filing events.
    for extra_csv, label in [
        (Path("data/fundamentals.csv"), "fundamentals"),
        (Path("data/market_factors.csv"), "market_factors"),
        (Path("data/filing_events.csv"), "filing_events"),
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(out_csv, index=False)
    print(f"→ {out_csv} ({len(universe)} rows, {len(universe.columns)} cols)")
    return universe


def _derive_valuation_and_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute valuation ratios from market cap + fundamentals, then bucketize
    all continuous factors into interpretable bands for the factor explorer."""
    from pipeline.fundamentals.market_factors import (
        bucket_growth, bucket_leverage, bucket_momentum, bucket_pe, bucket_roe, bucket_volatility,
    )

    have = set(df.columns)
    if {"vintage_price", "shares_outstanding"} <= have:
        df["market_cap"] = df["vintage_price"] * df["shares_outstanding"]
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
        print(f"→ uploaded {n} rows to {client.config.base_url}/api/v1/data/{table}")


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
