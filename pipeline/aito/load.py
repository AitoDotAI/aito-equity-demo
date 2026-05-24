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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(out_csv, index=False)
    print(f"→ {out_csv} ({len(universe)} rows, {len(universe.columns)} cols)")
    return universe


def df_to_aito_rows(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to Aito-friendly row dicts.

    - NaN → None (Aito rejects NaN; nulls are fine for optional columns)
    - bool/numeric coerced to Python primitives so JSON serialises cleanly
    """
    rows: list[dict] = []
    for raw in df.to_dict(orient="records"):
        clean: dict = {}
        for k, v in raw.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            if pd.isna(v):
                continue
            if hasattr(v, "item"):  # numpy scalar
                v = v.item()
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
