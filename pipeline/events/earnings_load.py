"""Build the earnings_events Aito table: LLM signals + bucketed reaction.

Joins the extracted press-release signals to the market reaction, buckets
the day-1 move into down / flat / up, and loads it to Aito as a second
table `earnings_events`. The Live Query / Predict-a-reaction surface then
does a real Aito `predict` of the reaction bucket given the signals a model
read from a pasted release.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from pipeline.aito.client import AitoClient

SIGNAL_COLS = [
    "headline_signal", "reported_beat", "guidance",
    "eps_direction", "revenue_direction", "one_time_items", "tone",
]

SCHEMA = {
    "schema": {
        "earnings_events": {
            "type": "table",
            "columns": {
                "ticker": {"type": "String", "nullable": True},
                "date": {"type": "String", "nullable": True},
                "headline_signal": {"type": "String", "nullable": True},
                "reported_beat": {"type": "String", "nullable": True},
                "guidance": {"type": "String", "nullable": True},
                "eps_direction": {"type": "String", "nullable": True},
                "revenue_direction": {"type": "String", "nullable": True},
                "one_time_items": {"type": "String", "nullable": True},
                "tone": {"type": "String", "nullable": True},
                "react_1d": {"type": "Decimal", "nullable": True},
                "react_1d_bucket": {"type": "String", "nullable": True},
                "react_20d": {"type": "Decimal", "nullable": True},
            },
        }
    }
}

TABLE = "earnings_events"


def bucket_1d(r: float) -> str:
    if r <= -2:
        return "down"
    if r < 2:
        return "flat"
    return "up"


def build(signals_csv: Path, events_csv: Path, out_csv: Path) -> pd.DataFrame:
    sig = pd.read_csv(signals_csv)
    ev = pd.read_csv(events_csv)
    ev = ev[ev["theme"] == "earnings"][["ticker", "date", "react_1d", "react_20d"]]
    df = sig.merge(ev, on=["ticker", "date"], how="inner")
    df = df[df["react_1d"].notna()].copy()
    df["react_1d_bucket"] = df["react_1d"].map(bucket_1d)
    keep = ["ticker", "date", *SIGNAL_COLS, "react_1d", "react_1d_bucket", "react_20d"]
    df = df[[c for c in keep if c in df.columns]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"→ {out_csv} ({len(df)} earnings events)")
    print("  react_1d_bucket:", df["react_1d_bucket"].value_counts().to_dict())
    return df


def to_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for raw in df.to_dict(orient="records"):
        clean = {}
        for k, v in raw.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            if pd.isna(v):
                continue
            if hasattr(v, "item"):
                v = v.item()
            clean[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(clean)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", default="data/earnings_signals.csv")
    parser.add_argument("--events", default="data/news_events.csv")
    parser.add_argument("--out", default="data/earnings_table.csv")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    df = build(Path(args.signals), Path(args.events), Path(args.out))
    if args.merge_only:
        return
    with AitoClient() as c:
        print(f"  drop + recreate table '{TABLE}'")
        c.delete_table(TABLE)
        c.put_schema(SCHEMA)
        n = c.upload_batch(TABLE, to_rows(df))
        print(f"→ uploaded {n} rows to {TABLE}")


if __name__ == "__main__":
    main()
