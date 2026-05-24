"""LLM extraction: 10-K + DEF 14A → qualitative grades + rationale strings.

Provider: OpenAI (gpt-5-mini by default; swap via OPENAI_MODEL env var).

For each (ticker, vintage):
  1. Load cached EDGAR filings (10-K + DEF 14A) from disk
  2. Trim to bounded excerpts (~14K tokens of filing text)
  3. For each of the 4 features, run 3 modal-aggregated calls
     Caching strategy:
       - Put the filing excerpt early in the system prompt (stable across
         all 12 calls per company). OpenAI auto-caches prefixes >1024
         tokens at the server side; the per-feature prompt at the end
         is the only fresh suffix.
  4. Aggregate modal categorical answer; retain first run's rationale

Cost guard (gpt-5-mini, conservative):
  14K input prefix + 12 calls × (~800 fresh input + ~300 output)
  Cached input is billed at a discount; mini-tier output tokens are cheap.
  Expected: ~$0.02-0.05 per company → ~$5-15 for 250 companies.

Run with OPENAI_API_KEY set. Use --confirm-cost to actually call the API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from pipeline.extraction.sections import excerpt_for_filing

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
FILING_TOKEN_BUDGET = 14_000
PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUT_CSV = Path("data/llm_features.csv")
FILINGS_DIR = Path("data/10k_excerpts")


# ── Pydantic schemas (one per feature) ──────────────────────────


class MarketPositionGrade(BaseModel):
    market_position: Literal["dominant", "strong", "competitive", "lagging"]
    rationale: str


class MoatGrade(BaseModel):
    moat_type: Literal[
        "network_effects",
        "switching_costs",
        "scale_economies",
        "brand",
        "regulatory",
        "cost_advantage",
        "none",
    ]
    moat_strength: Literal["wide", "narrow", "none"]
    rationale: str


class MarketQualityGrade(BaseModel):
    market_quality: Literal["secular_growth", "stable", "cyclical", "declining", "disrupted"]
    rationale: str


class LeadershipSubScores(BaseModel):
    capital_allocation: int = Field(ge=1, le=5)
    strategic_clarity: int = Field(ge=1, le=5)
    execution_track_record: int = Field(ge=1, le=5)


class LeadershipGrade(BaseModel):
    leadership_quality: int = Field(ge=1, le=5)
    sub_scores: LeadershipSubScores
    rationale: str


FEATURE_SCHEMAS: dict[str, type[BaseModel]] = {
    "market_position": MarketPositionGrade,
    "moat": MoatGrade,
    "market_quality": MarketQualityGrade,
    "leadership": LeadershipGrade,
}

PRIMARY_FIELD: dict[str, str] = {
    "market_position": "market_position",
    "moat": "moat_strength",
    "market_quality": "market_quality",
    "leadership": "leadership_quality",
}


# ── Helpers ────────────────────────────────────────────────────


def load_prompt(feature: str, vintage_date: date) -> str:
    text = (PROMPTS_DIR / f"{feature}.md").read_text(encoding="utf-8")
    return text.replace("{VINTAGE_DATE}", vintage_date.isoformat())


def load_filings_for(ticker: str, vintage_date: date) -> dict[str, str]:
    """Return {filing_type: full_text} for the latest 10-K + DEF 14A before vintage_date."""
    by_type: dict[str, tuple[date, str]] = {}
    ticker_dir = FILINGS_DIR / ticker
    if not ticker_dir.exists():
        return {}
    for accession_dir in ticker_dir.iterdir():
        meta_path = accession_dir / "meta.json"
        text_path = accession_dir / "text.txt"
        if not (meta_path.exists() and text_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        filed = date.fromisoformat(meta["filed_date"])
        if filed >= vintage_date:
            continue
        prev = by_type.get(meta["filing_type"])
        if prev is None or filed > prev[0]:
            by_type[meta["filing_type"]] = (filed, text_path.read_text(encoding="utf-8", errors="replace"))
    return {ft: payload[1] for ft, payload in by_type.items()}


def build_filing_excerpt(filings: dict[str, str]) -> str:
    """Combine + budget-trim 10-K and DEF 14A excerpts into one prompt block."""
    parts: list[str] = []
    if "10-K" in filings:
        ex = excerpt_for_filing("10-K", filings["10-K"])
        parts.append(ex.to_prompt_text(int(FILING_TOKEN_BUDGET * 0.75)))
    if "DEF 14A" in filings:
        ex = excerpt_for_filing("DEF 14A", filings["DEF 14A"])
        parts.append(ex.to_prompt_text(int(FILING_TOKEN_BUDGET * 0.25)))
    return "\n\n".join(parts)


def build_system_message(filing_excerpt: str) -> str:
    """One large stable string — placed in `system` so OpenAI auto-caches it.

    OpenAI's automatic prompt caching keys on the prefix of the messages
    array; putting all stable content (analyst persona + filing text)
    here means the per-feature prompt below it is the only fresh suffix.
    """
    return (
        "You are an experienced equity research analyst grading companies for a "
        "structured database. Apply the criteria literally, return strict JSON in "
        "the structured-output format requested, and respect point-in-time discipline "
        "(no post-vintage knowledge — base your assessment only on the filings excerpt below).\n\n"
        "FILINGS EXCERPT (point-in-time, as of the vintage date):\n\n"
        f"{filing_excerpt}"
    )


# ── LLM grader ─────────────────────────────────────────────────


def grade_one(
    client: OpenAI,
    feature: str,
    filing_excerpt: str,
    vintage_date: date,
    run_idx: int,
) -> BaseModel | None:
    """Single grading call. Returns the parsed Pydantic model or None on failure."""
    schema = FEATURE_SCHEMAS[feature]
    prompt = load_prompt(feature, vintage_date)
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": build_system_message(filing_excerpt)},
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"  ✗ {feature} run {run_idx}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def modal_aggregate(results: list[BaseModel], primary_field: str) -> tuple[BaseModel, int]:
    """Return (modal-winning result, count). First-occurrence wins on tie."""
    values = [getattr(r, primary_field) for r in results]
    counter = Counter(values)
    top_value, top_count = counter.most_common(1)[0]
    for r in results:
        if getattr(r, primary_field) == top_value:
            return r, top_count
    return results[0], top_count


def grade_company(
    client: OpenAI,
    ticker: str,
    vintage_date: date,
    n_runs: int = 3,
) -> dict[str, BaseModel] | None:
    """Run 4 features × n_runs calls. Returns modal-aggregated grades per feature."""
    filings = load_filings_for(ticker, vintage_date)
    if not filings:
        return None
    excerpt = build_filing_excerpt(filings)
    if not excerpt.strip():
        return None

    out: dict[str, BaseModel] = {}
    for feature in FEATURE_SCHEMAS:
        results: list[BaseModel] = []
        for run_idx in range(n_runs):
            r = grade_one(client, feature, excerpt, vintage_date, run_idx)
            if r is not None:
                results.append(r)
        if not results:
            print(f"  ✗ {ticker} {feature}: 0 of {n_runs} runs succeeded", file=sys.stderr)
            continue
        modal, _ = modal_aggregate(results, PRIMARY_FIELD[feature])
        out[feature] = modal
    return out


# ── CSV materialisation ─────────────────────────────────────────


def grades_to_row(ticker: str, vintage_year: int, vintage_date: date, grades: dict[str, BaseModel]) -> dict:
    row = {
        "ticker": ticker,
        "vintage_year": vintage_year,
        "vintage_date": vintage_date.isoformat(),
    }
    if "market_position" in grades:
        g: MarketPositionGrade = grades["market_position"]  # type: ignore[assignment]
        row["market_position"] = g.market_position
        row["market_position_rationale"] = g.rationale
    if "moat" in grades:
        g: MoatGrade = grades["moat"]  # type: ignore[assignment]
        row["moat_type"] = g.moat_type
        row["moat_strength"] = g.moat_strength
        row["moat_rationale"] = g.rationale
    if "market_quality" in grades:
        g: MarketQualityGrade = grades["market_quality"]  # type: ignore[assignment]
        row["market_quality"] = g.market_quality
        row["market_quality_rationale"] = g.rationale
    if "leadership" in grades:
        g: LeadershipGrade = grades["leadership"]  # type: ignore[assignment]
        row["leadership_quality"] = g.leadership_quality
        row["capital_allocation"] = g.sub_scores.capital_allocation
        row["strategic_clarity"] = g.sub_scores.strategic_clarity
        row["execution_track_record"] = g.sub_scores.execution_track_record
        row["leadership_rationale"] = g.rationale
    return row


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default=str(OUTPUT_CSV))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--model", default=MODEL, help=f"OpenAI model id (default {MODEL})")
    args = parser.parse_args()

    df = pd.read_csv(args.universe)
    if args.tickers:
        df = df[df["ticker"].isin(args.tickers)]
    elif args.limit:
        df = df.head(args.limit)

    MODEL = args.model

    n_calls = len(df) * 4 * args.n_runs
    if not args.confirm_cost:
        print(
            f"DRY RUN: would call OpenAI {MODEL} for {len(df)} (ticker, vintage) rows × "
            f"4 features × {args.n_runs} runs = {n_calls} calls. "
            f"Estimated cost (mini tier): ~${len(df) * 0.03:.2f}. "
            f"Pass --confirm-cost to actually run."
        )
        return

    client = OpenAI()  # uses OPENAI_API_KEY env var
    print(f"→ Grading {len(df)} (ticker, vintage) rows × 4 features × {args.n_runs} runs (model={MODEL})")

    rows: list[dict] = []
    no_filings = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        vintage_date = date.fromisoformat(row.vintage_date)
        grades = grade_company(client, row.ticker, vintage_date, n_runs=args.n_runs)
        if grades is None:
            no_filings += 1
            print(f"  ⚠ {row.ticker} @ {row.vintage_year}: no cached filings; skipped")
            continue
        rows.append(grades_to_row(row.ticker, row.vintage_year, vintage_date, grades))
        if i % 5 == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  {i}/{len(df)} done (no_filings={no_filings})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"→ {args.out} ({len(rows)} rows, no_filings={no_filings})")


if __name__ == "__main__":
    main()
