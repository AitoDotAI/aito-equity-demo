"""LLM extraction: 10-K + DEF 14A → qualitative grades + rationale strings.

Provider: Azure OpenAI (deployment-based; configured via env vars).

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

Azure env contract (see .env.example):
  OPENAI_MODEL_URL          — resource endpoint, e.g.
                              https://swedencentral.api.cognitive.microsoft.com
  OPENAI_MODEL_API_KEY      — resource API key
  OPENAI_MODEL_DEPLOYMENT   — deployment name (used as the `model` arg)
  OPENAI_MODEL_API_VERSION  — API version, e.g. 2024-08-01-preview

Use --confirm-cost to actually call the API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from pipeline.extraction.sections import excerpt_for_filing

load_dotenv()

# On Azure OpenAI, the `model` argument is the *deployment name*, not the
# base model id. Default to the deployment; allow --model override.
MODEL = os.environ.get("OPENAI_MODEL_DEPLOYMENT", "gpt-5-mini")
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


class CombinedGrade(BaseModel):
    """All four features in one structured response — one call per company."""

    market_position: Literal["dominant", "strong", "competitive", "lagging"]
    market_position_rationale: str
    moat_type: Literal[
        "network_effects", "switching_costs", "scale_economies",
        "brand", "regulatory", "cost_advantage", "none",
    ]
    moat_strength: Literal["wide", "narrow", "none"]
    moat_rationale: str
    market_quality: Literal["secular_growth", "stable", "cyclical", "declining", "disrupted"]
    market_quality_rationale: str
    leadership_quality: int = Field(ge=1, le=5)
    capital_allocation: int = Field(ge=1, le=5)
    strategic_clarity: int = Field(ge=1, le=5)
    execution_track_record: int = Field(ge=1, le=5)
    leadership_rationale: str


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


def make_client() -> AzureOpenAI:
    """Construct the Azure OpenAI client from env vars."""
    endpoint = os.environ.get("OPENAI_MODEL_URL")
    api_key = os.environ.get("OPENAI_MODEL_API_KEY")
    api_version = os.environ.get("OPENAI_MODEL_API_VERSION", "2024-08-01-preview")
    if not (endpoint and api_key):
        raise RuntimeError(
            "OPENAI_MODEL_URL and OPENAI_MODEL_API_KEY must be set (see .env.example)."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        max_retries=8,   # Azure mini deployments have low TPM; lean on SDK backoff
        timeout=90.0,
    )


def grade_one(
    client: AzureOpenAI,
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
            model=MODEL,  # on Azure, this is the deployment name
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


def grade_company_combined(
    client: AzureOpenAI,
    feature_excerpt: str,
    vintage_date: date,
    n_runs: int = 1,
) -> CombinedGrade | None:
    """One structured call grading all four features at once.

    4× fewer calls and 4× less token volume than per-feature grading — the
    14K-token filing excerpt is sent once, not four times. With n_runs > 1,
    modal-aggregate on moat_strength (the most judgment-heavy axis).
    """
    prompt = load_prompt("combined", vintage_date)
    results: list[CombinedGrade] = []
    for run_idx in range(n_runs):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": build_system_message(feature_excerpt)},
                    {"role": "user", "content": prompt},
                ],
                response_format=CombinedGrade,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                results.append(parsed)
        except Exception as e:
            print(f"  ✗ combined run {run_idx}: {type(e).__name__}: {e}", file=sys.stderr)
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    modal, _ = modal_aggregate(results, "moat_strength")
    return modal  # type: ignore[return-value]


def grade_company(
    client: AzureOpenAI,
    ticker: str,
    vintage_date: date,
    n_runs: int = 1,
) -> CombinedGrade | None:
    filings = load_filings_for(ticker, vintage_date)
    if not filings:
        return None
    excerpt = build_filing_excerpt(filings)
    if not excerpt.strip():
        return None
    return grade_company_combined(client, excerpt, vintage_date, n_runs=n_runs)


# ── CSV materialisation ─────────────────────────────────────────


def grades_to_row(ticker: str, vintage_year: int, vintage_date: date, g: CombinedGrade) -> dict:
    return {
        "ticker": ticker,
        "vintage_year": vintage_year,
        "vintage_date": vintage_date.isoformat(),
        "market_position": g.market_position,
        "market_position_rationale": g.market_position_rationale,
        "moat_type": g.moat_type,
        "moat_strength": g.moat_strength,
        "moat_rationale": g.moat_rationale,
        "market_quality": g.market_quality,
        "market_quality_rationale": g.market_quality_rationale,
        "leadership_quality": g.leadership_quality,
        "capital_allocation": g.capital_allocation,
        "strategic_clarity": g.strategic_clarity,
        "execution_track_record": g.execution_track_record,
        "leadership_rationale": g.leadership_rationale,
    }


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--out", default=str(OUTPUT_CSV))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Modal-aggregation runs per feature (default 1; 3 for robustness)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Concurrent company-grading threads (default 3; Azure mini quota is low)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (ticker, vintage) rows already present in --out")
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--model", default=MODEL, help=f"Azure deployment name (default {MODEL})")
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
            f"DRY RUN: would call Azure OpenAI deployment '{MODEL}' for {len(df)} "
            f"(ticker, vintage) rows × 4 features × {args.n_runs} runs = {n_calls} calls. "
            f"Estimated cost (mini tier): ~${len(df) * 0.03:.2f}. "
            f"Pass --confirm-cost to actually run."
        )
        return

    client = make_client()
    print(
        f"→ Grading {len(df)} (ticker, vintage) rows × 4 features × {args.n_runs} runs "
        f"(deployment={MODEL}, concurrency={args.concurrency})"
    )

    # Resume support: skip rows already present in the output CSV.
    out_path = Path(args.out)
    done_keys: set[tuple[str, int]] = set()
    existing_rows: list[dict] = []
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        existing_rows = prev.to_dict(orient="records")
        done_keys = {(r["ticker"], int(r["vintage_year"])) for r in existing_rows}
        print(f"  resume: {len(done_keys)} rows already graded; skipping them")

    work = [
        row
        for row in df.itertuples(index=False)
        if (row.ticker, int(row.vintage_year)) not in done_keys
    ]

    rows: list[dict] = list(existing_rows)
    counters = {"done": 0, "no_filings": 0, "failed": 0}
    lock = threading.Lock()
    total = len(work)

    def worker(row) -> dict | None:
        vintage_date = date.fromisoformat(row.vintage_date)
        try:
            grades = grade_company(client, row.ticker, vintage_date, n_runs=args.n_runs)
        except Exception as e:
            print(f"  ✗ {row.ticker} @ {row.vintage_year}: {type(e).__name__}: {e}", file=sys.stderr)
            with lock:
                counters["failed"] += 1
            return None
        if grades is None:
            with lock:
                counters["no_filings"] += 1
            return None
        return grades_to_row(row.ticker, row.vintage_year, vintage_date, grades)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(worker, row): row for row in work}
        for n, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            with lock:
                if result is not None:
                    rows.append(result)
                    counters["done"] += 1
                processed = n
            if processed % 10 == 0 or processed == total:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with lock:
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                print(
                    f"  {processed}/{total} processed · graded={counters['done']} "
                    f"no_filings={counters['no_filings']} failed={counters['failed']}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(
        f"→ {out_path} ({len(rows)} rows total · graded this run={counters['done']} "
        f"no_filings={counters['no_filings']} failed={counters['failed']})"
    )


if __name__ == "__main__":
    main()
