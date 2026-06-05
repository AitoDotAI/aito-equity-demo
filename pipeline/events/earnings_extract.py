"""LLM-extract structured signals from earnings press releases.

The trader's question: can you read the news at t=0 and anticipate the
reaction? We extract the signals a human skims for in the headline +
first paragraphs — direction vs expectations, revenue/EPS direction,
guidance change, one-offs, tone — and (downstream) test which of them
actually predict the market reaction.

Honest caveat baked into the prompt: the "beat/miss" a press release
states is the company's framing, not the move vs live Wall Street
consensus (which isn't free). So `reported_beat` captures what the
release claims; the genuinely-new signals (guidance change, one-offs)
are the ones expected to carry predictive weight.

Output: data/earnings_signals.csv. Azure gpt-5-mini, concurrent + resume.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel

load_dotenv(override=True)

MODEL = os.environ.get("OPENAI_MODEL_DEPLOYMENT", "gpt-5-mini")
CACHE_DIR = Path("data/earnings_8k")
RELEASE_CHAR_CAP = 9000  # press releases are short; keep the headline + financials


class EarningsSignals(BaseModel):
    reported_beat: Literal["beat", "miss", "inline", "not_stated"]
    revenue_direction: Literal["up", "flat", "down", "not_stated"]
    eps_direction: Literal["up", "flat", "down", "not_stated"]
    guidance: Literal["raised", "maintained", "lowered", "withdrawn", "none"]
    one_time_items: Literal["none", "charges", "gains", "both"]
    tone: Literal["confident", "neutral", "cautious"]
    headline_signal: Literal["clearly_positive", "mixed", "clearly_negative"]


PROMPT = """You are reading a company's quarterly earnings press release as it hits the wire.
Extract the signals a trader skims for in the headline and first paragraphs.

Return the structured object:
- reported_beat: did the company frame results as beating / missing / in line with expectations? "not_stated" if no expectation reference.
- revenue_direction / eps_direction: year-over-year direction of revenue / EPS.
- guidance: did forward guidance get raised / maintained / lowered / withdrawn, or is there none?
- one_time_items: any one-off charges, gains, both, or none.
- tone: overall tone of management's quoted commentary.
- headline_signal: taking the headline + lead bullets only, is the news clearly positive, mixed, or clearly negative?

Base this ONLY on the release text. Do not use outside knowledge of how the stock moved."""


def make_client() -> AzureOpenAI:
    endpoint = os.environ.get("OPENAI_MODEL_URL")
    key = os.environ.get("OPENAI_MODEL_API_KEY")
    ver = os.environ.get("OPENAI_MODEL_API_VERSION", "2024-08-01-preview")
    if not (endpoint and key):
        raise RuntimeError("OPENAI_MODEL_URL and OPENAI_MODEL_API_KEY must be set")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=ver, max_retries=8, timeout=90.0)


def grade_release(client: AzureOpenAI, text: str) -> EarningsSignals | None:
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You extract structured signals from earnings press releases for a trading database. Be literal; base everything on the text."},
                {"role": "user", "content": f"{PROMPT}\n\nPRESS RELEASE:\n\n{text[:RELEASE_CHAR_CAP]}"},
            ],
            response_format=EarningsSignals,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}", file=sys.stderr)
        return None


def iter_releases(events_csv: Path):
    df = pd.read_csv(events_csv)
    for r in df.itertuples(index=False):
        acc = str(r.accession).replace("-", "")
        rel = CACHE_DIR / r.ticker / acc / "release.txt"
        if rel.exists():
            yield r.ticker, r.date, str(r.accession), rel


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="data/earnings_events.csv")
    parser.add_argument("--out", default="data/earnings_signals.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    MODEL = args.model

    work = list(iter_releases(Path(args.events)))
    if args.limit:
        work = work[: args.limit]

    out_path = Path(args.out)
    done = set()
    existing = []
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        existing = prev.to_dict(orient="records")
        done = {(r["ticker"], str(r["accession"])) for r in existing}
        work = [w for w in work if (w[0], w[2]) not in done]
        print(f"  resume: {len(done)} already done")

    if not args.confirm_cost:
        print(f"DRY RUN: would extract signals from {len(work)} earnings releases "
              f"(~${len(work)*0.004:.2f}, gpt-5-mini). Pass --confirm-cost to run.")
        return

    client = make_client()
    print(f"→ Extracting earnings signals from {len(work)} releases (concurrency={args.concurrency})")
    rows = list(existing)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}

    def worker(item):
        ticker, dt, acc, rel = item
        text = rel.read_text(encoding="utf-8", errors="replace")
        sig = grade_release(client, text)
        if sig is None:
            with lock:
                counters["fail"] += 1
            return None
        return {"ticker": ticker, "date": dt, "accession": acc, **sig.model_dump()}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(worker, w): w for w in work}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                with lock:
                    rows.append(r)
                    counters["ok"] += 1
            if n % 25 == 0 or n == len(work):
                with lock:
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"  {n}/{len(work)} · ok={counters['ok']} fail={counters['fail']}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"→ {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
