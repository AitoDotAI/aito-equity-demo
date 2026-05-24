"""LLM extraction: 10-K + proxy → qualitative grades + rationale strings.

For each (ticker, vintage):
  1. Fetch filings via FilingFetcher (10-K + DEF 14A filed before vintage_date)
  2. For each of the 4 features (market_position, moat, market_quality, leadership),
     call Claude Sonnet 4.6 with the per-feature prompt at temperature 0.3,
     three times; take the modal categorical answer, retain the first run's
     rationale string
  3. Append to data/llm_features.csv

Cost guard: ~250 companies × 4 features × 3 runs × ~3K tokens ≈ 9M tokens.
Roughly $30-50 with Sonnet. Switch to Haiku for the cheaper grades
(market_position, market_quality) if budget tightens.

Critical: prompts must include ONLY the documents filed BEFORE vintage_date,
and end with the lookahead-bias guardrail (see prompts/*.md). This is not a
perfect defence against LLM training-data leakage, but it's documentable and
the multi-vintage lift comparison serves as an empirical leakage probe.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pipeline.filings.base import Filing

PROMPTS_DIR = Path(__file__).parent / "prompts"
FEATURES = ("market_position", "moat", "market_quality", "leadership")
FeatureName = Literal["market_position", "moat", "market_quality", "leadership"]


@dataclass(frozen=True)
class FeatureGrade:
    feature: FeatureName
    value: str  # the categorical answer
    rationale: str  # one or two sentences quoting the filings
    runs: int  # how many model calls were made (modal aggregation)


def load_prompt(feature: FeatureName) -> str:
    return (PROMPTS_DIR / f"{feature}.md").read_text(encoding="utf-8")


def modal(answers: list[str]) -> str:
    """Return the most common answer; ties broken by first occurrence."""
    if not answers:
        raise ValueError("modal() on empty list")
    counter = Counter(answers)
    top = counter.most_common(1)[0][0]
    return top


def grade_feature(
    feature: FeatureName,
    filings: list[Filing],
    vintage_date: str,
    n_runs: int = 3,
    model: str = "claude-sonnet-4-6",
) -> FeatureGrade:
    """Run modal-of-3 grading for one feature on one company."""
    raise NotImplementedError(
        "grade_feature pending — Anthropic SDK wiring in next pipeline iteration. "
        "See aito-equity-demo-TASK.md → Data Pipeline → Qualitative features."
    )
