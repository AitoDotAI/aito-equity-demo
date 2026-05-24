"""Extract relevant sections from 10-K and DEF 14A plain text.

Goal: produce a token-bounded "excerpt" the LLM grader actually needs,
rather than feeding entire filings (50K-200K tokens) into every call.

10-K sections we want:
  - Item 1   "Business"
  - Item 1A  "Risk Factors"
  - Item 7   "Management's Discussion and Analysis"

DEF 14A sections we want:
  - Compensation Discussion and Analysis (CD&A)
  - Director / executive officer biographies

This is best-effort regex-based extraction. Modern filings are mostly
well-formed but layout varies. If section markers aren't found, fall
back to a head-truncation of the document.

Token budget is enforced by chars-per-token approximation (4 chars ≈ 1
token for English prose). Truncation is hard but split-aware: we keep
all of the smaller sections and trim the largest one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4  # rough English-prose approximation

# 10-K section headers. Variants observed in real filings:
#   "Item 1. Business"  /  "ITEM 1. BUSINESS"  /  "Item 1 Business"
TEN_K_SECTIONS: list[tuple[str, str]] = [
    ("item_1", r"(?im)^\s*item\s*1\.?\s+business\b"),
    ("item_1a", r"(?im)^\s*item\s*1a\.?\s+risk\s+factors\b"),
    ("item_7", r"(?im)^\s*item\s*7\.?\s+management.s\s+discussion"),
]
# Sentinel: the start of Item 1B is the end of Item 1A.
TEN_K_BOUNDARIES = {
    "item_1": r"(?im)^\s*item\s*1a\.?\s+",
    "item_1a": r"(?im)^\s*item\s*1b\.?\s+",
    "item_7": r"(?im)^\s*item\s*7a\.?\s+",
}

DEF_14A_HEADERS = [
    ("cd_a", r"(?im)compensation\s+discussion\s+and\s+analysis"),
    ("biographies", r"(?im)(?:director|executive\s+officer)s?\s+(?:biographies|of\s+the\s+registrant)"),
]


@dataclass(frozen=True)
class Excerpt:
    filing_type: str
    sections: dict[str, str]
    total_chars: int

    def to_prompt_text(self, max_tokens: int) -> str:
        """Concatenate sections, budget-trimmed to `max_tokens`."""
        budget_chars = max_tokens * CHARS_PER_TOKEN
        if self.total_chars <= budget_chars:
            return self._concat()

        # Trim the largest section to fit; keep smaller ones whole.
        sizes = {k: len(v) for k, v in self.sections.items()}
        sorted_small_first = sorted(sizes.items(), key=lambda kv: kv[1])

        kept: dict[str, str] = {}
        remaining = budget_chars
        for k, sz in sorted_small_first[:-1]:
            kept[k] = self.sections[k]
            remaining -= sz

        # Last (largest) section gets whatever's left.
        last_key = sorted_small_first[-1][0]
        last_text = self.sections[last_key]
        kept[last_key] = last_text[: max(0, remaining)] + (
            "\n\n[…section truncated for token budget…]"
            if remaining < len(last_text)
            else ""
        )
        return self._concat(kept)

    def _concat(self, sections: dict[str, str] | None = None) -> str:
        sections = sections or self.sections
        out = []
        for k, v in sections.items():
            out.append(f"========== {self.filing_type} · {k} ==========\n{v.strip()}\n")
        return "\n".join(out)


def _last_match(pattern: str, text: str, start: int = 0) -> re.Match | None:
    """Return the last match of `pattern` in `text[start:]`.

    The first occurrence of section headers is usually the table of contents
    (which is empty body); the *last* occurrence is the actual section. We
    iterate finditer() and keep the last hit.
    """
    last = None
    for m in re.finditer(pattern, text[start:]):
        last = m
    return last


def extract_10k_sections(text: str) -> Excerpt:
    """Take two contiguous windows from a 10-K: post-TOC head, then MD&A.

    Regex-based "find Item 1A" extraction is brittle on real filings (TOC-only
    markers, "Risk Factors (Continued)" pagination, table-of-contents-only
    items). The cover + TOC live in the first ~8K chars; Items 1 and 1A
    typically sit between 8K-50K chars; Item 7 (MD&A) anywhere from 60K-130K
    chars in.

    Two windows let us cover business + risks + MD&A without buying the whole
    filing into context. Budget enforcement happens in `to_prompt_text`.
    """
    sections: dict[str, str] = {}

    # Window 1: post-TOC head — covers Items 1 (Business) + 1A (Risk Factors).
    if len(text) > 9_000:
        sections["business_and_risks"] = text[8_000 : min(50_000, len(text))]

    # Window 2: MD&A — anchored on the first body-level heading we find past
    # the TOC, with a fallback to the canonical position.
    mda_anchor = re.search(
        r"(?im)^\s*management.s\s+discussion\s+and\s+analysis", text[50_000:]
    )
    if mda_anchor:
        mda_start = 50_000 + mda_anchor.start()
        sections["mda"] = text[mda_start : mda_start + 30_000]
    elif len(text) > 90_000:
        sections["mda"] = text[80_000 : min(110_000, len(text))]

    if not sections:
        sections["head"] = text[:40_000]
    return Excerpt(
        filing_type="10-K",
        sections=sections,
        total_chars=sum(len(v) for v in sections.values()),
    )


def extract_def14a_sections(text: str) -> Excerpt:
    sections: dict[str, str] = {}
    for key, header_re in DEF_14A_HEADERS:
        m = _last_match(header_re, text)
        if not m:
            continue
        # Take 20K chars after the header (sections aren't well bounded).
        body = text[m.start() : m.start() + 20_000]
        if len(body) >= 1000:
            sections[key] = body
    if not sections:
        sections["head"] = text[:30_000]
    return Excerpt(
        filing_type="DEF 14A",
        sections=sections,
        total_chars=sum(len(v) for v in sections.values()),
    )


def excerpt_for_filing(filing_type: str, text: str) -> Excerpt:
    """Dispatch based on filing type."""
    if filing_type == "10-K":
        return extract_10k_sections(text)
    if filing_type == "DEF 14A":
        return extract_def14a_sections(text)
    return Excerpt(
        filing_type=filing_type,
        sections={"head": text[:60_000]},
        total_chars=min(len(text), 60_000),
    )
