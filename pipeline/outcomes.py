"""Outcome bucketing — window-aware annualised return thresholds.

The reference HTML's labels (disaster −50%+, poor −50–0%, market 0–100%,
good 100–300%, great 300%+) were tuned to a 12-year window. For other
windows those absolute thresholds break (a 6-year 300% gain compounds at
26%/yr; a 12-year 300% gain compounds at 12%/yr — very different).

This module computes the bucket from the COMPOUND ANNUAL GROWTH RATE.
Display labels per window are emitted by the JSON-precompute stage so the
UI shows the right "Great >= 300%" / "Great >= 100%" string for a given
focal company's window.

Documented in ADR 0003.
"""

from __future__ import annotations

from typing import Literal

OutcomeBucket = Literal["disaster", "poor", "market", "good", "great"]

# Annualised CAGR thresholds. The "market" boundary tracks long-run real
# US equity return; "great" sits at roughly the rate a top-decile compounder
# sustains over a decade-plus window.
THRESHOLDS_CAGR: list[tuple[OutcomeBucket, float]] = [
    ("disaster", -0.10),
    ("poor", 0.00),
    ("market", 0.07),
    ("good", 0.13),
    ("great", float("inf")),
]


def cagr_from_total_return(total_return_pct: float, window_years: float) -> float:
    if window_years <= 0:
        raise ValueError(f"window_years must be positive (got {window_years})")
    return (1 + total_return_pct / 100) ** (1 / window_years) - 1


def bucket_for_return(total_return_pct: float, window_years: float) -> OutcomeBucket:
    cagr = cagr_from_total_return(total_return_pct, window_years)
    for label, ceiling in THRESHOLDS_CAGR:
        if cagr < ceiling:
            return label
    return "great"


def absolute_range_label(window_years: float) -> dict[OutcomeBucket, str]:
    """Return human-readable absolute-return ranges per bucket for a given window.

    Used by the JSON-precompute stage so the UI shows ranges that match the
    horizon (e.g. for a 12-year window 'Great >= 300%', for a 6-year window
    'Great >= 110%').
    """
    labels: dict[OutcomeBucket, str] = {}
    # invert: at boundary CAGR c over window y, total return = (1+c)^y - 1
    prev_pct: float | None = None
    for label, ceiling in THRESHOLDS_CAGR:
        if ceiling == float("inf"):
            labels[label] = f">= {prev_pct:+.0f}%" if prev_pct is not None else "any"
            continue
        pct = ((1 + ceiling) ** window_years - 1) * 100
        if prev_pct is None:
            labels[label] = f"< {pct:+.0f}%"
        else:
            labels[label] = f"{prev_pct:+.0f}% to {pct:+.0f}%"
        prev_pct = pct
    return labels
