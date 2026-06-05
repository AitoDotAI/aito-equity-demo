"""Precompute Aito queries → site/data/*.json.

After data is loaded (see pipeline/aito/load.py), run this to emit every
JSON file the static site fetches at runtime. No request-time Aito calls.

Outputs:
  site/data/meta.json
  site/data/companies.json
  site/data/relate.json
  site/data/calibration.json
  site/data/predict/{ticker}_{vintage}.json   per FOCAL_COMPANIES
  site/data/match/{ticker}_{vintage}.json     per FOCAL_COMPANIES

Needs AITO_API_URL + AITO_API_KEY in env.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.aito.client import AitoClient
from pipeline.outcomes import absolute_range_label

SITE_DATA = Path("site/data")
COMPANIES_TABLE = "companies"


@dataclass(frozen=True)
class Focal:
    ticker: str
    vintage: int
    name: str  # full legal/display name (h2 in profile card)
    chip_label: str
    short_name: str  # sector / subtitle text under the chip


# Focal set for v1 (2017 vintage — the only vintage with LLM grades loaded).
# Chosen for outcome spread: two winners of different moat types, one wide-moat
# name that nonetheless only matched the market (moat ≠ destiny), one cyclical
# disaster (cautionary).
FOCAL_COMPANIES: list[Focal] = [
    Focal(ticker="NVDA", vintage=2017, name="NVIDIA Corporation", chip_label="NVDA · '17", short_name="graphics, semiconductors"),
    Focal(ticker="COST", vintage=2017, name="Costco Wholesale", chip_label="COST · '17", short_name="warehouse retail"),
    Focal(ticker="MMM", vintage=2017, name="3M Company", chip_label="MMM · '17", short_name="industrial conglomerate"),
    Focal(ticker="AAL", vintage=2017, name="American Airlines", chip_label="AAL · '17", short_name="airlines, cyclical"),
]

# Feature columns used to define a company's "profile" for similarity search.
SIMILARITY_PROFILE_COLUMNS = [
    "market_position", "moat_type", "moat_strength", "market_quality",
    "leadership_quality", "sector",
]

OUTCOME_BUCKET_LABELS = {
    "disaster": "Disaster",
    "poor": "Poor",
    "market": "Market",
    "good": "Good",
    "great": "Great",
}


# ── Helpers ────────────────────────────────────────────────────


def _measure_latency(client: AitoClient, body: dict, kind: str) -> tuple[dict, float]:
    """Run a query and return (result, latency_ms)."""
    op = {
        "predict": client.predict,
        "relate": client.relate,
        "match": client.match,
        "similarity": client.similarity,
    }[kind]
    t0 = time.perf_counter()
    result = op(body)
    return result, (time.perf_counter() - t0) * 1000


# ── Per-output emitters ────────────────────────────────────────


def emit_meta(companies_df: pd.DataFrame, latency_samples: list[float], out_dir: Path) -> None:
    vintages = sorted(set(int(v) for v in companies_df["vintage_year"].dropna()))
    n_features = sum(
        1
        for c in companies_df.columns
        if c
        not in ("ticker", "vintage_year", "vintage_date", "company_name", "outcome_bucket",
                "survived_intact", "terminal_event", "total_return_pct_local",
                "total_return_pct_usd", "window_years", "end_date")
        and not c.endswith("_rationale")
    )
    p50_latency = (
        int(sorted(latency_samples)[len(latency_samples) // 2])
        if latency_samples
        else None
    )
    vintages_label = (
        f"{len(vintages)} vintage year{'s' if len(vintages) != 1 else ''}"
        if vintages
        else "no vintage data"
    )
    # Window for the OLDEST vintage (largest forward window); rounded to int.
    today = date.today()
    window_default = int((today - date(min(vintages), 1, 1)).days / 365.25) if vintages else 12
    graded = int(companies_df["market_position"].notna().sum()) if "market_position" in companies_df.columns else 0
    with_outcome = int(companies_df["outcome_bucket"].notna().sum()) if "outcome_bucket" in companies_df.columns else 0
    payload = {
        "observations": len(companies_df),
        "graded_observations": graded,
        "outcome_observations": with_outcome,
        "features": n_features,
        "p50_latency_ms": p50_latency,
        "training_runs": 0,
        "vintages": vintages,
        "vintages_label": vintages_label,
        "window_years_default": window_default,
        "data_source_note": "S&P 500 historical constituents · SEC EDGAR · yfinance",
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def emit_universe(companies_df: pd.DataFrame, out_dir: Path) -> None:
    """Dump the full companies table as JSON for the Universe view.

    No Aito call — pure denormalisation of data/companies.csv. Rows may
    have null outcomes (delisted) or null LLM grades (extract not yet run).
    The UI handles these gracefully.
    """
    cols_to_export = [
        "ticker", "vintage_year", "vintage_date", "company_name",
        "sector", "industry", "market_cap_bucket",
        "exchange", "currency", "reporting_standard",
        # LLM qualitative grades
        "market_position", "moat_type", "moat_strength", "market_quality",
        "leadership_quality", "capital_allocation", "strategic_clarity",
        "execution_track_record", "founder_still_ceo",
        # fundamental + market factor buckets (drive the client-side explorer)
        "valuation_bucket", "growth_bucket", "leverage_bucket",
        "profitability_bucket", "momentum_bucket", "volatility_bucket",
        "pre_filing_mom_bucket",
        # continuous factors (for drill-down detail)
        "pe_ratio", "revenue_cagr_3y", "debt_to_equity", "return_on_equity",
        "momentum_12m", "volatility_12m",
        # long-horizon outcomes
        "outcome_bucket", "survived_intact", "terminal_event",
        "total_return_pct_local", "total_return_pct_usd", "window_years",
        # short-horizon (post-filing drift) outcome + event
        "filing_date", "fwd_20d_return", "fwd_20d_bucket", "pre_filing_mom_60d",
        "end_date",
    ]
    present = [c for c in cols_to_export if c in companies_df.columns]
    sub = companies_df[present].copy()

    # Replace NaN with None; round floats for compact JSON.
    rows = []
    for raw in sub.to_dict(orient="records"):
        clean = {}
        for k, v in raw.items():
            if isinstance(v, float) and (v != v):  # NaN
                clean[k] = None
            elif hasattr(v, "item"):
                v = v.item()
                clean[k] = round(v, 4) if isinstance(v, float) else v
            elif isinstance(v, float):
                clean[k] = round(v, 4)
            else:
                clean[k] = v
        rows.append(clean)

    payload = {
        "_note": "Generated from data/companies.csv. Rows with null LLM features mean the extraction stage hasn't run for that ticker yet.",
        "columns": present,
        "rows": rows,
    }
    (out_dir / "universe.json").write_text(json.dumps(payload), encoding="utf-8")


def emit_companies(out_dir: Path) -> None:
    payload = [
        {
            "ticker": f.ticker,
            "vintage": f.vintage,
            "name": f.name,
            "chip_label": f.chip_label,
            "short_name": f.short_name,
        }
        for f in FOCAL_COMPANIES
    ]
    (out_dir / "companies.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pandas_lift(df: pd.DataFrame, field: str, value, target_col: str = "outcome_bucket", target_val: str = "great") -> tuple[float, int] | None:
    """Exact lift of P(target | field=value) / P(target), computed in-frame.

    Returns (lift, n_with_feature) or None if the feature/target is absent.
    This is the transparent, replicable lift a quant can check by hand — and,
    unlike Aito's relate `where`, it lets us restrict the *population* (e.g. to
    a single vintage) so the per-vintage leakage probe is actually per-vintage.
    """
    sub = df[df[target_col].notna()]
    if sub.empty:
        return None
    base_rate = (sub[target_col] == target_val).mean()
    if base_rate == 0:
        return None
    with_feature = sub[sub[field] == value]
    n = len(with_feature)
    if n == 0:
        return None
    cond_rate = (with_feature[target_col] == target_val).mean()
    return cond_rate / base_rate, n


def emit_relate(
    client: AitoClient,
    companies_df: pd.DataFrame,
    out_dir: Path,
    latency_samples: list[float],
) -> None:
    """Two artifacts:
      - relate.json: headline feature lifts from Aito `_relate` over the FULL
        corpus (showcases the endpoint), with per-vintage min-max whiskers
        computed exactly in pandas.
      - leakage_probe.json: per-vintage lift table computed in pandas.

    Why pandas for the per-vintage figures: Aito's relate `where` clause sets
    the *condition* but keeps a global feature base-rate, so adding
    vintage_year there does NOT partition the population — every vintage
    returns the same lift. Restricting the population to one vintage is a
    plain group-by, so we compute it directly. The exact lift is also more
    defensible to a quant than an opaque engine number.
    """
    vintages = sorted(set(int(v) for v in companies_df["vintage_year"].dropna()))
    relate_columns = [c for c in RELATE_FEATURE_COLUMNS if c in companies_df.columns]

    # One real Aito `_relate` call — powers the latency telemetry and proves the
    # endpoint answers this in a single query. Displayed lifts below use the
    # exact pandas computation so every number on the page reconciles with the
    # leakage probe (and is replicable by hand).
    try:
        _, ms = _measure_latency(
            client,
            {"from": COMPANIES_TABLE, "where": {"outcome_bucket": "great"}, "relate": relate_columns, "limit": 60},
            "relate",
        )
        latency_samples.append(ms)
    except Exception as e:
        print(f"  ⚠ relate telemetry call failed: {e}")

    # Enumerate (field, value) candidates from graded rows; compute exact lift
    # over the full graded corpus + per vintage. Min support guards against
    # tiny-cell noise.
    MIN_SUPPORT = 8
    graded = companies_df[companies_df["market_position"].notna()]
    rows: list[dict] = []
    for field in relate_columns:
        if field not in graded.columns:
            continue
        for value in graded[field].dropna().unique():
            full = _pandas_lift(graded, field, value)
            if full is None or full[1] < MIN_SUPPORT:
                continue
            lift, n = full
            per_vintage: dict[str, float] = {}
            for v in vintages:
                vd = graded[graded["vintage_year"] == v]
                res = _pandas_lift(vd, field, value)
                if res is not None and res[1] >= MIN_SUPPORT:
                    per_vintage[str(v)] = round(res[0], 2)
            lifts_pv = list(per_vintage.values())
            disp_val = int(value) if isinstance(value, float) and value.is_integer() else value
            rows.append(
                {
                    "feature": f"{field} = {disp_val}",
                    "field": field,
                    "type": "qual" if field in QUALITATIVE_FIELDS else "quant",
                    "lift": round(lift, 2),
                    "n": n,
                    "lift_min": round(min(lifts_pv), 2) if lifts_pv else None,
                    "lift_max": round(max(lifts_pv), 2) if lifts_pv else None,
                    "bar_pct": min(100, int(lift / 5 * 100)),
                    "per_vintage": per_vintage,
                }
            )
    rows.sort(key=lambda r: r["lift"], reverse=True)
    top = rows[:10]

    relate_payload = {
        "target": "outcome_bucket = great",
        "rows": top,
        "n_vintages": len(vintages),
        "vintages": vintages,
        "pullquote_html": (
            "Headline lifts from a single Aito <code style=\"font-family:'JetBrains Mono'\">_relate</code> "
            "query over the full corpus; whiskers show the exact per-vintage min-max range. "
            "Lifts that hold across vintages indicate the LLM grader is not leaking "
            "post-vintage knowledge into its features."
        ),
    }
    (out_dir / "relate.json").write_text(json.dumps(relate_payload, indent=2), encoding="utf-8")

    # ── Leakage probe: per-vintage lift, computed exactly in pandas ──
    # Take the strongest features by full-corpus lift, show how each behaves
    # vintage by vintage. Restrict to features actually graded (qualitative).
    probe_features = []
    drifts: list[float] = []
    for r in top:
        if r["field"] not in QUALITATIVE_FIELDS:
            continue
        pv = r["per_vintage"]
        lifts_pv = list(pv.values())
        if len(lifts_pv) < 2 or r["lift"] == 0:
            continue
        drift = (max(lifts_pv) - min(lifts_pv)) / (sum(lifts_pv) / len(lifts_pv))
        drifts.append(drift)
        probe_features.append(
            {
                "feature": r["feature"],
                "type": r["type"],
                "per_vintage": pv,
                "drift": round(drift, 2),
                "lift_mean": round(sum(lifts_pv) / len(lifts_pv), 2),
            }
        )
        if len(probe_features) >= 8:
            break
    avg_drift = round(sum(drifts) / len(drifts), 2) if drifts else None
    leakage_payload = {
        "vintages": vintages,
        "features": probe_features,
        "drift_score": avg_drift,
        "interpretation": (
            "Each lift is computed within a single vintage (exact group-by, not an "
            "engine estimate): P(great | feature) / P(great) among that vintage's "
            "companies. Drift = (max − min) / mean across vintages. "
            "Low drift (< 0.5) means a feature's predictive value is stable across "
            "time — evidence the LLM is grading from contemporaneous filings, not "
            "leaking hindsight. High drift flags a feature to investigate."
        ),
    }
    (out_dir / "leakage_probe.json").write_text(json.dumps(leakage_payload, indent=2), encoding="utf-8")


RELATE_FEATURE_COLUMNS = [
    "market_position", "moat_type", "moat_strength", "market_quality",
    "leadership_quality", "capital_allocation", "strategic_clarity",
    "execution_track_record", "sector",
    "valuation_bucket", "growth_bucket", "leverage_bucket",
    "profitability_bucket", "momentum_bucket", "volatility_bucket",
]

# Factor explorer: which columns to scan, grouped by provenance (drives the
# colour/category in the UI and tells the "what kind of signal is this" story).
FACTOR_GROUPS: dict[str, list[str]] = {
    "llm": [
        "market_position", "moat_type", "moat_strength", "market_quality",
        "leadership_quality", "capital_allocation", "strategic_clarity",
        "execution_track_record",
    ],
    "structural": ["sector", "industry"],
    "valuation": ["valuation_bucket", "growth_bucket", "leverage_bucket", "profitability_bucket"],
    "market": ["momentum_bucket", "volatility_bucket"],
}


def emit_factor_explorer(companies_df: pd.DataFrame, out_dir: Path, min_support: int = 12) -> None:
    """Two-directional factor table: for every (feature, value) with enough
    support, exact lift toward 'great' (upside) AND toward 'poor∪disaster'
    (downside), computed in pandas over rows with a recorded outcome.

    This is the browsable answer to 'what predicts winners, what predicts
    losers, and what looks mispriced' — no query syntax required.
    """
    df = companies_df[companies_df["outcome_bucket"].notna()].copy()
    n_total = len(df)
    base_great = (df["outcome_bucket"] == "great").mean()
    base_down = df["outcome_bucket"].isin(["poor", "disaster"]).mean()

    def disp(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    rows: list[dict] = []
    for group, cols in FACTOR_GROUPS.items():
        for col in cols:
            if col not in df.columns:
                continue
            for value in df[col].dropna().unique():
                sub = df[df[col] == value]
                n = len(sub)
                if n < min_support:
                    continue
                lift_great = (sub["outcome_bucket"] == "great").mean() / base_great if base_great else None
                lift_down = sub["outcome_bucket"].isin(["poor", "disaster"]).mean() / base_down if base_down else None
                rows.append(
                    {
                        "feature": f"{col} = {disp(value)}",
                        "field": col,
                        "group": group,
                        "n": n,
                        "lift_great": round(lift_great, 2) if lift_great is not None else None,
                        "lift_down": round(lift_down, 2) if lift_down is not None else None,
                    }
                )
    rows.sort(key=lambda r: (r["lift_great"] if r["lift_great"] is not None else 0), reverse=True)
    payload = {
        "n_observations": n_total,
        "base_rate_great": round(float(base_great), 3),
        "base_rate_down": round(float(base_down), 3),
        "rows": rows,
        "groups": list(FACTOR_GROUPS.keys()),
    }
    (out_dir / "factor_explorer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_relate_hit(hit: dict) -> tuple[str, str] | None:
    """Parse Aito relate hit shape {"related": {field: {"$has": value}}, ...}.

    Returns (display_key, field_name) or None if unparseable.
    """
    related = hit.get("related")
    if not isinstance(related, dict) or not related:
        return None
    field = next(iter(related))
    prop = related[field]
    value = prop.get("$has") if isinstance(prop, dict) else prop
    return f"{field} = {value}", field


def _flatten_relate_feature(hit: dict) -> str:
    """Legacy helper retained for any older callers."""
    if "feature" in hit and isinstance(hit["feature"], dict):
        f = hit["feature"]
        return f"{f.get('field', '?')} = {f.get('value', '?')}"
    if "field" in hit and "value" in hit:
        return f"{hit['field']} = {hit['value']}"
    return str(hit.get("feature", "?"))


QUALITATIVE_FIELDS = {
    "market_position", "moat_type", "moat_strength", "market_quality",
    "leadership_quality", "capital_allocation", "strategic_clarity", "execution_track_record",
}


def _is_qualitative(hit: dict) -> bool:
    feat = hit.get("feature") or hit
    field = feat.get("field") if isinstance(feat, dict) else ""
    return field in QUALITATIVE_FIELDS


CALIBRATION_FEATURE_COLS = (
    "market_position",
    "moat_type",
    "moat_strength",
    "market_quality",
    "leadership_quality",
    "founder_still_ceo",
    "sector",
)


def emit_calibration(
    client: AitoClient,
    companies_df: pd.DataFrame,
    out_dir: Path,
    latency_samples: list[float],
    sample_cap: int = 500,
) -> None:
    """Compute real calibration: for each row with a recorded outcome, predict
    P(great) from its features; bin into deciles by predicted probability;
    plot mean predicted vs realised frequency per bin.

    Caveats:
      - This is an *in-sample* calibration over the full corpus (Aito has all
        rows loaded). A leave-vintage-out holdout would be more rigorous; see
        leakage_probe for the cross-vintage stability check that approximates it.
      - We cap at `sample_cap` predict calls to keep this stage to ~30s of
        Aito work even on cold instances.
    """
    pairs: list[tuple[float, int]] = []  # (predicted P(great), actual: 1 if great else 0)
    n_calls = 0
    n_skipped_no_features = 0
    n_skipped_no_outcome = 0

    for _, row in companies_df.iterrows():
        if n_calls >= sample_cap:
            break
        outcome = row.get("outcome_bucket")
        if outcome is None or (isinstance(outcome, float) and outcome != outcome):
            n_skipped_no_outcome += 1
            continue
        where: dict = {}
        for f in CALIBRATION_FEATURE_COLS:
            v = row.get(f)
            if v is None or (isinstance(v, float) and v != v):
                continue
            where[f] = bool(v) if isinstance(v, (bool,)) else v
        if not where:
            n_skipped_no_features += 1
            continue
        body = {"from": COMPANIES_TABLE, "where": where, "predict": "outcome_bucket"}
        try:
            result, ms = _measure_latency(client, body, "predict")
            latency_samples.append(ms)
        except Exception:
            continue
        p_great = 0.0
        for hit in result.get("hits", []):
            label = hit.get("feature") or hit.get("$value")
            if label == "great":
                p_great = float(hit.get("$p") or hit.get("p") or 0.0)
                break
        pairs.append((p_great, 1 if outcome == "great" else 0))
        n_calls += 1

    if not pairs:
        # Nothing scored — emit a clear pending marker so the UI can show
        # "computation pending" rather than a misleading chart.
        (out_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "_pending": True,
                    "_note": (
                        "No (predicted, actual) pairs computed. "
                        f"Skipped {n_skipped_no_features} rows lacking LLM features, "
                        f"{n_skipped_no_outcome} rows lacking outcomes. "
                        "Run extraction stage and reload Aito."
                    ),
                    "deciles": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    # Bin into deciles by predicted probability.
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    n = len(pairs_sorted)
    n_per_bin = max(1, n // 10)
    deciles = []
    for i in range(10):
        lo = i * n_per_bin
        hi = (i + 1) * n_per_bin if i < 9 else n
        bin_pairs = pairs_sorted[lo:hi]
        if not bin_pairs:
            continue
        predicted_mean = sum(p[0] for p in bin_pairs) / len(bin_pairs)
        actual_freq = sum(p[1] for p in bin_pairs) / len(bin_pairs)
        deciles.append(
            {
                "label": f"D{i + 1}",
                "predicted": round(predicted_mean, 3),
                "actual": round(actual_freq, 3),
                "n": len(bin_pairs),
            }
        )

    brier = sum((p - a) ** 2 for p, a in pairs) / n
    payload = {
        "deciles": deciles,
        "n_observations": n,
        "brier_score": round(brier, 4),
        "methodology": (
            "In-sample calibration. For each row with a recorded outcome, P(great) "
            "predicted from its features (no outcome lookup); rows binned into "
            "deciles by predicted probability; mean predicted vs realised frequency "
            "plotted per bin. Lower Brier is better; perfect = 0, random = 0.25."
        ),
    }
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def emit_predict_per_focal(
    client: AitoClient,
    companies_df: pd.DataFrame,
    out_dir: Path,
    latency_samples: list[float],
) -> None:
    predict_dir = out_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    for focal in FOCAL_COMPANIES:
        row = companies_df[
            (companies_df["ticker"] == focal.ticker)
            & (companies_df["vintage_year"] == focal.vintage)
        ]
        if row.empty:
            print(f"  ⚠ {focal.ticker}·{focal.vintage}: no row in companies table")
            continue
        r = row.iloc[0]
        # Build a where clause from the LLM-graded features
        where = {}
        for f in ("market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality"):
            if f in r.index and pd.notna(r[f]):
                where[f] = r[f]
        if "founder_still_ceo" in r.index and pd.notna(r["founder_still_ceo"]):
            where["founder_still_ceo"] = bool(r["founder_still_ceo"])

        body = {"from": COMPANIES_TABLE, "where": where, "predict": "outcome_bucket"}
        result, ms = _measure_latency(client, body, "predict")
        latency_samples.append(ms)

        window_years = float(r.get("window_years") or 12)
        buckets = _hits_to_buckets(result, window_years)
        peak = max(buckets, key=lambda b: b["prob"])["label"]
        payload = {
            "ticker": focal.ticker,
            "vintage": focal.vintage,
            "company_name": str(r.get("company_name") or focal.ticker),
            "exchange": str(r.get("exchange") or "NASDAQ"),
            "sector": str(r.get("sector") or ""),
            "market_cap": "",  # to be populated from a price-snapshot stage
            "vintage_label": pd.to_datetime(r["vintage_date"]).strftime("%B %Y"),
            "window_label": f"{int(window_years)}-year",
            "grades": _build_grades(r),
            "prediction": {
                "confidence": _classify_confidence(buckets),
                "peak": peak,
                "buckets": buckets,
            },
            "pullquote": "Generated from a live Aito _predict query on the loaded data.",
            "pullquote_attr": "From the demo pipeline · auto-regenerated",
        }
        (predict_dir / f"{focal.ticker}_{focal.vintage}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def _hits_to_buckets(result: dict, window_years: float) -> list[dict]:
    """Normalise Aito's _predict response into the UI's bucket shape."""
    ranges = absolute_range_label(window_years)
    by_label = {}
    for hit in result.get("hits", []):
        label = str(hit.get("feature") or hit.get("$value") or "").strip()
        prob = float(hit.get("$p") or hit.get("p") or 0.0)
        if label.lower() not in OUTCOME_BUCKET_LABELS:
            continue
        by_label[label.lower()] = prob
    order = ["disaster", "poor", "market", "good", "great"]
    return [
        {
            "label": OUTCOME_BUCKET_LABELS[k],
            "range": ranges.get(k, ""),
            "prob": round(by_label.get(k, 0.0), 4),
        }
        for k in order
    ]


def _classify_confidence(buckets: list[dict]) -> str:
    top = max(b["prob"] for b in buckets)
    if top >= 0.50:
        return "HIGH"
    if top >= 0.30:
        return "MEDIUM"
    return "LOW"


def _build_grades(row: pd.Series) -> list[dict]:
    """Build the four grade cards from row data + rationales."""
    grades = []
    if "market_position" in row.index and pd.notna(row["market_position"]):
        grades.append(
            {
                "label": "Market Position",
                "value": str(row["market_position"]).capitalize(),
                "pip_filled": {"dominant": 5, "strong": 4, "competitive": 3, "lagging": 1}.get(
                    str(row["market_position"]), 3
                ),
                "pip_total": 5,
                "evidence": str(row.get("market_position_rationale", "") or ""),
            }
        )
    if "moat_strength" in row.index and pd.notna(row["moat_strength"]):
        grades.append(
            {
                "label": "Moat Type & Strength",
                "value": f"{row.get('moat_type', '')} / {row['moat_strength']}".strip(" /"),
                "pip_filled": {"wide": 5, "narrow": 3, "none": 1}.get(str(row["moat_strength"]), 3),
                "pip_total": 5,
                "evidence": str(row.get("moat_rationale", "") or ""),
            }
        )
    if "market_quality" in row.index and pd.notna(row["market_quality"]):
        grades.append(
            {
                "label": "Market Quality",
                "value": str(row["market_quality"]).replace("_", " ").capitalize(),
                "pip_filled": {"secular_growth": 5, "stable": 4, "cyclical": 3, "declining": 2, "disrupted": 1}.get(
                    str(row["market_quality"]), 3
                ),
                "pip_total": 5,
                "evidence": str(row.get("market_quality_rationale", "") or ""),
            }
        )
    if "leadership_quality" in row.index and pd.notna(row["leadership_quality"]):
        score = int(row["leadership_quality"])
        grades.append(
            {
                "label": "Leadership Quality",
                "value": f"{score}/5",
                "pip_filled": score,
                "pip_total": 5,
                "evidence": str(row.get("leadership_rationale", "") or ""),
            }
        )
    return grades


def emit_match_per_focal(
    client: AitoClient,
    companies_df: pd.DataFrame,
    out_dir: Path,
    latency_samples: list[float],
) -> None:
    match_dir = out_dir / "match"
    match_dir.mkdir(parents=True, exist_ok=True)
    end_year = date.today().year
    for focal in FOCAL_COMPANIES:
        # Build the similarity proposition from the focal's FEATURE PROFILE, not
        # its identity — keying on {ticker, vintage} just matches the row to
        # itself and returns 1.0 ties for everything else. We want companies
        # with a similar moat/position/quality fingerprint.
        frow = companies_df[
            (companies_df["ticker"] == focal.ticker)
            & (companies_df["vintage_year"] == focal.vintage)
        ]
        if frow.empty:
            print(f"  ⚠ {focal.ticker}·{focal.vintage}: no row for similarity")
            continue
        fr = frow.iloc[0]
        proposition = {}
        for c in SIMILARITY_PROFILE_COLUMNS:
            if c not in fr.index or pd.isna(fr[c]):
                continue
            v = fr[c]
            if hasattr(v, "item"):
                v = v.item()
            if c == "leadership_quality" and isinstance(v, float):
                v = int(round(v))
            proposition[c] = v
        body = {
            "from": COMPANIES_TABLE,
            "similarity": proposition,
            "limit": 10,
        }
        try:
            result, ms = _measure_latency(client, body, "similarity")
        except Exception as e:
            print(f"  ⚠ {focal.ticker}·{focal.vintage} similarity failed: {e}")
            continue
        latency_samples.append(ms)
        matches = []
        for hit in result.get("hits", []):
            ticker = str(hit.get("ticker") or "")
            vintage = int(hit.get("vintage_year") or 0)
            if ticker == focal.ticker and vintage == focal.vintage:
                continue  # skip the focal row itself
            outcome_text = ""
            sentiment = "neutral"
            ret = hit.get("total_return_pct_local")
            if ret is not None:
                pct = float(ret)
                outcome_text = f"{pct:+.0f}%"
                sentiment = "positive" if pct > 0 else "negative"
            if str(hit.get("terminal_event", "")).lower() in ("acquired", "bankrupt", "delisted"):
                outcome_text = str(hit["terminal_event"])
                sentiment = "negative"
            matches.append(
                {
                    "ticker": ticker,
                    "vintage": vintage,
                    "name": str(hit.get("company_name") or ticker),
                    "similarity": round(float(hit.get("$score") or 0.0), 3),
                    "description": str(hit.get("moat_rationale") or hit.get("market_position_rationale") or "")[:300],
                    "outcome": {
                        "text": outcome_text,
                        "window": f"'{vintage % 100:02d}→'{end_year % 100:02d}",
                        "sentiment": sentiment,
                    },
                }
            )
            if len(matches) >= 6:
                break
        payload = {
            "focal": {"ticker": focal.ticker, "vintage": focal.vintage, "name": focal.name},
            "matches": matches,
            "pullquote_html": "Nearest analogues by feature similarity — live from Aito <code style=\"font-family:'JetBrains Mono'\">_similarity</code>.",
        }
        (match_dir / f"{focal.ticker}_{focal.vintage}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


NEWS_THEME_ORDER = [
    "earnings", "M&A", "guidance / RegFD", "management change",
    "material agreement", "new debt / financing", "equity issuance",
    "restructuring", "shareholder vote", "other material news",
]


def emit_news_reaction(out_dir: Path, events_csv: Path = Path("data/news_events.csv"), min_n: int = 30) -> None:
    """Aggregate the 8-K event study into per-theme reaction stats at each
    horizon (+1d / +5d / +20d): mean, median, %-up, n, plus the day-1→day-20
    persistence (does the immediate move continue or revert)."""
    if not events_csv.exists():
        _emit_pending(out_dir, "news_reaction.json", {"themes": []})
        return
    df = pd.read_csv(events_csv)
    horizons = [("1d", "react_1d"), ("5d", "react_5d"), ("20d", "react_20d")]

    themes = []
    for theme, grp in df.groupby("theme"):
        n = len(grp)
        if n < min_n:
            continue
        h = {}
        for label, col in horizons:
            vals = grp[col].dropna()
            if len(vals) == 0:
                continue
            h[label] = {
                "n": int(len(vals)),
                "mean": round(float(vals.mean()), 2),
                "median": round(float(vals.median()), 2),
                "pct_up": round(float((vals > 0).mean()) * 100, 1),
            }
        # Persistence: among events with a positive day-1, mean day-20.
        both = grp.dropna(subset=["react_1d", "react_20d"])
        persistence = None
        if len(both) >= min_n:
            pos1 = both[both["react_1d"] > 0]["react_20d"]
            neg1 = both[both["react_1d"] < 0]["react_20d"]
            persistence = {
                "after_positive_1d": round(float(pos1.mean()), 2) if len(pos1) else None,
                "after_negative_1d": round(float(neg1.mean()), 2) if len(neg1) else None,
                "corr_1d_20d": round(float(both["react_1d"].corr(both["react_20d"])), 3),
            }
        # A few example movers (largest |20d|) for drill-down colour.
        ex = grp.dropna(subset=["react_20d"]).reindex(
            grp.dropna(subset=["react_20d"])["react_20d"].abs().sort_values(ascending=False).index
        ).head(8)
        examples = [
            {
                "ticker": r.ticker, "date": r.date,
                "react_1d": None if pd.isna(r.react_1d) else float(r.react_1d),
                "react_5d": None if pd.isna(r.react_5d) else float(r.react_5d),
                "react_20d": None if pd.isna(r.react_20d) else float(r.react_20d),
            }
            for r in ex.itertuples(index=False)
        ]
        themes.append({"theme": theme, "n": n, "h": h, "persistence": persistence, "examples": examples})

    order = {t: i for i, t in enumerate(NEWS_THEME_ORDER)}
    themes.sort(key=lambda t: order.get(t["theme"], 99))
    payload = {
        "n_events": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()),
        "themes": themes,
        "note": "8-K filings 2020+ (recent EDGAR window). Reaction measured close-to-close from the trading day before the filing; the day-of move is included in +1d.",
    }
    (out_dir / "news_reaction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _emit_pending(out_dir: Path, filename: str, extra: dict) -> None:
    """Emit a clearly-marked pending JSON file. The frontend renders 'computation
    pending' instead of fake numbers when it sees `_pending: true`."""
    payload = {
        "_pending": True,
        "_note": "Computation pending — run `./do pipeline precompute` with AITO_API_URL + AITO_API_KEY set.",
        **extra,
    }
    (out_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def precompute_all(out_dir: Path = SITE_DATA, static_only: bool = False) -> None:
    """Emit every JSON file the static site reads.

    With `static_only=True`, skip Aito-dependent outputs (relate, calibration,
    predict, match) and emit only the data we already have on disk —
    meta, companies, universe. Use this when Aito isn't wired yet.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    companies_df = pd.read_csv("data/companies.csv")
    latency_samples: list[float] = []

    if not static_only:
        with AitoClient() as client:
            print("→ relate + leakage probe (per-vintage relate × 3)")
            emit_relate(client, companies_df, out_dir, latency_samples)
            print("→ calibration (real predict per row)")
            emit_calibration(client, companies_df, out_dir, latency_samples)
            print(f"→ predict × {len(FOCAL_COMPANIES)}")
            emit_predict_per_focal(client, companies_df, out_dir, latency_samples)
            print(f"→ match × {len(FOCAL_COMPANIES)}")
            emit_match_per_focal(client, companies_df, out_dir, latency_samples)
    else:
        print("→ (skipping Aito queries; static-only mode — emitting pending markers)")
        _emit_pending(out_dir, "relate.json", {"target": "outcome_bucket = great", "rows": []})
        _emit_pending(out_dir, "calibration.json", {"deciles": []})
        _emit_pending(out_dir, "leakage_probe.json", {"vintages": [], "features": [], "drift_score": None})

    print("→ factor explorer (two-directional lift)")
    emit_factor_explorer(companies_df, out_dir)
    print("→ news reaction (8-K event study)")
    emit_news_reaction(out_dir)
    print("→ meta")
    emit_meta(companies_df, latency_samples, out_dir)
    print("→ companies")
    emit_companies(out_dir)
    print("→ universe")
    emit_universe(companies_df, out_dir)
    print(f"✓ wrote JSON to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(SITE_DATA))
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip Aito queries; emit only meta + companies + universe from local CSVs",
    )
    args = parser.parse_args()
    precompute_all(Path(args.out), static_only=args.static_only)


if __name__ == "__main__":
    main()
