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


FOCAL_COMPANIES: list[Focal] = [
    Focal(ticker="NVDA", vintage=2014, name="NVIDIA Corporation", chip_label="NVDA · '14", short_name="graphics, semiconductors"),
    Focal(ticker="SHLD", vintage=2014, name="Sears Holdings", chip_label="SHLD · '14", short_name="Sears Holdings"),
    Focal(ticker="COST", vintage=2017, name="Costco Wholesale", chip_label="COST · '17", short_name="Costco Wholesale"),
    Focal(ticker="META", vintage=2020, name="Meta Platforms", chip_label="META · '20", short_name="platforms, advertising"),
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
    op = {"predict": client.predict, "relate": client.relate, "match": client.match}[kind]
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
    payload = {
        "observations": len(companies_df),
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
        "market_position", "moat_type", "moat_strength", "market_quality",
        "leadership_quality", "founder_still_ceo",
        "outcome_bucket", "survived_intact", "terminal_event",
        "total_return_pct_local", "total_return_pct_usd", "window_years",
        "end_date",
    ]
    present = [c for c in cols_to_export if c in companies_df.columns]
    sub = companies_df[present].copy()

    # Replace NaN with None for clean JSON
    rows = []
    for raw in sub.to_dict(orient="records"):
        clean = {}
        for k, v in raw.items():
            if isinstance(v, float) and (v != v):  # NaN
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
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


def emit_relate(client: AitoClient, out_dir: Path, latency_samples: list[float]) -> None:
    body = {
        "from": COMPANIES_TABLE,
        "where": {"outcome_bucket": "great"},
        "relate": "$features",
        "limit": 12,
    }
    result, ms = _measure_latency(client, body, "relate")
    latency_samples.append(ms)
    rows = []
    for hit in result.get("hits", []):
        feature_str = _flatten_relate_feature(hit)
        lift = float(hit.get("lift") or hit.get("$lift") or 0.0)
        ftype = "qual" if _is_qualitative(hit) else "quant"
        rows.append(
            {
                "feature": feature_str,
                "type": ftype,
                "lift": round(lift, 2),
                "bar_pct": min(100, int(lift / 5 * 100)),
            }
        )
    payload = {
        "target": "outcome_bucket = great",
        "rows": rows[:10],
        "pullquote_html": (
            "Across the universe, these features are most associated with landing in the "
            "<em>great</em> outcome bucket."
        ),
    }
    (out_dir / "relate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _flatten_relate_feature(hit: dict) -> str:
    """Aito relate hits can name the related field + value in a few shapes."""
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


def emit_calibration(client: AitoClient, out_dir: Path, latency_samples: list[float]) -> None:
    """Calibration plot: predicted vs realised by decile.

    Methodology: for each row, call `_predict` on outcome_bucket given the
    company's grades; bucket the resulting P(great) into deciles; compare
    predicted P(great) to the actual frequency of `outcome_bucket = great`
    within that decile.

    This is a relatively expensive cross-validation pass — we sample a
    subset rather than processing the whole universe.
    """
    # Placeholder: emit a synthetic well-calibrated curve so the chart still
    # renders. Real calibration computation is a separate, larger job —
    # marked as a follow-up in pipeline notebooks.
    deciles = [
        {"label": f"{d}0%", "predicted": d / 10, "actual": max(0.0, d / 10 + (0.05 if d % 2 else -0.03))}
        for d in range(1, 11)
    ]
    payload = {
        "_note": "Placeholder. Real calibration computation pending — see notebooks/02_predict.ipynb.",
        "deciles": deciles,
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
    for focal in FOCAL_COMPANIES:
        body = {
            "from": COMPANIES_TABLE,
            "match": {"ticker": focal.ticker, "vintage_year": focal.vintage},
            "limit": 6,
        }
        try:
            result, ms = _measure_latency(client, body, "match")
        except Exception as e:
            print(f"  ⚠ {focal.ticker}·{focal.vintage} match failed: {e}")
            continue
        latency_samples.append(ms)
        matches = []
        for hit in result.get("hits", [])[:6]:
            ticker = str(hit.get("ticker") or "")
            vintage = int(hit.get("vintage_year") or 0)
            outcome_text = ""
            sentiment = "neutral"
            if hit.get("total_return_pct_local") is not None:
                pct = float(hit["total_return_pct_local"])
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
                    "similarity": round(float(hit.get("$score") or 0.0), 2),
                    "description": str(hit.get("market_position_rationale") or "")[:300],
                    "outcome": {
                        "text": outcome_text,
                        "window": f"'{vintage % 100:02d}→'{26}",
                        "sentiment": sentiment,
                    },
                }
            )
        payload = {
            "focal": {"ticker": focal.ticker, "vintage": focal.vintage, "name": focal.short_name},
            "matches": matches,
            "pullquote_html": "Nearest analogues by feature distance — live from Aito _match.",
        }
        (match_dir / f"{focal.ticker}_{focal.vintage}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


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
            print("→ relate")
            emit_relate(client, out_dir, latency_samples)
            print("→ calibration (placeholder)")
            emit_calibration(client, out_dir, latency_samples)
            print(f"→ predict × {len(FOCAL_COMPANIES)}")
            emit_predict_per_focal(client, companies_df, out_dir, latency_samples)
            print(f"→ match × {len(FOCAL_COMPANIES)}")
            emit_match_per_focal(client, companies_df, out_dir, latency_samples)
    else:
        print("→ (skipping Aito queries; static-only mode)")

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
