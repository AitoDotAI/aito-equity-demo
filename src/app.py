"""Static-serving stub + one /api/query proxy for the Live Query console.

The demo is mostly static — every chart you see in `site/index.html` was
precomputed by `./do pipeline precompute` and saved to `site/data/*.json`.
**Exception:** the Live Query view (view 6) lets the user paste their own
Aito query body and watch the response. That needs a single proxy route
so the Aito API key stays server-side. Everything else is static files.

Routes:
  - GET  /health         cheap liveness, no Aito call
  - GET  /api/health     readiness, includes Aito connectivity check
  - POST /api/query      proxies to Aito's _predict / _relate / _match
  - GET  /               serves site/ via StaticFiles

The /api/query route is constrained:
  - `kind` must be one of {predict, relate, match}
  - `body.from` must equal "companies" (the only table this demo loads)
  - 15-second upstream timeout

If AITO_API_URL / AITO_API_KEY aren't set, /api/query returns a clearly-
labelled mock response so the UI still works for offline demos.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(
    title="aito-equity-demo",
    description="Predictive equity research — LLM-extracted qualitative features + Aito predictive database.",
    version="0.2.0",
)

ALLOWED_FROM = "companies"  # this demo only exposes one table
ALLOWED_KIND = {"predict", "relate", "match"}


# ── Health ─────────────────────────────────────────────────────


@app.get("/health")
def liveness() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/health")
def readiness() -> dict[str, str | bool]:
    aito_configured = bool(os.environ.get("AITO_API_URL") and os.environ.get("AITO_API_KEY"))
    return {
        "status": "ok",
        "backend": "static-with-query-proxy",
        "aito_configured": aito_configured,
    }


# ── Live Query proxy ──────────────────────────────────────────


class QueryRequest(BaseModel):
    kind: Literal["predict", "relate", "match"]
    body: dict = Field(..., description="The Aito query body. `from` must be 'companies'.")


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    """Proxy a user-supplied Aito query for the Live Query console.

    Validates the body, calls Aito's `/api/v1/_{kind}`, returns the response
    plus timing. Mocks if Aito creds aren't configured.
    """
    from_table = req.body.get("from")
    if from_table != ALLOWED_FROM:
        raise HTTPException(
            status_code=400,
            detail=f"`body.from` must be '{ALLOWED_FROM}' (got '{from_table}'). "
            f"This demo only exposes the companies table.",
        )

    aito_url = os.environ.get("AITO_API_URL")
    aito_key = os.environ.get("AITO_API_KEY")

    if not (aito_url and aito_key):
        return {
            "ok": True,
            "source": "mock",
            "latency_ms": 12.0,
            "result": _mock_response(req.kind, req.body),
            "_note": "AITO_API_URL / AITO_API_KEY not set; returning a mock. Configure both in .env to call the real Aito instance.",
        }

    endpoint = f"/api/v1/_{req.kind}"
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{aito_url.rstrip('/')}{endpoint}",
                json=req.body,
                headers={"x-api-key": aito_key, "Content-Type": "application/json"},
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            return {
                "ok": False,
                "source": "aito",
                "latency_ms": round(latency_ms, 1),
                "error": f"Aito returned {r.status_code}: {r.text[:500]}",
            }
        return {
            "ok": True,
            "source": "aito",
            "latency_ms": round(latency_ms, 1),
            "result": r.json(),
        }
    except httpx.TimeoutException:
        return {"ok": False, "source": "aito", "error": "Aito request timed out after 15s"}
    except Exception as e:
        return {"ok": False, "source": "aito", "error": f"{type(e).__name__}: {e}"}


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="An earnings press release (or its headline + lead).")


@app.post("/api/classify-earnings")
def classify_earnings(req: ClassifyRequest) -> dict:
    """Live demo: read a pasted earnings release the way the pipeline does —
    LLM-extract the signals, then Aito-predict the day-1 reaction bucket given
    those signals. Both keys stay server-side. Mocks if either isn't set."""
    text = (req.text or "").strip()
    if len(text) < 40:
        raise HTTPException(status_code=400, detail="Paste a longer snippet (headline + a few lines).")

    openai_ok = bool(os.environ.get("OPENAI_MODEL_API_KEY") and os.environ.get("OPENAI_MODEL_URL"))
    aito_url = os.environ.get("AITO_API_URL")
    aito_key = os.environ.get("AITO_API_KEY")

    # 1. Extract signals.
    t0 = time.perf_counter()
    if openai_ok:
        try:
            from pipeline.events.earnings_extract import grade_release, make_client
            sig = grade_release(make_client(), text)
            signals = sig.model_dump() if sig else None
        except Exception as e:
            signals = None
            print(f"classify-earnings extract error: {e}")
    else:
        signals = None
    extract_ms = (time.perf_counter() - t0) * 1000

    if signals is None:
        signals = _mock_signals(text)
        signal_source = "mock"
    else:
        signal_source = "llm"

    # 2. Predict the reaction bucket from the signals.
    where = {k: v for k, v in signals.items() if v not in (None, "not_stated", "none")}
    t1 = time.perf_counter()
    prediction, pred_source = None, "mock"
    if aito_url and aito_key:
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.post(
                    f"{aito_url.rstrip('/')}/api/v1/_predict",
                    json={"from": "earnings_events", "where": where, "predict": "react_1d_bucket"},
                    headers={"x-api-key": aito_key, "Content-Type": "application/json"},
                )
            if r.status_code < 400:
                hits = r.json().get("hits", [])
                prediction = [{"bucket": h.get("feature"), "p": round(float(h.get("$p", 0)), 3)} for h in hits]
                pred_source = "aito"
        except Exception as e:
            print(f"classify-earnings predict error: {e}")
    predict_ms = (time.perf_counter() - t1) * 1000
    if prediction is None:
        prediction = _mock_prediction(signals)

    return {
        "ok": True,
        "signals": signals,
        "signal_source": signal_source,
        "prediction": prediction,
        "prediction_source": pred_source,
        "where": where,
        "extract_ms": round(extract_ms, 1),
        "predict_ms": round(predict_ms, 1),
    }


def _mock_signals(text: str) -> dict:
    """Keyword heuristic when no OpenAI key — clearly a fallback."""
    t = text.lower()
    neg = any(k in t for k in ("miss", "below expectations", "decline", "lower", "cut guidance", "disappoint"))
    pos = any(k in t for k in ("record", "beat", "raise", "exceeded", "strong", "growth"))
    return {
        "headline_signal": "clearly_negative" if neg and not pos else ("clearly_positive" if pos and not neg else "mixed"),
        "reported_beat": "miss" if "miss" in t else ("beat" if "beat" in t else "not_stated"),
        "guidance": "lowered" if "lower" in t or "cut" in t else ("raised" if "raise" in t else "none"),
        "eps_direction": "down" if neg and not pos else "up",
        "revenue_direction": "up" if "record revenue" in t or "revenue grew" in t else "not_stated",
        "one_time_items": "charges" if "charge" in t or "impairment" in t else "none",
        "tone": "cautious" if neg else "confident",
    }


def _mock_prediction(signals: dict) -> list[dict]:
    g = signals.get("guidance")
    hs = signals.get("headline_signal")
    if g == "lowered" or signals.get("reported_beat") == "miss" or hs == "clearly_negative":
        return [{"bucket": "down", "p": 0.55}, {"bucket": "flat", "p": 0.28}, {"bucket": "up", "p": 0.17}]
    if g == "raised" or hs == "clearly_positive":
        return [{"bucket": "up", "p": 0.5}, {"bucket": "flat", "p": 0.28}, {"bucket": "down", "p": 0.22}]
    return [{"bucket": "flat", "p": 0.4}, {"bucket": "up", "p": 0.32}, {"bucket": "down", "p": 0.28}]


def _mock_response(kind: str, body: dict) -> dict:
    """Deterministic stand-in for the Live Query view when Aito isn't wired.

    The structure matches Aito's real response shape so the UI can render
    against either. Numbers are illustrative.
    """
    if kind == "predict":
        return {
            "hits": [
                {"feature": "great", "$p": 0.412},
                {"feature": "good", "$p": 0.286},
                {"feature": "market", "$p": 0.184},
                {"feature": "poor", "$p": 0.087},
                {"feature": "disaster", "$p": 0.031},
            ],
        }
    if kind == "relate":
        return {
            "hits": [
                {"feature": {"field": "moat_strength", "value": "wide"}, "lift": 4.82, "$p": 0.18},
                {"feature": {"field": "market_position", "value": "dominant"}, "lift": 4.21, "$p": 0.16},
                {"feature": {"field": "market_quality", "value": "secular_growth"}, "lift": 3.58, "$p": 0.14},
                {"feature": {"field": "founder_still_ceo", "value": True}, "lift": 3.05, "$p": 0.12},
                {"feature": {"field": "moat_type", "value": "network_effects"}, "lift": 2.74, "$p": 0.10},
            ],
        }
    if kind == "match":
        return {
            "hits": [
                {"ticker": "MSFT", "vintage_year": 1995, "$score": 0.87, "company_name": "Microsoft"},
                {"ticker": "CSCO", "vintage_year": 1995, "$score": 0.81, "company_name": "Cisco Systems"},
                {"ticker": "ADBE", "vintage_year": 2010, "$score": 0.78, "company_name": "Adobe Systems"},
            ],
        }
    return {"hits": []}


# ── Static site — keep this last ───────────────────────────────


_site_dir = Path(__file__).resolve().parent.parent / "site"
if _site_dir.exists():
    app.mount("/", StaticFiles(directory=str(_site_dir), html=True), name="site")
