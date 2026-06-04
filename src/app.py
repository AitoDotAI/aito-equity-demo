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
