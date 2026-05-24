"""Static-serving stub for the aito-equity-demo.

This demo has no runtime backend — all Aito queries are precomputed
by the Python pipeline (see `pipeline/`) and written to `site/data/`
as JSON. The frontend (`site/index.html`) reads those JSON files
directly. This file exists only to satisfy the aito-demos-unified
platform contract:

  - GET /health      : cheap liveness probe nginx routes to
  - GET /api/health  : readiness probe (always ok — nothing to check)
  - GET /            : serves `site/` (the static frontend)

If you find yourself adding /api/* routes here, you've probably drifted
from the design. The intent is that the site works as a pure static
deployment; the uvicorn process is just there because the platform
expects one per demo.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="aito-equity-demo",
    description="Predictive equity research — LLM-extracted qualitative features + Aito predictive database.",
    version="0.1.0",
)


@app.get("/health")
def liveness() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/health")
def readiness() -> dict[str, str | bool]:
    return {"status": "ok", "backend": "static-only"}


# Static site — must be last so it doesn't shadow /api/* or /health.
_site_dir = Path(__file__).resolve().parent.parent / "site"
if _site_dir.exists():
    app.mount("/", StaticFiles(directory=str(_site_dir), html=True), name="site")
