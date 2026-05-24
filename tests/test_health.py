"""Plain pytest unit tests for the health-stub.

This demo's `src/app.py` is a static-serving stub with no Aito client;
the health endpoints just verify the uvicorn process is up. All real
behaviour lives in the pipeline (see `pipeline/`) and the static site.
"""

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_api_health_endpoint_returns_ok() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "static-only"
