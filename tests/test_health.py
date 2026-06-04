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
    assert body["backend"] == "static-with-query-proxy"


def test_api_query_validates_from() -> None:
    """The /api/query route must reject `from` values other than 'companies'."""
    r = client.post("/api/query", json={"kind": "predict", "body": {"from": "other_table"}})
    assert r.status_code == 400
    assert "companies" in r.json()["detail"]


def test_api_query_returns_mock_without_creds(monkeypatch) -> None:
    """When AITO_API_URL / AITO_API_KEY aren't set, the route returns a labelled mock."""
    monkeypatch.delenv("AITO_API_URL", raising=False)
    monkeypatch.delenv("AITO_API_KEY", raising=False)
    r = client.post(
        "/api/query",
        json={"kind": "predict", "body": {"from": "companies", "where": {}}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["source"] == "mock"
    assert "hits" in body["result"]


def test_api_query_validates_kind() -> None:
    """Pydantic should reject unknown kind values."""
    r = client.post("/api/query", json={"kind": "lol", "body": {"from": "companies"}})
    assert r.status_code == 422  # pydantic validation
