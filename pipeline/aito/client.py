"""Thin Aito HTTP client.

Aito's REST API is small enough that a published SDK would be overkill.
All endpoints are POST with JSON; auth is via `x-api-key` header.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class AitoConfig:
    base_url: str  # e.g. https://shared.aito.ai/db/your-db
    api_key: str

    @classmethod
    def from_env(cls) -> "AitoConfig":
        url = os.environ.get("AITO_API_URL")
        key = os.environ.get("AITO_API_KEY")
        if not url or not key:
            raise RuntimeError(
                "AITO_API_URL and AITO_API_KEY must be set. Copy .env.example → .env"
            )
        return cls(base_url=url.rstrip("/"), api_key=key)


class AitoClient:
    """Minimal Aito API client — schema CRUD + predict/relate/match queries."""

    def __init__(self, config: AitoConfig | None = None, timeout: float = 60.0) -> None:
        self.config = config or AitoConfig.from_env()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers={"x-api-key": self.config.api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    # ── Schema ─────────────────────────────────────────────────

    def get_schema(self) -> dict:
        r = self._client.get("/api/v1/schema")
        r.raise_for_status()
        return r.json()

    def put_schema(self, schema: dict) -> dict:
        """Create or replace the full DB schema (top-level `schema` key)."""
        r = self._client.put("/api/v1/schema", json=schema)
        r.raise_for_status()
        return r.json()

    def delete_table(self, table: str) -> None:
        """Best-effort delete. Returns silently if the table doesn't exist."""
        r = self._client.delete(f"/api/v1/schema/{table}")
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()

    # ── Data ───────────────────────────────────────────────────

    def upload_batch(self, table: str, rows: list[dict], batch_size: int = 1000) -> int:
        """Upload `rows` into `table` in chunks. Returns total rows uploaded."""
        uploaded = 0
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            r = self._client.post(f"/api/v1/data/{table}/batch", json=chunk)
            r.raise_for_status()
            uploaded += len(chunk)
        return uploaded

    # ── Queries ────────────────────────────────────────────────

    def predict(self, body: dict) -> dict:
        return self._post_json("/api/v1/_predict", body)

    def relate(self, body: dict) -> dict:
        return self._post_json("/api/v1/_relate", body)

    def match(self, body: dict) -> dict:
        return self._post_json("/api/v1/_match", body)

    def search(self, body: dict) -> dict:
        return self._post_json("/api/v1/_search", body)

    def _post_json(self, path: str, body: dict) -> dict:
        r = self._client.post(path, json=body)
        r.raise_for_status()
        return r.json()

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> "AitoClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
