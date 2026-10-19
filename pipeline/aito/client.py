"""Thin Aito HTTP client — Aito v2 API (`/api/v2`).

Aito's REST API is small enough that a published SDK would be overkill.
All endpoints are POST with JSON; auth is via `x-api-key` header.

Every endpoint hangs off `api_root()`: `AITO_API_URL[/env/AITO_ENV]/api/v2`.
`AITO_ENV` (optional) points the whole demo at a branched environment — e.g. a
`v2` env whose tables were migrated with `_migrate` — without touching master.
Keys are database-level, so the same key reads/writes every env.

v1 → v2 notes that matter to callers (see docs/v2-migration.md):
  - ranked-value hits carry `$value`, not `feature` (`hit_value()` reads either)
  - `_evaluate` / `_estimate` / `_aggregate` answer `{kind, data}`; rows stay bare
    (`unwrap()` handles both)
  - tables are created as `collection` (the v2 / rep2 engine); `table` is still
    the v2 default and would silently keep the legacy v1 engine
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

# override=True so .env wins over placeholder exports from shell.nix
# (which sets AITO_API_URL=http://localhost:8200 / AITO_API_KEY="" as defaults).
load_dotenv(override=True)

API_VERSION = "v2"


def api_root(base_url: str | None = None, env: str | None = None) -> str:
    """`{AITO_API_URL}[/env/{AITO_ENV}]/api/v2` — the prefix of every endpoint."""
    base = (base_url or os.environ["AITO_API_URL"]).rstrip("/")
    env = os.environ.get("AITO_ENV") if env is None else env
    return f"{base}/env/{env}/api/{API_VERSION}" if env else f"{base}/api/{API_VERSION}"


def hit_value(hit: dict) -> Any:
    """The candidate value of a `_predict` / `_match` / `_recommend` hit (v2 `$value`, v1 `feature`)."""
    return hit["$value"] if "$value" in hit else hit.get("feature")


def unwrap(res: dict) -> Any:
    """v2 envelope: scalar kinds are `{kind, data}`, rows stay bare (`kind ?? "rows"`)."""
    return res["data"] if isinstance(res, dict) and "kind" in res and "data" in res else res


@dataclass(frozen=True)
class AitoConfig:
    base_url: str  # e.g. https://shared.aito.ai/db/your-db
    api_key: str
    env: str | None = None

    @classmethod
    def from_env(cls) -> "AitoConfig":
        url = os.environ.get("AITO_API_URL")
        key = os.environ.get("AITO_API_KEY")
        if not url or not key:
            raise RuntimeError(
                "AITO_API_URL and AITO_API_KEY must be set. Copy .env.example → .env"
            )
        return cls(base_url=url.rstrip("/"), api_key=key, env=os.environ.get("AITO_ENV") or None)

    @property
    def api_root(self) -> str:
        return api_root(self.base_url, self.env or "")


class AitoClient:
    """Minimal Aito API client — schema CRUD + predict/relate/match/evaluate queries."""

    def __init__(self, config: AitoConfig | None = None, timeout: float = 60.0) -> None:
        self.config = config or AitoConfig.from_env()
        self._client = httpx.Client(
            base_url=self.config.api_root,
            headers={"x-api-key": self.config.api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    # ── Schema ─────────────────────────────────────────────────

    def get_schema(self) -> dict:
        r = self._client.get("/schema")
        r.raise_for_status()
        return r.json()

    def put_schema(self, schema: dict) -> dict:
        """Create every table in `{"schema": {name: table_def}}`.

        v2 creates tables one at a time (`PUT /schema/{table}`). A `type: table`
        definition is created as a `collection` — the v2 engine — since `table`
        would keep the legacy v1 storage.
        """
        out = {}
        for name, tdef in schema["schema"].items():
            tdef = {**tdef, "type": "collection"} if tdef.get("type", "table") == "table" else tdef
            r = self._client.put(f"/schema/{name}", json=tdef)
            r.raise_for_status()
            out[name] = r.json()
        return out

    def delete_table(self, table: str) -> None:
        """Best-effort delete. Returns silently if the table doesn't exist."""
        r = self._client.delete(f"/schema/{table}")
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()

    # ── Data ───────────────────────────────────────────────────

    def upload_batch(self, table: str, rows: list[dict], batch_size: int = 1000) -> int:
        """Upload `rows` into `table` in chunks. Returns total rows uploaded."""
        uploaded = 0
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            r = self._client.post(f"/data/{table}/batch", json=chunk)
            r.raise_for_status()
            uploaded += len(chunk)
        return uploaded

    # ── Queries ────────────────────────────────────────────────

    def predict(self, body: dict) -> dict:
        return self._post_json("/_predict", body)

    def relate(self, body: dict) -> dict:
        return self._post_json("/_relate", body)

    def match(self, body: dict) -> dict:
        return self._post_json("/_match", body)

    def similarity(self, body: dict) -> dict:
        # Not in the v2 docs, but answered on /api/v2 (v2.11.6). v2 has no documented
        # idiom for categorical profile similarity: `orderBy: "$similarity"` needs a
        # Text `$match` term. Kept on _similarity until that has a v2 answer.
        return self._post_json("/_similarity", body)

    def search(self, body: dict) -> dict:
        return self._post_json("/_query", body)

    def evaluate(self, body: dict) -> dict:
        return unwrap(self._post_json("/_evaluate", body))

    def _post_json(self, path: str, body: dict) -> dict:
        r = self._client.post(path, json=body)
        r.raise_for_status()
        return r.json()

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> "AitoClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
