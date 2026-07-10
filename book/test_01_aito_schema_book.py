"""Smoke-test the configured Aito DB by fetching its schema.

booktest captures the httpx interactions on first run and replays them
on subsequent runs — so Aito-dependent tests stay fast and deterministic.

When the Aito schema legitimately changes (you load a new table, add a
column), regenerate the snapshot with:

    ./do test-book --update-snapshots

Run normally with:

    ./do test-book
"""

import os

import booktest as bt
import httpx
from dotenv import load_dotenv


@bt.snapshot_httpx()
def test_aito_schema(t: bt.TestCaseRun):
    """Fetch the schema and book the table list."""
    load_dotenv(override=True)
    # The backend stub was purged; read the env contract directly (no src.config).
    aito_url = os.environ.get("AITO_API_URL", "https://shared.aito.ai/db/aito-equity-demo")
    aito_key = os.environ.get("AITO_API_KEY", "")

    t.h1("Aito schema")
    t.tln(f"DB: `{aito_url}`")
    t.tln("")

    with httpx.Client(
        base_url=aito_url,
        headers={"x-api-key": aito_key, "content-type": "application/json"},
        timeout=10.0,
    ) as client:
        r = client.get("/api/v1/schema")
        r.raise_for_status()
        data = r.json()

    tables = sorted((data.get("schema") or {}).keys())

    t.h2("Tables")
    if not tables:
        t.tln("_(empty schema — load data first)_")
    else:
        for table in tables:
            cols = (data["schema"][table].get("columns") or {})
            t.iln(f"- `{table}` — {len(cols)} columns")

    t.tln("")
    t.assertln("schema returns successfully", isinstance(data, dict))
