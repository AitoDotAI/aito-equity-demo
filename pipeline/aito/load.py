"""Load data/companies.csv → Aito instance, using schema.json.

Idempotent: drops + recreates the `companies` table on every run. For
incremental updates, build a separate upsert flow — but a full reload
is the right default for this demo (datasets are small, point-in-time
snapshots don't drift).

Env contract:
  AITO_API_URL    — base URL (e.g. https://shared.aito.ai/db/your-db)
  AITO_API_KEY    — read/write key
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_companies_csv_to_aito(
    csv_path: Path = Path("data/companies.csv"),
    *,
    drop_existing: bool = True,
) -> None:
    raise NotImplementedError(
        "load_companies_csv_to_aito pending — "
        "see aito-equity-demo-TASK.md → Build order → Aito loading"
    )
