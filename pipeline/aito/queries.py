"""Precompute Aito queries → site/data/*.json (the static-site data layer).

Run after data is loaded. Outputs:

  site/data/meta.json                          — observations / features / vintages / latency
  site/data/companies.json                     — focal-company chip list
  site/data/relate.json                        — feature lift vs. outcome=great
  site/data/calibration.json                   — predicted vs. realised by decile
  site/data/predict/{ticker}_{vintage}.json    — per-focal predict + grades + evidence
  site/data/match/{ticker}_{vintage}.json      — per-focal nearest analogues

The static site (site/index.html) fetches these directly. No request-time
Aito calls.

The set of focal companies (and therefore which predict/match files are
emitted) is configured in FOCAL_COMPANIES below. v1: NVDA·14, SHLD·14,
COST·17, META·20 — the four narrative quadrants from the TASK doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SITE_DATA = Path("site/data")


@dataclass(frozen=True)
class Focal:
    ticker: str
    vintage: int
    chip_label: str
    short_name: str


FOCAL_COMPANIES: list[Focal] = [
    Focal(ticker="NVDA", vintage=2014, chip_label="NVDA · '14", short_name="graphics, semiconductors"),
    Focal(ticker="SHLD", vintage=2014, chip_label="SHLD · '14", short_name="Sears Holdings"),
    Focal(ticker="COST", vintage=2017, chip_label="COST · '17", short_name="Costco Wholesale"),
    Focal(ticker="META", vintage=2020, chip_label="META · '20", short_name="platforms, advertising"),
]


def precompute_all(out_dir: Path = SITE_DATA) -> None:
    """Emit every JSON file the static site reads.

    Order:
      1. meta.json
      2. companies.json
      3. relate.json
      4. calibration.json
      5. predict/{focal}.json   for each FOCAL_COMPANIES entry
      6. match/{focal}.json     for each FOCAL_COMPANIES entry
    """
    raise NotImplementedError(
        "precompute_all pending — "
        "see aito-equity-demo-TASK.md → Build order → Frontend wiring"
    )
