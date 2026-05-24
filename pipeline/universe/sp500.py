"""S&P 500 historical constituents from Wikipedia page-edit history.

Wikipedia's S&P 500 article has been continuously maintained: the edit
history names companies added/removed on specific dates. Reconstruct
point-in-time membership by replaying diffs back from today to the
vintage date.

Quality is good for >= 2010 vintages; pre-2010 has gaps from inconsistent
editor practice. Sanity-check the 2014/2017/2020 lists against an archived
constituent snapshot (SlickCharts historical, e.g.) before treating as
ground truth.
"""

from __future__ import annotations

from datetime import date

from pipeline.universe.base import UniverseEntry


class SP500WikipediaSource:
    """US-market UniverseSource — S&P 500 from Wikipedia edit history."""

    market: str = "US"

    def list_constituents(self, vintage_date: date) -> list[UniverseEntry]:
        raise NotImplementedError(
            "SP500WikipediaSource.list_constituents pending — "
            "see aito-equity-demo-TASK.md → Data Pipeline → Universe construction"
        )
