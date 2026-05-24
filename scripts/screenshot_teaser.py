"""Render assets/teaser.html → assets/teaser.png at 1200×630.

The landing page (aito.ai) thumbnails this PNG, so the size is fixed
by the platform's social-card / Open Graph layout.

Usage:
    ./do screenshot-teaser                  # writes assets/teaser.png
    ./do screenshot-teaser custom.png       # writes the given path
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
TEASER_HTML = ROOT / "assets" / "teaser.html"
DEFAULT_OUT = ROOT / "assets" / "teaser.png"
WIDTH, HEIGHT = 1200, 630


def render(out_path: Path) -> None:
    if not TEASER_HTML.exists():
        raise FileNotFoundError(f"missing source: {TEASER_HTML}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.goto(TEASER_HTML.as_uri())
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(out_path), omit_background=False)
        finally:
            browser.close()
    print(f"→ {out_path.relative_to(ROOT)} ({WIDTH}×{HEIGHT})")


def main(argv: list[str]) -> int:
    out_path = Path(argv[0]) if argv else DEFAULT_OUT
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    render(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
