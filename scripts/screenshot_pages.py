"""Full-page desktop screenshots of given site paths.

Spins up a temporary http.server in front of site/, then drives a
headless chromium against it. Output goes to scripts/output/.

Usage:
    ./do screenshot-pages                      # screenshots /
    ./do screenshot-pages / /  /thesis         # screenshots each path
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from scripts._server import serve_site

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scripts" / "output"
WIDTH, HEIGHT = 1440, 900  # MacBook 14"-ish viewport


def safe_filename(path: str) -> str:
    """Turn a URL path into a safe filename. '/' → 'index'."""
    s = path.strip("/").replace("/", "_") or "index"
    return s


def capture(paths: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with serve_site() as base_url, sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for p in paths:
                page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
                url = f"{base_url}{p if p.startswith('/') else '/' + p}"
                page.goto(url)
                page.wait_for_load_state("networkidle")
                out = OUT_DIR / f"{safe_filename(p)}.desktop.png"
                page.screenshot(path=str(out), full_page=True)
                print(f"→ {out.relative_to(ROOT)} ({url})")
                page.close()
        finally:
            browser.close()


def main(argv: list[str]) -> int:
    paths = argv or ["/"]
    capture(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
