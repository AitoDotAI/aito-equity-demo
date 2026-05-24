"""iPhone-sized screenshots of given site paths (layout review).

Uses Playwright's iPhone 13 device descriptor. Spins up a temporary
http.server in front of site/. Output goes to scripts/output/.

Usage:
    ./do inspect-mobile                  # iPhone screenshot of /
    ./do inspect-mobile / /thesis        # several paths
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from scripts._server import serve_site

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scripts" / "output"
DEVICE_NAME = "iPhone 13"


def safe_filename(path: str) -> str:
    s = path.strip("/").replace("/", "_") or "index"
    return s


def capture(paths: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with serve_site() as base_url, sync_playwright() as pw:
        device = pw.devices[DEVICE_NAME]
        browser = pw.chromium.launch()
        try:
            for p in paths:
                context = browser.new_context(**device)
                page = context.new_page()
                url = f"{base_url}{p if p.startswith('/') else '/' + p}"
                page.goto(url)
                page.wait_for_load_state("networkidle")
                out = OUT_DIR / f"{safe_filename(p)}.mobile.png"
                page.screenshot(path=str(out), full_page=True)
                print(f"→ {out.relative_to(ROOT)} ({url}, {DEVICE_NAME})")
                context.close()
        finally:
            browser.close()


def main(argv: list[str]) -> int:
    paths = argv or ["/"]
    capture(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
