"""Tiny http.server context manager for screenshot scripts.

Serves the `site/` directory on a random free port. Used by
`screenshot_pages.py` and `inspect_mobile.py` so they don't depend on a
separately-running `./do serve`.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import socket
import socketserver
import threading
from pathlib import Path
from typing import Iterator

SITE_DIR = Path(__file__).resolve().parent.parent / "site"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # noqa: D401
        pass


@contextlib.contextmanager
def serve_site() -> Iterator[str]:
    """Yield the base URL of an http.server serving site/ on a random port."""
    port = _free_port()
    handler = functools.partial(_QuietHandler, directory=str(SITE_DIR))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()
