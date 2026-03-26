"""Browser auto-open utility with cross-platform support.

Uses Python's stdlib ``webbrowser`` module to open the default browser.
Waits for the server to be responsive before opening, with a configurable
timeout.
"""

from __future__ import annotations

import logging
import socket
import time
import webbrowser
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Maximum time (seconds) to wait for the server to respond before
# opening the browser anyway.
_POLL_TIMEOUT: float = 10.0

# Interval between polling attempts.
_POLL_INTERVAL: float = 0.25


def _wait_for_server(host: str, port: int, timeout: float = _POLL_TIMEOUT) -> bool:
    """Poll the server until it accepts a TCP connection.

    Args:
        host: The hostname or IP to connect to.
        port: The TCP port to probe.
        timeout: Maximum seconds to wait.

    Returns:
        True if the server responded within the timeout, False otherwise.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(_POLL_INTERVAL)
    return False


def open_browser(url: str, *, timeout: float = _POLL_TIMEOUT) -> bool:
    """Open the default browser to *url* after the server is responsive.

    Waits up to *timeout* seconds for the server to accept connections.
    If the server is not ready in time, the browser is opened anyway
    (the user will see a loading page).

    Args:
        url: The full URL to open (e.g. ``http://localhost:8484``).
        timeout: Maximum seconds to wait for the server.

    Returns:
        True if ``webbrowser.open`` succeeded, False otherwise.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    ready = _wait_for_server(host, port, timeout=timeout)
    if not ready:
        logger.warning(
            "Server at %s:%d not responsive after %.1f seconds; opening browser anyway",
            host,
            port,
            timeout,
        )

    try:
        return webbrowser.open(url)
    except Exception:
        logger.warning("Could not open browser for %s", url, exc_info=True)
        return False
