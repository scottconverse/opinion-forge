"""Tests for the browser auto-open utility.

Verifies open_browser() behaviour including server polling, timeout,
and graceful failure when the browser cannot be opened.
"""

from __future__ import annotations

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from opinionforge.desktop.browser import _wait_for_server, open_browser


# ---------------------------------------------------------------------------
# _wait_for_server
# ---------------------------------------------------------------------------


class TestWaitForServer:
    """Tests for the _wait_for_server helper."""

    def test_returns_true_when_server_responds(self) -> None:
        """Server that accepts connections immediately returns True."""
        # Spin up a tiny TCP server on an ephemeral port.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert _wait_for_server("127.0.0.1", port, timeout=2.0) is True
        finally:
            srv.close()

    def test_returns_false_on_timeout(self) -> None:
        """No server running on the port should time out quickly."""
        # Use a port that is (almost certainly) not in use.
        assert _wait_for_server("127.0.0.1", 19999, timeout=0.3) is False

    def test_waits_until_server_starts(self) -> None:
        """Server that starts after a delay is detected within the timeout."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.close()  # Close so the port is free initially

        def _delayed_listen() -> None:
            time.sleep(0.3)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(1)
            # Keep it alive long enough for the poll to succeed
            time.sleep(2.0)
            s.close()

        t = threading.Thread(target=_delayed_listen, daemon=True)
        t.start()

        result = _wait_for_server("127.0.0.1", port, timeout=3.0)
        assert result is True


# ---------------------------------------------------------------------------
# open_browser
# ---------------------------------------------------------------------------


class TestOpenBrowser:
    """Tests for the open_browser function."""

    @patch("opinionforge.desktop.browser.webbrowser.open", return_value=True)
    @patch("opinionforge.desktop.browser._wait_for_server", return_value=True)
    def test_calls_webbrowser_open(
        self, mock_wait: MagicMock, mock_wb: MagicMock
    ) -> None:
        """open_browser delegates to webbrowser.open with the URL."""
        result = open_browser("http://localhost:8484")
        mock_wb.assert_called_once_with("http://localhost:8484")
        assert result is True

    @patch("opinionforge.desktop.browser.webbrowser.open", return_value=True)
    @patch("opinionforge.desktop.browser._wait_for_server", return_value=True)
    def test_returns_true_on_success(
        self, mock_wait: MagicMock, mock_wb: MagicMock
    ) -> None:
        """Returns True when the browser opens successfully."""
        assert open_browser("http://localhost:8484") is True

    @patch("opinionforge.desktop.browser.webbrowser.open", return_value=False)
    @patch("opinionforge.desktop.browser._wait_for_server", return_value=True)
    def test_returns_false_when_browser_fails(
        self, mock_wait: MagicMock, mock_wb: MagicMock
    ) -> None:
        """Returns False when webbrowser.open returns False."""
        assert open_browser("http://localhost:8484") is False

    @patch("opinionforge.desktop.browser.webbrowser.open", side_effect=Exception("no display"))
    @patch("opinionforge.desktop.browser._wait_for_server", return_value=True)
    def test_returns_false_on_exception(
        self, mock_wait: MagicMock, mock_wb: MagicMock
    ) -> None:
        """Returns False (no raise) when webbrowser.open throws."""
        assert open_browser("http://localhost:8484") is False

    @patch("opinionforge.desktop.browser.webbrowser.open", return_value=True)
    @patch("opinionforge.desktop.browser._wait_for_server", return_value=False)
    def test_opens_after_timeout(
        self, mock_wait: MagicMock, mock_wb: MagicMock
    ) -> None:
        """Browser is still opened even if server polling times out."""
        result = open_browser("http://localhost:8484", timeout=0.1)
        assert result is True
        mock_wb.assert_called_once_with("http://localhost:8484")

    @patch("opinionforge.desktop.browser.webbrowser.open", return_value=True)
    @patch("opinionforge.desktop.browser._wait_for_server", return_value=True)
    def test_url_correctly_passed(
        self, mock_wait: MagicMock, mock_wb: MagicMock
    ) -> None:
        """The exact URL is forwarded to webbrowser.open."""
        open_browser("http://127.0.0.1:9999/settings")
        mock_wb.assert_called_once_with("http://127.0.0.1:9999/settings")
