"""Tests for the system tray integration.

All tests mock pystray so no real tray icon is created. Verifies menu
construction, action callbacks, graceful degradation when pystray is
absent, and tooltip content.
"""

from __future__ import annotations

import contextlib
import os
import signal
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_tray_module() -> ModuleType:
    """Import the tray module fresh (needed for PYSTRAY_AVAILABLE patching)."""
    import opinionforge.desktop.tray as tray_mod

    return tray_mod


@contextlib.contextmanager
def _ensure_pystray_mock(tray_mod: ModuleType):
    """Context manager that injects a mock pystray if the real one is absent.

    When pystray *is* installed the real module is left in place.
    When it is *not* installed, a lightweight MagicMock is swapped in
    for the duration of the block so that production code paths
    (``_build_menu``, ``start``) execute instead of short-circuiting.
    """
    if tray_mod.PYSTRAY_AVAILABLE:
        yield tray_mod.pystray
        return

    mock_pystray = MagicMock()
    # MenuItem: store the label so str(item) returns it.
    mock_pystray.MenuItem.side_effect = lambda label, cb, **kw: MagicMock(
        __str__=lambda self: label
    )
    # Menu: return a MagicMock that is iterable over its positional args.
    mock_pystray.Menu.side_effect = lambda *items: MagicMock(
        __iter__=lambda self: iter(items)
    )
    # Icon: just a mock instance whose run() is a no-op.
    mock_icon = MagicMock()
    mock_pystray.Icon.return_value = mock_icon

    original_pystray = tray_mod.pystray
    original_flag = tray_mod.PYSTRAY_AVAILABLE
    tray_mod.pystray = mock_pystray
    tray_mod.PYSTRAY_AVAILABLE = True
    try:
        yield mock_pystray
    finally:
        tray_mod.pystray = original_pystray
        tray_mod.PYSTRAY_AVAILABLE = original_flag


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSystemTrayApp:
    """Tests for SystemTrayApp with mocked pystray."""

    def test_creation(self) -> None:
        """SystemTrayApp can be instantiated with port and host."""
        tray_mod = _import_tray_module()
        app = tray_mod.SystemTrayApp(port=8484, host="127.0.0.1")
        assert app.port == 8484
        assert app.host == "127.0.0.1"
        assert app.server_url == "http://127.0.0.1:8484"

    def test_menu_has_three_items(self) -> None:
        """The tray menu contains Open, Settings, and Quit items."""
        tray_mod = _import_tray_module()

        with _ensure_pystray_mock(tray_mod):
            app = tray_mod.SystemTrayApp(port=8484)
            menu = app._build_menu()
            items = list(menu)
            assert len(items) == 3
            assert "Open in Browser" in str(items[0])
            assert "Settings" in str(items[1])
            assert "Quit" in str(items[2])

    @patch("webbrowser.open")
    def test_open_action_calls_browser(self, mock_wb: MagicMock) -> None:
        """The Open in Browser action opens the server URL."""
        tray_mod = _import_tray_module()
        app = tray_mod.SystemTrayApp(port=8484)
        app._on_open()
        mock_wb.assert_called_once_with("http://127.0.0.1:8484")

    @patch("webbrowser.open")
    def test_settings_action_opens_settings(self, mock_wb: MagicMock) -> None:
        """The Settings action opens the /settings URL."""
        tray_mod = _import_tray_module()
        app = tray_mod.SystemTrayApp(port=8484)
        app._on_settings()
        mock_wb.assert_called_once_with("http://127.0.0.1:8484/settings")

    def test_quit_action_calls_shutdown_callback(self) -> None:
        """The Quit action invokes the shutdown callback."""
        tray_mod = _import_tray_module()
        cb = MagicMock()
        app = tray_mod.SystemTrayApp(port=8484, shutdown_callback=cb)
        app._on_quit()
        cb.assert_called_once()

    def test_tray_disabled_when_pystray_not_installed(self) -> None:
        """start() does not raise when pystray is unavailable."""
        tray_mod = _import_tray_module()
        app = tray_mod.SystemTrayApp(port=8484)

        # Temporarily pretend pystray is unavailable
        original = tray_mod.PYSTRAY_AVAILABLE
        tray_mod.PYSTRAY_AVAILABLE = False
        try:
            # Should not raise
            app.start()
        finally:
            tray_mod.PYSTRAY_AVAILABLE = original

    def test_tooltip_contains_port(self) -> None:
        """The tray tooltip includes the port number."""
        tray_mod = _import_tray_module()

        with _ensure_pystray_mock(tray_mod) as mock_pystray:
            app = tray_mod.SystemTrayApp(port=9123)
            app.start()

            mock_pystray.Icon.assert_called_once()
            call_kwargs = mock_pystray.Icon.call_args
            tooltip = call_kwargs.kwargs.get("title", "")
            if not tooltip and len(call_kwargs.args) > 2:
                tooltip = call_kwargs.args[2]
            assert "9123" in str(tooltip)

    def test_server_url_property(self) -> None:
        """server_url returns the correct URL."""
        tray_mod = _import_tray_module()
        app = tray_mod.SystemTrayApp(port=8485, host="0.0.0.0")
        assert app.server_url == "http://0.0.0.0:8485"
