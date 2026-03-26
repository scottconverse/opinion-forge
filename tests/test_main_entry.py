"""Tests for the updated __main__.py / CLI entry point behaviour.

Verifies bare command launches server, browser auto-open, --no-browser,
port fallback, URL printing, and subcommand routing.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from opinionforge.cli import app, _find_available_port

runner = CliRunner()


# ---------------------------------------------------------------------------
# Port finding
# ---------------------------------------------------------------------------


class TestFindAvailablePort:
    """Tests for _find_available_port helper."""

    def test_returns_first_available(self) -> None:
        """When the default port is free, returns it."""
        # Use a real socket to find a free port and verify
        port = _find_available_port(start=8484, end=8494)
        assert 8484 <= port <= 8494

    def test_skips_occupied_ports(self) -> None:
        """When first port is taken, returns the next free one."""
        # Occupy port 8484 with a real socket
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", 8484))
            srv.listen(1)
            port = _find_available_port(start=8484, end=8494)
            assert port > 8484
            assert port <= 8494
        finally:
            srv.close()

    def test_exits_when_all_ports_taken(self) -> None:
        """When all ports in a tiny range are taken, exits with code 1."""
        import click

        # Occupy a single port range
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        occupied_port = srv.getsockname()[1]
        srv.listen(1)
        try:
            with pytest.raises((SystemExit, click.exceptions.Exit)):
                _find_available_port(start=occupied_port, end=occupied_port)
        finally:
            srv.close()


# ---------------------------------------------------------------------------
# Bare command
# ---------------------------------------------------------------------------


class TestBareCommand:
    """Tests for bare 'opinionforge' invocation."""

    @patch("opinionforge.cli._launch_server")
    def test_bare_command_launches_server(self, mock_launch: MagicMock) -> None:
        """Running opinionforge with no args calls _launch_server."""
        result = runner.invoke(app, [])
        mock_launch.assert_called_once_with(open_browser=True)

    @patch("opinionforge.cli._launch_server")
    def test_bare_command_opens_browser(self, mock_launch: MagicMock) -> None:
        """Bare command passes open_browser=True."""
        runner.invoke(app, [])
        call_kwargs = mock_launch.call_args
        assert call_kwargs.kwargs.get("open_browser") is True


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for the 'serve' subcommand."""

    @patch("opinionforge.cli._launch_server")
    def test_serve_no_browser(self, mock_launch: MagicMock) -> None:
        """serve --no-browser suppresses browser auto-open."""
        result = runner.invoke(app, ["serve", "--no-browser"])
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["open_browser"] is False

    @patch("opinionforge.cli._launch_server")
    def test_serve_default_opens_browser(self, mock_launch: MagicMock) -> None:
        """serve without --no-browser opens the browser."""
        result = runner.invoke(app, ["serve"])
        _, kwargs = mock_launch.call_args
        assert kwargs["open_browser"] is True


# ---------------------------------------------------------------------------
# URL printed to console
# ---------------------------------------------------------------------------


class TestServerURLPrinted:
    """Tests that the server URL is printed to the console."""

    @patch("uvicorn.run")
    @patch("opinionforge.cli._find_available_port", return_value=8484)
    def test_url_printed(
        self, mock_port: MagicMock, mock_uvicorn: MagicMock
    ) -> None:
        """_launch_server prints the URL to the console."""
        from opinionforge.cli import _launch_server

        # Capture console output by running launch_server without browser
        result = runner.invoke(app, ["serve", "--no-browser"])
        # The output should contain the port
        assert "8484" in result.stdout


# ---------------------------------------------------------------------------
# Subcommands still work
# ---------------------------------------------------------------------------


class TestSubcommandsWork:
    """Verify that existing subcommands are not broken."""

    def test_modes_subcommand(self) -> None:
        """'opinionforge modes' still works."""
        result = runner.invoke(app, ["modes"])
        assert result.exit_code == 0

    @patch("opinionforge.cli._launch_server")
    def test_write_subcommand_requires_topic(self, mock_launch: MagicMock) -> None:
        """'opinionforge write' without topic exits with error."""
        result = runner.invoke(app, ["write"])
        # write without topic exits with code 2
        assert result.exit_code == 2
        # _launch_server should NOT be called — write is a subcommand
        mock_launch.assert_not_called()
