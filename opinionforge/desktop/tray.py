"""System tray icon with menu: Open, Settings, Quit.

Uses ``pystray`` for cross-platform system tray support. If pystray is
not installed, all tray functionality is silently disabled — it is an
optional dependency.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import webbrowser
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Attempt to import pystray; set a flag so callers can check availability.
try:
    import pystray  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]

    PYSTRAY_AVAILABLE = True
except ImportError:
    pystray = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment,misc]
    PYSTRAY_AVAILABLE = False


def _default_icon_image() -> Any:
    """Create a simple default icon image when no logo is bundled.

    Returns:
        A PIL Image suitable for use as a tray icon, or None if PIL is
        unavailable.
    """
    if Image is None:
        return None
    # Create a 64x64 teal square as a fallback icon.
    img = Image.new("RGB", (64, 64), color=(0, 150, 136))
    return img


class SystemTrayApp:
    """System tray application for OpinionForge.

    Provides a tray icon with a context menu containing Open in Browser,
    Settings, and Quit actions.

    Args:
        port: The port the web server is running on.
        host: The host the web server is bound to.
        shutdown_callback: A callable invoked when the user clicks Quit.
            If None, sends SIGINT to the current process.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        shutdown_callback: Callable[[], None] | None = None,
    ) -> None:
        self.port = port
        self.host = host
        self._shutdown_callback = shutdown_callback
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    @property
    def server_url(self) -> str:
        """Return the base URL for the running server."""
        return f"http://{self.host}:{self.port}"

    def _on_open(self) -> None:
        """Open the default browser to the server URL."""
        try:
            webbrowser.open(self.server_url)
        except Exception:
            logger.warning("Could not open browser from tray", exc_info=True)

    def _on_settings(self) -> None:
        """Open the default browser to the settings page."""
        try:
            webbrowser.open(f"{self.server_url}/settings")
        except Exception:
            logger.warning("Could not open settings from tray", exc_info=True)

    def _on_quit(self) -> None:
        """Stop the tray icon and trigger server shutdown."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

        if self._shutdown_callback is not None:
            self._shutdown_callback()
        else:
            # Default: send SIGINT to allow graceful shutdown
            os.kill(os.getpid(), signal.SIGINT)

    def _build_menu(self) -> Any:
        """Build the tray context menu.

        Returns:
            A pystray.Menu instance with Open, Settings, and Quit items.
        """
        if pystray is None:
            return None
        return pystray.Menu(
            pystray.MenuItem("Open in Browser", lambda: self._on_open(), default=True),
            pystray.MenuItem("Settings", lambda: self._on_settings()),
            pystray.MenuItem("Quit", lambda: self._on_quit()),
        )

    def start(self) -> None:
        """Start the system tray icon in a background thread.

        If pystray is not installed, logs a debug message and returns
        without error.
        """
        if not PYSTRAY_AVAILABLE:
            logger.debug("pystray not installed; skipping system tray")
            return

        try:
            icon_image = _default_icon_image()
            tooltip = f"OpinionForge \u2014 Running on port {self.port}"
            self._icon = pystray.Icon(
                name="opinionforge",
                icon=icon_image,
                title=tooltip,
                menu=self._build_menu(),
            )
            self._thread = threading.Thread(
                target=self._icon.run,
                daemon=True,
                name="opinionforge-tray",
            )
            self._thread.start()
            logger.info("System tray icon started")
        except Exception:
            logger.warning(
                "Failed to create system tray icon; continuing without tray",
                exc_info=True,
            )

    def stop(self) -> None:
        """Stop the system tray icon if it is running."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None
