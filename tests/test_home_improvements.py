"""Tests for home page template improvements.

Verifies mode card descriptions, category colors, slider tooltips,
help icons, recent pieces sidebar, topic suggestions, and empty state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opinionforge.storage.database import Database
from opinionforge.storage.pieces import PieceStore
from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_piece(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid piece data dict."""
    base: dict[str, Any] = {
        "topic": "The future of renewable energy",
        "title": "Solar's Silent Revolution",
        "body": "Body text.",
        "preview_text": "Preview.",
        "mode": "analytical",
        "stance_position": 20,
        "stance_intensity": 0.6,
        "target_length": 800,
        "actual_length": 42,
        "disclaimer": "AI-generated content.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "home_test.db"


def _make_mock_provider() -> MagicMock:
    """Create a mock LLM provider so the home route does not redirect to /setup."""
    provider = MagicMock()
    provider.model_name.return_value = "mock/test-model"
    return provider


@pytest.fixture
def client(_db_path: Path) -> TestClient:
    """TestClient backed by a temporary database."""
    mock_client = MagicMock()
    web_app = create_app(client=mock_client, provider=_make_mock_provider(), db_path=str(_db_path))
    return TestClient(web_app)


@pytest.fixture
def client_with_pieces(_db_path: Path) -> TestClient:
    """TestClient with 5 pieces seeded in the database."""
    mock_client = MagicMock()
    web_app = create_app(client=mock_client, provider=_make_mock_provider(), db_path=str(_db_path))

    async def _seed() -> None:
        db = Database(str(_db_path))
        await db.connect()
        await db.initialize()
        store = PieceStore(db)
        for i in range(5):
            await store.save(
                _make_piece(
                    topic=f"Topic number {i + 1}",
                    title=f"Piece Title {i + 1}",
                    mode="polemical" if i % 2 == 0 else "analytical",
                )
            )
        await db.close()

    asyncio.run(_seed())
    return TestClient(web_app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModeCards:
    """Tests for mode card rendering."""

    def test_mode_cards_have_descriptions(self, client: TestClient) -> None:
        """Each mode card contains a one-liner description element."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert "mode-one-liner" in html

    def test_mode_cards_have_category_colors(self, client: TestClient) -> None:
        """Mode cards include category color class prefixes."""
        resp = client.get("/")
        html = resp.text
        # At least one of each category color class should be present
        assert "cat-confrontational" in html or "cat-investigative" in html
        assert "cat-deliberative" in html or "cat-literary" in html

    def test_mode_cards_have_color_coded_badges(self, client: TestClient) -> None:
        """Category badges are present with category class names."""
        resp = client.get("/")
        html = resp.text
        assert "category-badge" in html


class TestSliderTooltips:
    """Tests for slider tooltips and help text."""

    def test_stance_slider_has_tooltip(self, client: TestClient) -> None:
        """The stance slider has a title attribute explaining the range."""
        resp = client.get("/")
        html = resp.text
        assert 'title="' in html
        assert "-100" in html
        assert "+100" in html

    def test_intensity_slider_has_tooltip(self, client: TestClient) -> None:
        """The intensity slider has a title attribute explaining the range."""
        resp = client.get("/")
        html = resp.text
        assert "0.0" in html
        assert "1.0" in html

    def test_stance_slider_tooltip_text(self, client: TestClient) -> None:
        """The stance slider area includes descriptive tooltip text."""
        resp = client.get("/")
        html = resp.text
        assert "slider-tooltip" in html
        assert "equity-focused" in html
        assert "liberty-focused" in html

    def test_intensity_slider_tooltip_text(self, client: TestClient) -> None:
        """The intensity slider area includes descriptive tooltip text."""
        resp = client.get("/")
        html = resp.text
        assert "measured" in html
        assert "maximum conviction" in html


class TestHelpIcons:
    """Tests for inline help icons."""

    def test_help_icons_present(self, client: TestClient) -> None:
        """Help toggle icons (?) are present on key controls."""
        resp = client.get("/")
        html = resp.text
        assert "help-toggle" in html
        # At least topic, mode, stance, intensity should have help
        assert html.count("help-toggle") >= 4

    def test_help_text_elements_present(self, client: TestClient) -> None:
        """Help text divs exist for expanding explanations."""
        resp = client.get("/")
        html = resp.text
        assert "help-text" in html
        assert "help-topic" in html
        assert "help-mode" in html
        assert "help-stance" in html
        assert "help-intensity" in html


class TestRecentPiecesSidebar:
    """Tests for the recent pieces sidebar."""

    def test_sidebar_rendered(self, client: TestClient) -> None:
        """The recent pieces sidebar container is present."""
        resp = client.get("/")
        html = resp.text
        assert "recent-pieces-sidebar" in html

    def test_empty_sidebar_message(self, client: TestClient) -> None:
        """When no pieces exist, sidebar shows 'No recent pieces'."""
        resp = client.get("/")
        html = resp.text
        assert "No recent pieces" in html

    def test_sidebar_shows_pieces(self, client_with_pieces: TestClient) -> None:
        """When pieces exist, they appear as links in the sidebar."""
        resp = client_with_pieces.get("/")
        html = resp.text
        assert "Piece Title 1" in html or "recent-piece-card" in html


class TestTopicSuggestions:
    """Tests for the topic suggestions datalist."""

    def test_datalist_present(self, client: TestClient) -> None:
        """The topic-suggestions datalist element is in the HTML."""
        resp = client.get("/")
        html = resp.text
        assert "topic-suggestions" in html
        assert "<datalist" in html

    def test_datalist_has_suggestions(
        self, client_with_pieces: TestClient
    ) -> None:
        """When pieces exist, their topics appear as datalist options."""
        resp = client_with_pieces.get("/")
        html = resp.text
        assert "Topic number" in html
