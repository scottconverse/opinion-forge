"""Tests that history templates render with expected structure.

Verifies the HTML structure of history and detail pages, ensuring
all required UI elements are present.
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
        "topic": "AI governance and democratic institutions",
        "title": "The Algorithmic Threat",
        "subtitle": "A deeper look at AI and democracy",
        "body": "<p>This is the body of the piece with enough content.</p>",
        "preview_text": "A preview of the piece.",
        "mode": "analytical",
        "stance_position": -30,
        "stance_intensity": 0.7,
        "target_length": 800,
        "actual_length": 450,
        "disclaimer": "AI-generated content disclaimer.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _db_path(tmp_path: Path) -> Path:
    """Return a temporary database path."""
    return tmp_path / "tpl_test.db"


@pytest.fixture
def client(_db_path: Path) -> TestClient:
    """Return a TestClient with a temp database."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Mock."
    app = create_app(client=mock_llm, db_path=str(_db_path))
    return TestClient(app)


def _seed(db_path: str | Path, pieces: list[dict]) -> list[str]:
    """Insert pieces into the database and return IDs."""
    ids: list[str] = []

    async def _insert() -> None:
        async with Database(db_path) as db:
            store = PieceStore(db)
            for p in pieces:
                ids.append(await store.save(p))

    asyncio.run(_insert())
    return ids


# ---------------------------------------------------------------------------
# Tests: History page structure
# ---------------------------------------------------------------------------


def test_history_page_has_search_input(client: TestClient) -> None:
    """History page contains a search input element."""
    response = client.get("/history")
    assert response.status_code == 200
    assert 'id="history-search-input"' in response.text


def test_history_page_has_filter_dropdowns(client: TestClient) -> None:
    """History page contains filter dropdowns for mode and sort."""
    response = client.get("/history")
    text = response.text
    assert 'id="filter-mode"' in text
    assert 'id="filter-sort"' in text


def test_history_page_has_sort_dropdown(client: TestClient) -> None:
    """History page sort dropdown has all required options."""
    response = client.get("/history")
    text = response.text
    assert "Newest first" in text
    assert "Oldest first" in text
    assert "Word count (high to low)" in text
    assert "Word count (low to high)" in text
    assert "Mode A-Z" in text


def test_piece_cards_contain_expected_fields(
    client: TestClient, _db_path: Path,
) -> None:
    """Piece cards display title, mode badge, date, and word count."""
    _seed(str(_db_path), [
        _make_piece(title="Card Test Piece", mode="analytical", actual_length=450),
    ])
    response = client.get("/history")
    text = response.text
    assert "Card Test Piece" in text
    assert "analytical" in text
    assert "450" in text


def test_detail_page_shows_full_body(
    client: TestClient, _db_path: Path,
) -> None:
    """Detail page renders the full piece body."""
    ids = _seed(str(_db_path), [
        _make_piece(body="<p>Full body content for the detail page.</p>"),
    ])
    response = client.get(f"/history/{ids[0]}")
    assert response.status_code == 200
    assert "Full body content for the detail page." in response.text


def test_detail_page_has_export_buttons(
    client: TestClient, _db_path: Path,
) -> None:
    """Detail page has export buttons for all 4 formats."""
    ids = _seed(str(_db_path), [_make_piece()])
    response = client.get(f"/history/{ids[0]}")
    text = response.text
    assert "Substack" in text
    assert "Medium" in text
    assert "Wordpress" in text
    assert "Twitter" in text


def test_detail_page_has_delete_button(
    client: TestClient, _db_path: Path,
) -> None:
    """Detail page has a delete button."""
    ids = _seed(str(_db_path), [_make_piece()])
    response = client.get(f"/history/{ids[0]}")
    assert "Delete Piece" in response.text


def test_empty_state_message(client: TestClient) -> None:
    """Empty state message appears when no pieces exist."""
    response = client.get("/history")
    assert "No pieces yet" in response.text
    assert "home page" in response.text


def test_nav_includes_history_link(client: TestClient) -> None:
    """Navigation bar includes a History link on all pages."""
    # Check on the home page
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/history"' in response.text


def test_nav_includes_settings_link(client: TestClient) -> None:
    """Navigation bar includes a Settings link."""
    response = client.get("/")
    assert 'href="/settings"' in response.text
