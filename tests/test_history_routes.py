"""Tests for history page backend routes.

Verifies GET /history, GET /history/{id}, POST /history/search,
POST /history/{id}/delete, POST /history/bulk-delete, and
POST /history/{id}/export.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opinionforge.storage.database import Database
from opinionforge.storage.exports import ExportStore
from opinionforge.storage.pieces import PieceStore
from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_piece(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid piece data dict with optional overrides."""
    base: dict[str, Any] = {
        "topic": "The future of renewable energy",
        "title": "Solar's Silent Revolution",
        "body": "The economics of solar energy have shifted decisively.",
        "preview_text": "The economics of solar energy have shifted.",
        "mode": "analytical",
        "stance_position": 20,
        "stance_intensity": 0.6,
        "target_length": 800,
        "actual_length": 42,
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
    return tmp_path / "history_test.db"


@pytest.fixture
def client(_db_path: Path) -> TestClient:
    """Return a TestClient backed by a temporary database."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Mock LLM response."
    app = create_app(client=mock_llm, db_path=str(_db_path))
    return TestClient(app)


def _seed_pieces(db_path: str | Path, pieces: list[dict[str, Any]]) -> list[str]:
    """Insert pieces into the database and return their IDs."""
    ids: list[str] = []

    async def _insert() -> None:
        async with Database(db_path) as db:
            store = PieceStore(db)
            for p in pieces:
                pid = await store.save(p)
                ids.append(pid)

    asyncio.run(_insert())
    return ids


# ---------------------------------------------------------------------------
# Tests: GET /history
# ---------------------------------------------------------------------------


def test_history_returns_200_empty_db(client: TestClient) -> None:
    """GET /history returns 200 even with an empty database."""
    response = client.get("/history")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_history_shows_empty_state(client: TestClient) -> None:
    """GET /history shows the empty state message when no pieces exist."""
    response = client.get("/history")
    assert "No pieces yet" in response.text


def test_history_returns_pieces(client: TestClient, _db_path: Path) -> None:
    """GET /history returns pieces when the database has data."""
    _seed_pieces(str(_db_path), [_make_piece(title="First Piece")])
    response = client.get("/history")
    assert response.status_code == 200
    assert "First Piece" in response.text


def test_history_pagination(client: TestClient, _db_path: Path) -> None:
    """GET /history paginates when piece count exceeds 20."""
    pieces = [_make_piece(title=f"Piece {i}") for i in range(25)]
    _seed_pieces(str(_db_path), pieces)

    page1 = client.get("/history?page=1")
    assert page1.status_code == 200
    # Check pagination text exists (avoid title-case triggering name sanitization)
    assert "1 of 2" in page1.text

    page2 = client.get("/history?page=2")
    assert page2.status_code == 200
    assert "2 of 2" in page2.text


# ---------------------------------------------------------------------------
# Tests: GET /history/{id}
# ---------------------------------------------------------------------------


def test_history_detail_returns_200(
    client: TestClient, _db_path: Path,
) -> None:
    """GET /history/{id} returns 200 for an existing piece."""
    ids = _seed_pieces(str(_db_path), [_make_piece(title="Detail Piece")])
    response = client.get(f"/history/{ids[0]}")
    assert response.status_code == 200
    assert "Detail Piece" in response.text


def test_history_detail_returns_404(client: TestClient) -> None:
    """GET /history/{id} returns 404 for a nonexistent piece."""
    response = client.get("/history/nonexistent-id-12345")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /history/search
# ---------------------------------------------------------------------------


def test_search_by_keyword(client: TestClient, _db_path: Path) -> None:
    """POST /history/search by keyword returns matching pieces."""
    _seed_pieces(str(_db_path), [
        _make_piece(topic="renewable energy", title="Solar"),
        _make_piece(topic="machine learning", title="ML Advances"),
    ])
    response = client.post("/history/search", data={"query": "renewable"})
    assert response.status_code == 200
    assert "Solar" in response.text
    assert "ML Advances" not in response.text


def test_search_no_results(client: TestClient, _db_path: Path) -> None:
    """POST /history/search with no results returns empty state."""
    _seed_pieces(str(_db_path), [_make_piece()])
    response = client.post("/history/search", data={"query": "zzzznonexistent"})
    assert response.status_code == 200
    assert "No pieces match" in response.text


def test_filter_by_mode(client: TestClient, _db_path: Path) -> None:
    """POST /history/search filtered by mode returns the correct subset."""
    _seed_pieces(str(_db_path), [
        _make_piece(mode="analytical", title="Analytical Piece"),
        _make_piece(mode="polemical", title="Polemical Piece"),
    ])
    response = client.post("/history/search", data={"mode": "polemical"})
    assert response.status_code == 200
    assert "Polemical Piece" in response.text
    assert "Analytical Piece" not in response.text


def test_filter_by_stance_range(client: TestClient, _db_path: Path) -> None:
    """POST /history/search filters by stance range."""
    _seed_pieces(str(_db_path), [
        _make_piece(stance_position=-50, title="Left Piece"),
        _make_piece(stance_position=50, title="Right Piece"),
    ])
    response = client.post(
        "/history/search",
        data={"stance_min": "30", "stance_max": "100"},
    )
    assert response.status_code == 200
    assert "Right Piece" in response.text
    assert "Left Piece" not in response.text


def test_filter_by_date_range(client: TestClient, _db_path: Path) -> None:
    """POST /history/search filters by date range."""
    _seed_pieces(str(_db_path), [_make_piece(title="Recent Piece")])
    # Use a wide date range that captures today
    response = client.post(
        "/history/search",
        data={"date_from": "2020-01-01", "date_to": "2030-12-31"},
    )
    assert response.status_code == 200
    assert "Recent Piece" in response.text


def test_sort_by_date_ascending(client: TestClient, _db_path: Path) -> None:
    """POST /history/search with sort_by=oldest returns oldest first."""
    _seed_pieces(str(_db_path), [
        _make_piece(title="First"),
        _make_piece(title="Second"),
    ])
    response = client.post("/history/search", data={"sort_by": "oldest"})
    assert response.status_code == 200
    text = response.text
    pos_first = text.find("First")
    pos_second = text.find("Second")
    assert pos_first < pos_second


def test_sort_by_word_count_desc(client: TestClient, _db_path: Path) -> None:
    """POST /history/search with sort_by=words_desc returns highest first."""
    _seed_pieces(str(_db_path), [
        _make_piece(actual_length=100, title="Short"),
        _make_piece(actual_length=5000, title="Long"),
    ])
    response = client.post("/history/search", data={"sort_by": "words_desc"})
    assert response.status_code == 200
    text = response.text
    assert text.find("Long") < text.find("Short")


# ---------------------------------------------------------------------------
# Tests: POST /history/{id}/delete
# ---------------------------------------------------------------------------


def test_delete_single_piece(client: TestClient, _db_path: Path) -> None:
    """POST /history/{id}/delete removes the piece and redirects."""
    ids = _seed_pieces(str(_db_path), [_make_piece(title="To Delete")])
    response = client.post(
        f"/history/{ids[0]}/delete", follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/history"

    # Verify piece is gone
    detail = client.get(f"/history/{ids[0]}")
    assert detail.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /history/bulk-delete
# ---------------------------------------------------------------------------


def test_bulk_delete_removes_pieces(
    client: TestClient, _db_path: Path,
) -> None:
    """POST /history/bulk-delete removes multiple pieces."""
    ids = _seed_pieces(str(_db_path), [
        _make_piece(title="Delete Me 1"),
        _make_piece(title="Delete Me 2"),
        _make_piece(title="Keep Me"),
    ])
    response = client.post(
        "/history/bulk-delete",
        data={"piece_ids": [ids[0], ids[1]]},
    )
    assert response.status_code == 200
    assert "Keep Me" in response.text
    assert "Delete Me 1" not in response.text
    assert "Delete Me 2" not in response.text


# ---------------------------------------------------------------------------
# Tests: POST /history/{id}/export
# ---------------------------------------------------------------------------


def test_export_piece_returns_content(
    client: TestClient, _db_path: Path,
) -> None:
    """POST /history/{id}/export returns formatted export content."""
    ids = _seed_pieces(str(_db_path), [
        _make_piece(
            title="Export Test",
            body="This is the body for export testing.",
        ),
    ])
    response = client.post(
        f"/history/{ids[0]}/export", data={"format": "substack"},
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_export_saves_to_export_store(
    client: TestClient, _db_path: Path,
) -> None:
    """POST /history/{id}/export saves the export record."""
    ids = _seed_pieces(str(_db_path), [
        _make_piece(title="Export Save Test", body="Body for export."),
    ])
    client.post(
        f"/history/{ids[0]}/export", data={"format": "medium"},
    )

    # Verify export was saved
    async def _check() -> list:
        async with Database(str(_db_path)) as db:
            es = ExportStore(db)
            return await es.get_by_piece(ids[0])

    exports = asyncio.run(_check())
    assert len(exports) == 1
    assert exports[0]["format"] == "medium"
