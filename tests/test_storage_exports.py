"""Tests for exports CRUD operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from opinionforge.storage.database import Database
from opinionforge.storage.exports import ExportStore
from opinionforge.storage.pieces import PieceStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_piece(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid piece data dict."""
    base: dict[str, Any] = {
        "topic": "Test topic",
        "title": "Test Title",
        "body": "Body text.",
        "mode": "analytical",
        "stance_position": 0,
    }
    base.update(overrides)
    return base


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Provide a connected and initialised temp Database."""
    db_path = tmp_path / "exports_test.db"
    database = Database(db_path)
    await database.connect()
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
async def piece_store(db: Database) -> PieceStore:
    """Provide a PieceStore for creating parent pieces."""
    return PieceStore(db)


@pytest.fixture
async def store(db: Database) -> ExportStore:
    """Provide an ExportStore backed by the temp Database."""
    return ExportStore(db)


@pytest.fixture
async def sample_piece_id(piece_store: PieceStore) -> str:
    """Create and return a sample piece ID for use as a foreign key."""
    return await piece_store.save(_make_piece())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_and_retrieve(
    store: ExportStore, sample_piece_id: str
) -> None:
    """A saved export can be retrieved by ID."""
    export_id = await store.save(sample_piece_id, "markdown", "# Title\nContent")
    result = await store.get(export_id)
    assert result is not None
    assert result["id"] == export_id
    assert result["piece_id"] == sample_piece_id
    assert result["format"] == "markdown"
    assert result["content"] == "# Title\nContent"


async def test_export_uuid_auto_generated(
    store: ExportStore, sample_piece_id: str
) -> None:
    """save() auto-generates a UUID for the export."""
    export_id = await store.save(sample_piece_id, "html", "<p>text</p>")
    assert export_id is not None
    assert len(export_id) == 36  # UUID4 string length


async def test_get_by_piece_returns_all(
    store: ExportStore, sample_piece_id: str
) -> None:
    """get_by_piece returns all exports for a given piece."""
    await store.save(sample_piece_id, "markdown", "# MD content")
    await store.save(sample_piece_id, "html", "<h1>HTML content</h1>")
    await store.save(sample_piece_id, "pdf", "PDF bytes placeholder")

    results = await store.get_by_piece(sample_piece_id)
    assert len(results) == 3
    formats = {r["format"] for r in results}
    assert formats == {"markdown", "html", "pdf"}


async def test_get_by_piece_empty_result(store: ExportStore, sample_piece_id: str) -> None:
    """get_by_piece returns empty list when no exports exist."""
    results = await store.get_by_piece(sample_piece_id)
    assert results == []


async def test_get_nonexistent_returns_none(store: ExportStore) -> None:
    """Getting a nonexistent export returns None."""
    assert await store.get("nonexistent-export-id") is None


async def test_delete_existing(store: ExportStore, sample_piece_id: str) -> None:
    """delete() returns True and removes an existing export."""
    export_id = await store.save(sample_piece_id, "markdown", "content")
    assert await store.delete(export_id) is True
    assert await store.get(export_id) is None


async def test_delete_nonexistent_returns_false(store: ExportStore) -> None:
    """delete() returns False when the export does not exist."""
    assert await store.delete("no-such-id") is False


async def test_foreign_key_enforcement(store: ExportStore) -> None:
    """Saving an export for a nonexistent piece raises IntegrityError."""
    with pytest.raises(aiosqlite.IntegrityError):
        await store.save("nonexistent-piece-id", "markdown", "content")
