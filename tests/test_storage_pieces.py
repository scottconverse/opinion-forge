"""Tests for pieces CRUD operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from opinionforge.storage.database import Database
from opinionforge.storage.pieces import PieceStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
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
        "actual_length": 812,
    }
    base.update(overrides)
    return base


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Provide a connected and initialised temp Database."""
    db_path = tmp_path / "pieces_test.db"
    database = Database(db_path)
    await database.connect()
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
async def store(db: Database) -> PieceStore:
    """Provide a PieceStore backed by the temp Database."""
    return PieceStore(db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_and_retrieve(store: PieceStore) -> None:
    """A saved piece can be retrieved by ID."""
    piece_id = await store.save(_make_piece())
    result = await store.get(piece_id)
    assert result is not None
    assert result["id"] == piece_id
    assert result["title"] == "Solar's Silent Revolution"


async def test_save_auto_generates_uuid(store: PieceStore) -> None:
    """save() generates a UUID when none is provided."""
    piece_id = await store.save(_make_piece())
    assert piece_id is not None
    assert len(piece_id) == 36  # UUID4 string length


async def test_save_uses_provided_id(store: PieceStore) -> None:
    """save() uses the provided id if present."""
    custom_id = "custom-id-12345"
    piece_id = await store.save(_make_piece(id=custom_id))
    assert piece_id == custom_id
    result = await store.get(custom_id)
    assert result is not None


async def test_get_nonexistent_returns_none(store: PieceStore) -> None:
    """Getting a nonexistent piece returns None."""
    assert await store.get("nonexistent-uuid") is None


async def test_list_all_with_limit_offset(store: PieceStore) -> None:
    """list_all respects limit and offset parameters."""
    for i in range(5):
        await store.save(_make_piece(topic=f"Topic {i}"))

    # Get first 2.
    results = await store.list_all(limit=2, offset=0)
    assert len(results) == 2

    # Get next 2.
    results2 = await store.list_all(limit=2, offset=2)
    assert len(results2) == 2

    # Get remaining.
    results3 = await store.list_all(limit=10, offset=4)
    assert len(results3) == 1


async def test_list_all_ordering(store: PieceStore) -> None:
    """list_all returns pieces newest first."""
    id1 = await store.save(_make_piece(topic="First"))
    id2 = await store.save(_make_piece(topic="Second"))
    id3 = await store.save(_make_piece(topic="Third"))

    # Explicitly set created_at so ordering is deterministic.
    db = store._db
    await db.execute(
        "UPDATE pieces SET created_at = '2025-01-01T00:00:00.000Z' WHERE id = ?",
        (id1,),
    )
    await db.execute(
        "UPDATE pieces SET created_at = '2025-06-01T00:00:00.000Z' WHERE id = ?",
        (id2,),
    )
    await db.execute(
        "UPDATE pieces SET created_at = '2025-12-01T00:00:00.000Z' WHERE id = ?",
        (id3,),
    )
    await db.commit()

    results = await store.list_all()
    # Newest should be first.
    assert results[0]["id"] == id3
    assert results[1]["id"] == id2
    assert results[2]["id"] == id1


async def test_search_by_keyword(store: PieceStore) -> None:
    """search() finds pieces by topic keyword."""
    await store.save(_make_piece(topic="Climate change effects"))
    await store.save(_make_piece(topic="Tax reform proposals"))
    await store.save(_make_piece(topic="Climate policy debate"))

    results = await store.search("Climate")
    assert len(results) == 2
    assert all("Climate" in r["topic"] for r in results)


async def test_search_no_results(store: PieceStore) -> None:
    """search() returns empty list when no matches."""
    await store.save(_make_piece(topic="Climate change"))
    results = await store.search("quantum computing")
    assert results == []


async def test_filter_by_mode(store: PieceStore) -> None:
    """filter_by with mode returns matching pieces only."""
    await store.save(_make_piece(mode="analytical"))
    await store.save(_make_piece(mode="polemical"))
    await store.save(_make_piece(mode="analytical"))

    results = await store.filter_by(mode="analytical")
    assert len(results) == 2
    assert all(r["mode"] == "analytical" for r in results)


async def test_filter_by_stance_range(store: PieceStore) -> None:
    """filter_by with stance_min/max filters correctly."""
    await store.save(_make_piece(stance_position=-50))
    await store.save(_make_piece(stance_position=0))
    await store.save(_make_piece(stance_position=50))

    results = await store.filter_by(stance_min=-10, stance_max=10)
    assert len(results) == 1
    assert results[0]["stance_position"] == 0


async def test_filter_by_date_range(store: PieceStore) -> None:
    """filter_by with date_from/date_to filters by created_at."""
    # Insert three pieces; they all get 'now' timestamps.
    id1 = await store.save(_make_piece(topic="Piece A"))
    id2 = await store.save(_make_piece(topic="Piece B"))
    id3 = await store.save(_make_piece(topic="Piece C"))

    # Manually update created_at for testing.
    db = store._db
    await db.execute(
        "UPDATE pieces SET created_at = '2025-01-01T00:00:00.000Z' WHERE id = ?",
        (id1,),
    )
    await db.execute(
        "UPDATE pieces SET created_at = '2025-06-15T00:00:00.000Z' WHERE id = ?",
        (id2,),
    )
    await db.execute(
        "UPDATE pieces SET created_at = '2025-12-31T00:00:00.000Z' WHERE id = ?",
        (id3,),
    )
    await db.commit()

    results = await store.filter_by(
        date_from="2025-03-01T00:00:00.000Z",
        date_to="2025-09-01T00:00:00.000Z",
    )
    assert len(results) == 1
    assert results[0]["id"] == id2


async def test_filter_combined(store: PieceStore) -> None:
    """filter_by with multiple criteria applies all of them."""
    await store.save(_make_piece(mode="analytical", stance_position=10))
    await store.save(_make_piece(mode="analytical", stance_position=80))
    await store.save(_make_piece(mode="polemical", stance_position=10))

    results = await store.filter_by(mode="analytical", stance_min=0, stance_max=50)
    assert len(results) == 1
    assert results[0]["mode"] == "analytical"
    assert results[0]["stance_position"] == 10


async def test_delete_existing_piece(store: PieceStore) -> None:
    """delete() returns True for an existing piece and removes it."""
    piece_id = await store.save(_make_piece())
    assert await store.delete(piece_id) is True
    assert await store.get(piece_id) is None


async def test_delete_nonexistent_returns_false(store: PieceStore) -> None:
    """delete() returns False when the piece does not exist."""
    assert await store.delete("nonexistent-uuid") is False


async def test_bulk_delete(store: PieceStore) -> None:
    """bulk_delete removes multiple pieces and returns the count."""
    ids = []
    for _ in range(4):
        ids.append(await store.save(_make_piece()))

    deleted = await store.bulk_delete(ids[:3])
    assert deleted == 3
    assert await store.count() == 1


async def test_bulk_delete_empty_list(store: PieceStore) -> None:
    """bulk_delete with an empty list returns 0."""
    assert await store.bulk_delete([]) == 0


async def test_count(store: PieceStore) -> None:
    """count() returns the total number of pieces."""
    assert await store.count() == 0
    await store.save(_make_piece())
    await store.save(_make_piece())
    assert await store.count() == 2


async def test_json_field_round_trip(store: PieceStore) -> None:
    """JSON fields are serialised on save and deserialised on get."""
    mode_config = {"modes": [["analytical", 60.0], ["polemical", 40.0]]}
    sources = [{"url": "https://example.com", "title": "Example"}]
    piece_id = await store.save(
        _make_piece(mode_config=mode_config, sources=sources)
    )
    result = await store.get(piece_id)
    assert result is not None
    assert result["mode_config"] == mode_config
    assert result["sources"] == sources
