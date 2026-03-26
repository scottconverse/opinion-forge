"""Tests for database initialization, schema creation, and migration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from opinionforge.storage.database import Database, SCHEMA_VERSION, get_database, get_db_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Provide a connected and initialised in-memory-style temp Database."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    await database.connect()
    await database.initialize()
    yield database
    await database.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_database_creation_at_specified_path(tmp_path: Path) -> None:
    """Database file is created at the specified path."""
    db_path = tmp_path / "custom.db"
    async with Database(db_path) as db:
        assert db_path.exists()


async def test_tables_exist_after_initialize(db: Database) -> None:
    """All expected tables exist after initialize."""
    rows = await db.execute_fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    table_names = sorted(row[0] for row in rows)
    assert "exports" in table_names
    assert "metadata" in table_names
    assert "pieces" in table_names
    assert "settings" in table_names


async def test_schema_version_tracked(db: Database) -> None:
    """Schema version is recorded in the metadata table."""
    row = await db.execute_fetchone(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    )
    assert row is not None
    assert row[0] == str(SCHEMA_VERSION)


async def test_wal_mode_enabled(tmp_path: Path) -> None:
    """WAL journal mode is active after connection."""
    db_path = tmp_path / "wal_test.db"
    async with Database(db_path) as db:
        row = await db.execute_fetchone("PRAGMA journal_mode")
        assert row is not None
        assert row[0].lower() == "wal"


async def test_concurrent_reads(db: Database) -> None:
    """Multiple concurrent reads do not raise errors."""
    # Insert a row so reads have something to find.
    await db.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)", ("k1", "v1")
    )
    await db.commit()

    # Perform several reads concurrently.
    results = await asyncio.gather(
        db.execute_fetchone("SELECT value FROM settings WHERE key = ?", ("k1",)),
        db.execute_fetchone("SELECT value FROM settings WHERE key = ?", ("k1",)),
        db.execute_fetchone("SELECT value FROM settings WHERE key = ?", ("k1",)),
    )
    assert all(r is not None and r[0] == "v1" for r in results)


async def test_directory_auto_creation(tmp_path: Path) -> None:
    """Parent directories are created automatically if missing."""
    deep_path = tmp_path / "a" / "b" / "c" / "test.db"
    assert not deep_path.parent.exists()
    async with Database(deep_path) as db:
        pass
    assert deep_path.exists()


def test_get_db_path_returns_platform_path() -> None:
    """get_db_path returns a path ending in opinionforge.db."""
    path = get_db_path()
    assert path.name == "opinionforge.db"
    # The parent should contain 'opinionforge' somewhere in the path.
    assert "opinionforge" in str(path).lower()


async def test_idempotent_initialize(tmp_path: Path) -> None:
    """Calling initialize twice is safe (no errors, schema unchanged)."""
    db_path = tmp_path / "idem.db"
    async with Database(db_path) as db:
        # initialize was already called by __aenter__; call it again.
        await db.initialize()
        row = await db.execute_fetchone(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        )
        assert row is not None
        assert row[0] == str(SCHEMA_VERSION)


async def test_database_file_created_on_disk(tmp_path: Path) -> None:
    """The .db file physically exists after closing."""
    db_path = tmp_path / "ondisk.db"
    async with Database(db_path):
        pass
    assert db_path.exists()
    assert db_path.stat().st_size > 0


async def test_in_memory_database() -> None:
    """An in-memory database (\":memory:\") works correctly."""
    async with Database(":memory:") as db:
        rows = await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [row[0] for row in rows]
        assert "pieces" in table_names


async def test_get_database_convenience(tmp_path: Path) -> None:
    """get_database returns a connected and initialised Database."""
    db_path = tmp_path / "convenience.db"
    db = await get_database(db_path)
    try:
        rows = await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [row[0] for row in rows]
        assert "pieces" in table_names
    finally:
        await db.close()


async def test_context_manager_closes_connection(tmp_path: Path) -> None:
    """Connection is closed after exiting the context manager."""
    db_path = tmp_path / "close_test.db"
    db = Database(db_path)
    async with db:
        assert db._conn is not None
    assert db._conn is None


async def test_foreign_keys_enabled(db: Database) -> None:
    """Foreign key constraints are enforced."""
    row = await db.execute_fetchone("PRAGMA foreign_keys")
    assert row is not None
    assert row[0] == 1
