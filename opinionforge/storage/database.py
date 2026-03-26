"""SQLite database connection manager with schema creation and migration support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Self

import aiosqlite
from platformdirs import user_data_dir

logger = logging.getLogger(__name__)

# Current schema version — bump this when migrations are added.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# SQL schema definitions
# ---------------------------------------------------------------------------

_CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_PIECES_TABLE = """
CREATE TABLE IF NOT EXISTS pieces (
    id                  TEXT PRIMARY KEY,
    topic               TEXT NOT NULL,
    title               TEXT,
    subtitle            TEXT,
    body                TEXT,
    preview_text        TEXT,
    mode                TEXT,
    mode_config         TEXT,
    stance_position     INTEGER DEFAULT 0,
    stance_intensity    REAL    DEFAULT 0.5,
    target_length       INTEGER,
    actual_length       INTEGER,
    sources             TEXT,
    research_queries    TEXT,
    disclaimer          TEXT,
    screening_details   TEXT,
    exported_formats    TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS settings (
    key       TEXT PRIMARY KEY,
    value     TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_EXPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS exports (
    id         TEXT PRIMARY KEY,
    piece_id   TEXT NOT NULL,
    format     TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (piece_id) REFERENCES pieces(id) ON DELETE CASCADE
);
"""

_ALL_SCHEMA_STATEMENTS = [
    _CREATE_METADATA_TABLE,
    _CREATE_PIECES_TABLE,
    _CREATE_SETTINGS_TABLE,
    _CREATE_EXPORTS_TABLE,
]


def get_db_path() -> Path:
    """Return the platform-appropriate path for the OpinionForge database.

    Uses ``platformdirs.user_data_dir`` so the database lives in the
    standard application-data directory on each OS (e.g.
    ``~/.local/share/opinionforge/opinionforge.db`` on Linux,
    ``~/Library/Application Support/opinionforge/opinionforge.db`` on
    macOS, or ``%LOCALAPPDATA%/opinionforge/opinionforge.db`` on
    Windows).

    Returns:
        A :class:`pathlib.Path` pointing to the database file.
    """
    data_dir = Path(user_data_dir("opinionforge"))
    return data_dir / "opinionforge.db"


class Database:
    """Async SQLite database manager for OpinionForge.

    Supports use as an async context manager::

        async with Database(path) as db:
            ...

    The database file (and parent directories) are created automatically
    on :meth:`initialize`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialise with an optional database path.

        Args:
            path: Filesystem path for the SQLite file.  Use the special
                value ``\":memory:\"`` for an in-memory database.  If
                *None*, the platform default from :func:`get_db_path` is
                used.
        """
        if path is None:
            self.path: str | Path = get_db_path()
        else:
            self.path = path
        self._conn: aiosqlite.Connection | None = None

    # -- async context manager ------------------------------------------------

    async def __aenter__(self) -> Self:
        """Open the database connection and initialise the schema."""
        await self.connect()
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the database connection."""
        await self.close()

    # -- connection lifecycle -------------------------------------------------

    async def connect(self) -> None:
        """Open the underlying ``aiosqlite`` connection.

        Creates the parent directory for the database file if it does
        not already exist (unless using an in-memory database).
        """
        path_str = str(self.path)
        if path_str != ":memory:":
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(path_str)
        # Enable WAL journal mode for concurrent read safety.
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        # Enable foreign key enforcement (off by default in SQLite).
        await self._conn.execute("PRAGMA foreign_keys=ON;")

    async def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # -- schema ---------------------------------------------------------------

    async def initialize(self) -> None:
        """Create all tables and record the schema version.

        This method is idempotent — calling it on an already-initialised
        database is safe and effectively a no-op.
        """
        conn = self.connection
        for stmt in _ALL_SCHEMA_STATEMENTS:
            await conn.executescript(stmt)

        # Record schema version (insert-or-ignore so re-runs don't overwrite).
        await conn.execute(
            "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        await conn.commit()
        logger.info("Database initialised (schema version %s)", SCHEMA_VERSION)

    # -- helpers --------------------------------------------------------------

    @property
    def connection(self) -> aiosqlite.Connection:
        """Return the active connection, raising if not connected.

        Returns:
            The underlying ``aiosqlite.Connection``.

        Raises:
            RuntimeError: If the database is not connected.
        """
        if self._conn is None:
            raise RuntimeError(
                "Database is not connected. Use 'async with Database(...)' "
                "or call connect() first."
            )
        return self._conn

    async def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] | list[object] = (),
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement and return the cursor.

        Args:
            sql: The SQL query string.
            parameters: Bind parameters for the query.

        Returns:
            The resulting ``aiosqlite.Cursor``.
        """
        return await self.connection.execute(sql, parameters)

    async def execute_fetchone(
        self,
        sql: str,
        parameters: tuple[object, ...] | list[object] = (),
    ) -> aiosqlite.Row | None:
        """Execute a query and return a single row.

        Args:
            sql: The SQL query string.
            parameters: Bind parameters for the query.

        Returns:
            The first matching row, or *None* if no rows match.
        """
        cursor = await self.connection.execute(sql, parameters)
        return await cursor.fetchone()

    async def execute_fetchall(
        self,
        sql: str,
        parameters: tuple[object, ...] | list[object] = (),
    ) -> list[aiosqlite.Row]:
        """Execute a query and return all matching rows.

        Args:
            sql: The SQL query string.
            parameters: Bind parameters for the query.

        Returns:
            A list of matching rows.
        """
        cursor = await self.connection.execute(sql, parameters)
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.connection.commit()


async def get_database(path: str | Path | None = None) -> Database:
    """Convenience function that returns a connected and initialised Database.

    Args:
        path: Optional database path.  Defaults to the platform path.

    Returns:
        A :class:`Database` instance that is connected and initialised.
        The caller is responsible for calling :meth:`Database.close`
        when finished.
    """
    db = Database(path)
    await db.connect()
    await db.initialize()
    return db
