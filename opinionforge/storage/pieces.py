"""CRUD operations for the pieces table."""

from __future__ import annotations

import json
import uuid
from typing import Any

from opinionforge.storage.database import Database

# Columns returned from SELECT * on the pieces table (in order).
_PIECE_COLUMNS = [
    "id",
    "topic",
    "title",
    "subtitle",
    "body",
    "preview_text",
    "mode",
    "mode_config",
    "stance_position",
    "stance_intensity",
    "target_length",
    "actual_length",
    "sources",
    "research_queries",
    "disclaimer",
    "screening_details",
    "exported_formats",
    "created_at",
    "updated_at",
]

# Fields that are stored as JSON text in the database.
_JSON_FIELDS = frozenset(
    {"mode_config", "sources", "research_queries", "screening_details", "exported_formats"}
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a database row to a dictionary, deserialising JSON fields.

    Args:
        row: A row tuple from an ``aiosqlite`` query.

    Returns:
        A dictionary mapping column names to their values, with JSON
        fields automatically parsed.
    """
    d: dict[str, Any] = {}
    for idx, col in enumerate(_PIECE_COLUMNS):
        value = row[idx]
        if col in _JSON_FIELDS and isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        d[col] = value
    return d


def _serialize_json(value: Any) -> str | None:
    """Serialise a value to a JSON string, or return None for None.

    Args:
        value: The value to serialise.

    Returns:
        A JSON string, or *None* if the input is *None*.
    """
    if value is None:
        return None
    return json.dumps(value)


class PieceStore:
    """CRUD operations for opinion pieces.

    All methods are async and operate on the ``pieces`` table via the
    provided :class:`Database` instance.

    Args:
        db: An initialised and connected :class:`Database`.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, piece_data: dict[str, Any]) -> str:
        """Insert a new piece and return its UUID.

        If the dictionary does not contain an ``'id'`` key a new UUID4
        is generated automatically.

        Args:
            piece_data: A dictionary of piece fields.  JSON-serialisable
                fields (``mode_config``, ``sources``, etc.) are
                automatically converted to JSON strings for storage.

        Returns:
            The piece's UUID string.
        """
        piece_id = piece_data.get("id") or str(uuid.uuid4())
        data = dict(piece_data)
        data["id"] = piece_id

        # Serialise JSON fields.
        for field in _JSON_FIELDS:
            if field in data and data[field] is not None:
                data[field] = _serialize_json(data[field])

        columns = [col for col in _PIECE_COLUMNS if col in data]
        placeholders = ", ".join("?" for _ in columns)
        col_names = ", ".join(columns)
        values = [data[col] for col in columns]

        await self._db.execute(
            f"INSERT INTO pieces ({col_names}) VALUES ({placeholders})",
            tuple(values),
        )
        await self._db.commit()
        return piece_id

    async def get(self, piece_id: str) -> dict[str, Any] | None:
        """Retrieve a piece by its UUID.

        Args:
            piece_id: The unique identifier of the piece.

        Returns:
            A dictionary of piece data, or *None* if not found.
        """
        row = await self._db.execute_fetchone(
            "SELECT * FROM pieces WHERE id = ?", (piece_id,)
        )
        if row is None:
            return None
        return _row_to_dict(row)

    async def list_all(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return pieces ordered by ``created_at`` descending.

        Args:
            limit: Maximum number of pieces to return.
            offset: Number of pieces to skip (for pagination).

        Returns:
            A list of piece dictionaries.
        """
        rows = await self._db.execute_fetchall(
            "SELECT * FROM pieces ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [_row_to_dict(r) for r in rows]

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search pieces by topic keyword (LIKE match).

        Args:
            query: The search keyword.

        Returns:
            A list of matching piece dictionaries.
        """
        rows = await self._db.execute_fetchall(
            "SELECT * FROM pieces WHERE topic LIKE ? ORDER BY created_at DESC",
            (f"%{query}%",),
        )
        return [_row_to_dict(r) for r in rows]

    async def filter_by(
        self,
        mode: str | None = None,
        stance_min: int | None = None,
        stance_max: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter pieces with optional criteria.

        All parameters are optional; only supplied criteria are applied.

        Args:
            mode: Filter by rhetorical mode.
            stance_min: Minimum stance position (inclusive).
            stance_max: Maximum stance position (inclusive).
            date_from: Earliest ``created_at`` (ISO string, inclusive).
            date_to: Latest ``created_at`` (ISO string, inclusive).

        Returns:
            A list of matching piece dictionaries ordered newest first.
        """
        clauses: list[str] = []
        params: list[object] = []

        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)
        if stance_min is not None:
            clauses.append("stance_position >= ?")
            params.append(stance_min)
        if stance_max is not None:
            clauses.append("stance_position <= ?")
            params.append(stance_max)
        if date_from is not None:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("created_at <= ?")
            params.append(date_to)

        where = " AND ".join(clauses) if clauses else "1=1"
        rows = await self._db.execute_fetchall(
            f"SELECT * FROM pieces WHERE {where} ORDER BY created_at DESC",
            tuple(params),
        )
        return [_row_to_dict(r) for r in rows]

    async def delete(self, piece_id: str) -> bool:
        """Delete a piece by UUID.

        Args:
            piece_id: The unique identifier of the piece.

        Returns:
            *True* if a piece was deleted, *False* if it did not exist.
        """
        cursor = await self._db.execute(
            "DELETE FROM pieces WHERE id = ?", (piece_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def bulk_delete(self, piece_ids: list[str]) -> int:
        """Delete multiple pieces and return the count deleted.

        Args:
            piece_ids: A list of piece UUIDs to delete.

        Returns:
            The number of pieces actually deleted.
        """
        if not piece_ids:
            return 0
        placeholders = ", ".join("?" for _ in piece_ids)
        cursor = await self._db.execute(
            f"DELETE FROM pieces WHERE id IN ({placeholders})",
            tuple(piece_ids),
        )
        await self._db.commit()
        return cursor.rowcount

    async def count(self) -> int:
        """Return the total number of pieces.

        Returns:
            An integer count.
        """
        row = await self._db.execute_fetchone("SELECT COUNT(*) FROM pieces")
        return row[0] if row else 0
