"""CRUD operations for the exports table."""

from __future__ import annotations

import uuid
from typing import Any

from opinionforge.storage.database import Database

_EXPORT_COLUMNS = ["id", "piece_id", "format", "content", "created_at"]


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an exports row tuple to a dictionary.

    Args:
        row: A row tuple from an ``aiosqlite`` query.

    Returns:
        A dictionary mapping column names to their values.
    """
    return {col: row[idx] for idx, col in enumerate(_EXPORT_COLUMNS)}


class ExportStore:
    """CRUD operations for piece export records.

    Each export is associated with a piece via a foreign key.  The
    ``piece_id`` must reference an existing row in the ``pieces`` table
    (enforced by the database's foreign key constraint).

    Args:
        db: An initialised and connected :class:`Database`.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, piece_id: str, format: str, content: str) -> str:
        """Insert a new export record and return its UUID.

        Args:
            piece_id: The UUID of the parent piece.
            format: The export format identifier (e.g. ``'markdown'``).
            content: The exported content body.

        Returns:
            The auto-generated UUID for this export.

        Raises:
            aiosqlite.IntegrityError: If ``piece_id`` does not reference
                an existing piece (foreign key violation).
        """
        export_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO exports (id, piece_id, format, content) "
            "VALUES (?, ?, ?, ?)",
            (export_id, piece_id, format, content),
        )
        await self._db.commit()
        return export_id

    async def get(self, export_id: str) -> dict[str, Any] | None:
        """Retrieve a specific export by UUID.

        Args:
            export_id: The unique identifier of the export.

        Returns:
            A dictionary of export data, or *None* if not found.
        """
        row = await self._db.execute_fetchone(
            "SELECT * FROM exports WHERE id = ?", (export_id,)
        )
        if row is None:
            return None
        return _row_to_dict(row)

    async def get_by_piece(self, piece_id: str) -> list[dict[str, Any]]:
        """Return all exports associated with a given piece.

        Args:
            piece_id: The UUID of the parent piece.

        Returns:
            A list of export dictionaries, ordered by creation time.
        """
        rows = await self._db.execute_fetchall(
            "SELECT * FROM exports WHERE piece_id = ? ORDER BY created_at",
            (piece_id,),
        )
        return [_row_to_dict(r) for r in rows]

    async def delete(self, export_id: str) -> bool:
        """Delete an export record.

        Args:
            export_id: The UUID of the export to remove.

        Returns:
            *True* if the export existed and was deleted, *False* otherwise.
        """
        cursor = await self._db.execute(
            "DELETE FROM exports WHERE id = ?", (export_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
