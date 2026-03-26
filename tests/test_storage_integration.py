"""Integration tests for cross-cutting storage operations.

Tests workflows that span multiple storage stores (pieces, exports,
settings) operating on a shared database, verifying referential
integrity, cascade behaviour, and data persistence across the full
storage subsystem.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from opinionforge.models.config import ProviderConfig, UserPreferences
from opinionforge.storage.database import Database
from opinionforge.storage.exports import ExportStore
from opinionforge.storage.pieces import PieceStore
from opinionforge.storage.settings import SettingsStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path):
    """Provide a connected and initialised temp Database."""
    db_path = tmp_path / "integration_test.db"
    database = Database(db_path)
    await database.connect()
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def pieces(db: Database) -> PieceStore:
    return PieceStore(db)


@pytest.fixture
def exports(db: Database) -> ExportStore:
    return ExportStore(db)


@pytest.fixture
def settings(db: Database) -> SettingsStore:
    return SettingsStore(db)


def _make_piece(topic: str = "Test topic", mode: str = "analytical", **overrides) -> dict:
    """Build a minimal piece data dict with optional overrides."""
    data = {
        "topic": topic,
        "title": f"Title: {topic}",
        "body": f"Body for {topic}",
        "mode": mode,
        "stance_position": 0,
        "stance_intensity": 0.5,
        "disclaimer": "AI-generated content.",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Cross-store workflow tests
# ---------------------------------------------------------------------------


class TestPieceExportWorkflow:
    """Create pieces, export them, verify linkage and cascade."""

    @pytest.mark.asyncio
    async def test_create_piece_then_export_multiple_formats(
        self, pieces: PieceStore, exports: ExportStore
    ):
        piece_id = await pieces.save(_make_piece("Climate policy"))

        md_id = await exports.save(piece_id, "markdown", "# Climate policy\n...")
        html_id = await exports.save(piece_id, "html", "<h1>Climate policy</h1>")
        tw_id = await exports.save(piece_id, "twitter", "1/ Climate policy thread...")

        piece_exports = await exports.get_by_piece(piece_id)
        assert len(piece_exports) == 3
        format_set = {e["format"] for e in piece_exports}
        assert format_set == {"markdown", "html", "twitter"}

        # Each export references the correct piece
        for exp in piece_exports:
            assert exp["piece_id"] == piece_id

    @pytest.mark.asyncio
    async def test_delete_piece_cascades_to_exports(
        self, pieces: PieceStore, exports: ExportStore
    ):
        piece_id = await pieces.save(_make_piece("Cascade test"))
        export_id = await exports.save(piece_id, "markdown", "# Content")

        # Export exists before delete
        assert await exports.get(export_id) is not None

        # Delete the piece — CASCADE should remove exports
        deleted = await pieces.delete(piece_id)
        assert deleted is True

        # Export should be gone
        assert await exports.get(export_id) is None
        assert await exports.get_by_piece(piece_id) == []

    @pytest.mark.asyncio
    async def test_bulk_delete_pieces_cascades_exports(
        self, pieces: PieceStore, exports: ExportStore
    ):
        id1 = await pieces.save(_make_piece("Piece one"))
        id2 = await pieces.save(_make_piece("Piece two"))
        await exports.save(id1, "markdown", "content 1")
        await exports.save(id2, "html", "content 2")

        deleted_count = await pieces.bulk_delete([id1, id2])
        assert deleted_count == 2

        assert await exports.get_by_piece(id1) == []
        assert await exports.get_by_piece(id2) == []

    @pytest.mark.asyncio
    async def test_orphan_export_rejected(self, exports: ExportStore):
        """Exports with nonexistent piece_id must raise IntegrityError."""
        with pytest.raises(aiosqlite.IntegrityError):
            await exports.save("nonexistent-piece-id", "markdown", "orphan")

    @pytest.mark.asyncio
    async def test_multiple_pieces_independent_exports(
        self, pieces: PieceStore, exports: ExportStore
    ):
        id1 = await pieces.save(_make_piece("Article A", mode="polemical"))
        id2 = await pieces.save(_make_piece("Article B", mode="satirical"))

        await exports.save(id1, "substack", "substack content A")
        await exports.save(id2, "medium", "medium content B")
        await exports.save(id2, "wordpress", "wordpress content B")

        assert len(await exports.get_by_piece(id1)) == 1
        assert len(await exports.get_by_piece(id2)) == 2


class TestSettingsPiecesInteraction:
    """Settings and pieces coexist independently in the same database."""

    @pytest.mark.asyncio
    async def test_preferences_and_pieces_independent(
        self, pieces: PieceStore, settings: SettingsStore
    ):
        # Save preferences
        prefs = UserPreferences(
            default_mode="polemical",
            default_stance=-30,
            default_intensity=0.8,
            theme="dark",
        )
        await settings.set_user_preferences(prefs)

        # Save a piece (using the same db)
        piece_id = await pieces.save(
            _make_piece("Independence test", mode="polemical", stance_position=-30)
        )

        # Both persist independently
        stored_prefs = await settings.get_user_preferences()
        assert stored_prefs.default_mode == "polemical"
        assert stored_prefs.theme == "dark"

        stored_piece = await pieces.get(piece_id)
        assert stored_piece is not None
        assert stored_piece["mode"] == "polemical"

        # Updating preferences does not affect pieces
        prefs.default_mode = "analytical"
        await settings.set_user_preferences(prefs)

        stored_piece_after = await pieces.get(piece_id)
        assert stored_piece_after["mode"] == "polemical"  # unchanged

    @pytest.mark.asyncio
    async def test_provider_config_persists_across_piece_operations(
        self, pieces: PieceStore, settings: SettingsStore
    ):
        config = ProviderConfig(
            provider_type="anthropic",
            model="claude-3-opus-20240229",
            api_key="sk-test-key",
            max_tokens=8192,
        )
        await settings.set_provider_config(config)

        # Piece CRUD should not affect provider config
        pid = await pieces.save(_make_piece("Provider test"))
        await pieces.delete(pid)

        stored_config = await settings.get_provider_config()
        assert stored_config is not None
        assert stored_config.provider_type == "anthropic"
        assert stored_config.model == "claude-3-opus-20240229"
        assert stored_config.max_tokens == 8192


class TestFullWorkflow:
    """End-to-end storage workflows combining all three stores."""

    @pytest.mark.asyncio
    async def test_complete_lifecycle(
        self, pieces: PieceStore, exports: ExportStore, settings: SettingsStore
    ):
        """Simulate: configure → create → export → query → delete."""
        # 1. Set user preferences
        prefs = UserPreferences(default_mode="forensic", default_intensity=0.7)
        await settings.set_user_preferences(prefs)

        # 2. Create pieces
        id1 = await pieces.save(
            _make_piece("Corporate lobbying", mode="forensic", stance_position=-40)
        )
        id2 = await pieces.save(
            _make_piece("Tech regulation", mode="satirical", stance_position=10)
        )
        id3 = await pieces.save(
            _make_piece("Climate legislation", mode="forensic", stance_position=-60)
        )

        assert await pieces.count() == 3

        # 3. Export selected pieces
        await exports.save(id1, "substack", "Substack: Corporate lobbying")
        await exports.save(id1, "twitter", "1/ Corporate lobbying thread")
        await exports.save(id3, "medium", "Medium: Climate legislation")

        # 4. Filter by mode
        forensic_pieces = await pieces.filter_by(mode="forensic")
        assert len(forensic_pieces) == 2
        topics = {p["topic"] for p in forensic_pieces}
        assert topics == {"Corporate lobbying", "Climate legislation"}

        # 5. Filter by stance range
        left_pieces = await pieces.filter_by(stance_min=-100, stance_max=-30)
        assert len(left_pieces) == 2  # -40 and -60

        # 6. Search by keyword
        climate_results = await pieces.search("Climate")
        assert len(climate_results) == 1
        assert climate_results[0]["id"] == id3

        # 7. Delete one piece — its exports cascade
        await pieces.delete(id1)
        assert await pieces.count() == 2
        assert await exports.get_by_piece(id1) == []

        # 8. Remaining exports still intact
        remaining = await exports.get_by_piece(id3)
        assert len(remaining) == 1
        assert remaining[0]["format"] == "medium"

        # 9. Settings still intact
        final_prefs = await settings.get_user_preferences()
        assert final_prefs.default_mode == "forensic"

    @pytest.mark.asyncio
    async def test_search_then_export_all_results(
        self, pieces: PieceStore, exports: ExportStore
    ):
        """Search for pieces by keyword, then export every match."""
        await pieces.save(_make_piece("Immigration and labor"))
        await pieces.save(_make_piece("Immigration policy reform"))
        await pieces.save(_make_piece("Climate and energy"))

        results = await pieces.search("Immigration")
        assert len(results) == 2

        for piece in results:
            await exports.save(piece["id"], "markdown", f"Export of {piece['topic']}")

        for piece in results:
            piece_exports = await exports.get_by_piece(piece["id"])
            assert len(piece_exports) == 1

    @pytest.mark.asyncio
    async def test_filter_by_multiple_criteria(self, pieces: PieceStore):
        await pieces.save(_make_piece("A", mode="analytical", stance_position=0))
        await pieces.save(_make_piece("B", mode="analytical", stance_position=50))
        await pieces.save(_make_piece("C", mode="polemical", stance_position=0))
        await pieces.save(_make_piece("D", mode="analytical", stance_position=-30))

        # Mode + stance range
        results = await pieces.filter_by(
            mode="analytical", stance_min=-10, stance_max=10
        )
        assert len(results) == 1
        assert results[0]["topic"] == "A"


class TestDatabasePersistence:
    """Verify data survives close/reopen cycles."""

    @pytest.mark.asyncio
    async def test_data_persists_across_sessions(self, tmp_path: Path):
        db_path = tmp_path / "persist_test.db"

        # Session 1: write data
        async with Database(db_path) as db1:
            ps = PieceStore(db1)
            ss = SettingsStore(db1)
            piece_id = await ps.save(_make_piece("Persistence test"))
            await ss.set_user_preferences(
                UserPreferences(default_mode="narrative", theme="dark")
            )

        # Session 2: read it back
        async with Database(db_path) as db2:
            ps2 = PieceStore(db2)
            ss2 = SettingsStore(db2)

            piece = await ps2.get(piece_id)
            assert piece is not None
            assert piece["topic"] == "Persistence test"

            prefs = await ss2.get_user_preferences()
            assert prefs.default_mode == "narrative"
            assert prefs.theme == "dark"

    @pytest.mark.asyncio
    async def test_exports_persist_with_pieces_across_sessions(self, tmp_path: Path):
        db_path = tmp_path / "export_persist.db"

        async with Database(db_path) as db1:
            ps = PieceStore(db1)
            es = ExportStore(db1)
            pid = await ps.save(_make_piece("Export persist"))
            eid = await es.save(pid, "substack", "Substack content")

        async with Database(db_path) as db2:
            es2 = ExportStore(db2)
            export = await es2.get(eid)
            assert export is not None
            assert export["format"] == "substack"
            assert export["piece_id"] == pid


class TestJsonRoundTrip:
    """Verify JSON fields survive the full storage round-trip in integration context."""

    @pytest.mark.asyncio
    async def test_complex_mode_config_round_trip(self, pieces: PieceStore):
        mode_config = {
            "modes": [["polemical", 60.0], ["analytical", 40.0]],
            "blend_strategy": "weighted",
        }
        sources = [
            {"url": "https://example.com/a", "title": "Source A"},
            {"url": "https://example.com/b", "title": "Source B"},
        ]
        piece_id = await pieces.save(
            _make_piece(
                "JSON test",
                mode_config=mode_config,
                sources=sources,
                screening_details={"passed": True, "score": 0.95},
                exported_formats=["substack", "medium"],
            )
        )

        stored = await pieces.get(piece_id)
        assert stored["mode_config"] == mode_config
        assert stored["sources"] == sources
        assert stored["screening_details"]["passed"] is True
        assert stored["exported_formats"] == ["substack", "medium"]

    @pytest.mark.asyncio
    async def test_settings_json_round_trip_with_all_fields(
        self, settings: SettingsStore
    ):
        prefs = UserPreferences(
            default_mode="satirical",
            default_stance=40,
            default_intensity=0.9,
            default_length="essay",
            theme="dark",
            auto_launch=True,
            onboarding_completed=True,
        )
        await settings.set_user_preferences(prefs)

        stored = await settings.get_user_preferences()
        assert stored.default_mode == "satirical"
        assert stored.default_stance == 40
        assert stored.default_intensity == 0.9
        assert stored.default_length == "essay"
        assert stored.theme == "dark"
        assert stored.auto_launch is True
        assert stored.onboarding_completed is True


class TestConcurrentTableOperations:
    """Multiple stores operating on the same database concurrently."""

    @pytest.mark.asyncio
    async def test_interleaved_operations(
        self, pieces: PieceStore, exports: ExportStore, settings: SettingsStore
    ):
        """Interleave writes across all three tables in a single session."""
        # Write to settings
        await settings.set("custom_key", "custom_value")

        # Write a piece
        pid = await pieces.save(_make_piece("Interleaved"))

        # Write to settings again
        await settings.set_provider_config(
            ProviderConfig(provider_type="openai", model="gpt-4")
        )

        # Export from the piece
        await exports.save(pid, "markdown", "# Interleaved")

        # Write another piece
        pid2 = await pieces.save(_make_piece("Second"))

        # Verify all data is intact
        assert await settings.get("custom_key") == "custom_value"
        assert (await settings.get_provider_config()).model == "gpt-4"
        assert await pieces.count() == 2
        assert len(await exports.get_by_piece(pid)) == 1

    @pytest.mark.asyncio
    async def test_settings_raw_and_typed_coexist(self, settings: SettingsStore):
        """Raw key-value settings and typed helpers share the table without conflict."""
        await settings.set("app_version", "2.0.0")
        await settings.set_user_preferences(UserPreferences(theme="dark"))
        await settings.set_provider_config(
            ProviderConfig(provider_type="ollama", model="llama3")
        )

        all_settings = await settings.get_all()
        assert "app_version" in all_settings
        assert "user_preferences" in all_settings
        assert "provider_config" in all_settings

        # Typed access still works
        assert (await settings.get_user_preferences()).theme == "dark"
        assert (await settings.get_provider_config()).model == "llama3"
        assert await settings.get("app_version") == "2.0.0"
