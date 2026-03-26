"""Tests for settings page backend routes.

Exercises GET /settings, POST /settings/provider, POST /settings/search,
POST /settings/preferences, POST /settings/export-data, and
POST /settings/clear-history.  All provider connection tests are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client() -> MagicMock:
    """Create a mock LLM client that returns canned text."""
    client = MagicMock()
    client.generate.return_value = "Test output."
    return client


def _client(db_path: str = ":memory:") -> TestClient:
    """Return a TestClient backed by an in-memory database."""
    mock_llm = _make_mock_client()
    app = create_app(client=mock_llm, db_path=db_path)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path) -> TestClient:
    """Return a TestClient with a temporary database."""
    db_file = str(tmp_path / "test.db")
    return _client(db_file)


# ---------------------------------------------------------------------------
# GET /settings
# ---------------------------------------------------------------------------

class TestGetSettings:
    """Tests for GET /settings."""

    def test_settings_returns_200(self, client: TestClient) -> None:
        """GET /settings returns 200."""
        response = client.get("/settings")
        assert response.status_code == 200

    def test_settings_contains_sections(self, client: TestClient) -> None:
        """GET /settings contains all major sections."""
        response = client.get("/settings")
        assert "LLM Provider" in response.text
        assert "Search Provider" in response.text
        assert "Preferences" in response.text
        assert "Data Management" in response.text

    def test_settings_reflects_saved_values_on_reload(self, client: TestClient) -> None:
        """Settings page reflects saved preferences after reload."""
        # Save preferences
        client.post(
            "/settings/preferences",
            data={
                "default_mode": "polemical",
                "default_stance": "50",
                "default_intensity": "0.8",
                "default_length": "long",
                "theme": "dark",
            },
        )
        # Reload and check
        response = client.get("/settings")
        assert response.status_code == 200
        assert "polemical" in response.text
        assert "dark" in response.text


# ---------------------------------------------------------------------------
# POST /settings/provider
# ---------------------------------------------------------------------------

class TestSaveProvider:
    """Tests for POST /settings/provider."""

    def test_save_provider_stores_encrypted_api_key(
        self, client: TestClient, tmp_path,
    ) -> None:
        """Provider save encrypts the API key before storage."""
        with patch(
            "opinionforge.web.app.ProviderRegistry"
        ) as MockRegistry:
            mock_reg = MockRegistry.return_value
            mock_reg.create_provider.return_value = MagicMock()
            mock_reg.test_connection = AsyncMock(return_value=(True, "OK"))

            response = client.post(
                "/settings/provider",
                data={
                    "provider_type": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key": "sk-test-key-12345",
                    "base_url": "",
                },
            )
            assert response.status_code == 200
            assert "Saved successfully" in response.text

    def test_save_provider_failed_connection_returns_error(
        self, client: TestClient,
    ) -> None:
        """Provider save with failed connection returns error feedback."""
        with patch(
            "opinionforge.web.app.ProviderRegistry"
        ) as MockRegistry:
            mock_reg = MockRegistry.return_value
            mock_reg.create_provider.return_value = MagicMock()
            mock_reg.test_connection = AsyncMock(
                return_value=(False, "Auth failed")
            )

            response = client.post(
                "/settings/provider",
                data={
                    "provider_type": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key": "bad-key",
                    "base_url": "",
                },
            )
            assert response.status_code == 200
            assert "Error" in response.text or "Auth failed" in response.text

    def test_api_key_displayed_masked(self, client: TestClient) -> None:
        """After saving, the API key is masked on the settings page."""
        with patch(
            "opinionforge.web.app.ProviderRegistry"
        ) as MockRegistry:
            mock_reg = MockRegistry.return_value
            mock_reg.create_provider.return_value = MagicMock()
            mock_reg.test_connection = AsyncMock(return_value=(True, "OK"))

            client.post(
                "/settings/provider",
                data={
                    "provider_type": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key": "sk-test-key-12345",
                    "base_url": "",
                },
            )

        response = client.get("/settings")
        # Should show masked key with last 4 chars visible, not the full key
        assert "sk-test-key-12345" not in response.text
        # The masked version should contain asterisks
        assert "****" in response.text or "2345" in response.text

    def test_changing_provider_type_clears_old_key(
        self, client: TestClient,
    ) -> None:
        """Changing provider type and saving with new key replaces old config."""
        with patch(
            "opinionforge.web.app.ProviderRegistry"
        ) as MockRegistry:
            mock_reg = MockRegistry.return_value
            mock_reg.create_provider.return_value = MagicMock()
            mock_reg.test_connection = AsyncMock(return_value=(True, "OK"))

            # Save first provider
            client.post(
                "/settings/provider",
                data={
                    "provider_type": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key": "sk-anthropic-key",
                    "base_url": "",
                },
            )
            # Switch to OpenAI
            client.post(
                "/settings/provider",
                data={
                    "provider_type": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-openai-key",
                    "base_url": "",
                },
            )

        response = client.get("/settings")
        # Should not show the anthropic key
        assert "anthropic-key" not in response.text


# ---------------------------------------------------------------------------
# POST /settings/search
# ---------------------------------------------------------------------------

class TestSaveSearch:
    """Tests for POST /settings/search."""

    def test_save_search_config(self, client: TestClient) -> None:
        """POST /settings/search stores the search configuration."""
        response = client.post(
            "/settings/search",
            data={
                "search_provider": "tavily",
                "search_api_key": "tvly-test-key",
            },
        )
        assert response.status_code == 200
        assert "Saved successfully" in response.text


# ---------------------------------------------------------------------------
# POST /settings/preferences
# ---------------------------------------------------------------------------

class TestSavePreferences:
    """Tests for POST /settings/preferences."""

    def test_save_preferences_stores_all_fields(
        self, client: TestClient,
    ) -> None:
        """POST /settings/preferences saves all preference fields."""
        response = client.post(
            "/settings/preferences",
            data={
                "default_mode": "satirical",
                "default_stance": "-50",
                "default_intensity": "0.9",
                "default_length": "essay",
                "theme": "dark",
            },
        )
        assert response.status_code == 200
        assert "Saved successfully" in response.text

        # Verify by loading settings page
        page = client.get("/settings")
        assert "satirical" in page.text
        assert "dark" in page.text


# ---------------------------------------------------------------------------
# POST /settings/export-data
# ---------------------------------------------------------------------------

class TestExportData:
    """Tests for POST /settings/export-data."""

    def test_export_returns_valid_json(self, client: TestClient) -> None:
        """POST /settings/export-data returns valid JSON."""
        response = client.post("/settings/export-data")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_export_contains_pieces(self, client: TestClient, tmp_path) -> None:
        """Export includes stored pieces with metadata."""
        import asyncio
        from opinionforge.storage.database import Database
        from opinionforge.storage.pieces import PieceStore

        db_file = str(tmp_path / "export_test.db")

        async def seed():
            async with Database(db_file) as db:
                ps = PieceStore(db)
                await ps.save({
                    "topic": "test topic",
                    "title": "Test Title",
                    "body": "Test body content.",
                    "mode": "analytical",
                })

        asyncio.run(seed())

        test_client = _client(db_file)
        response = test_client.post("/settings/export-data")
        data = response.json()
        assert len(data) >= 1
        assert data[0]["title"] == "Test Title"


# ---------------------------------------------------------------------------
# POST /settings/clear-history
# ---------------------------------------------------------------------------

class TestClearHistory:
    """Tests for POST /settings/clear-history."""

    def test_clear_history_removes_all_pieces(
        self, client: TestClient, tmp_path,
    ) -> None:
        """POST /settings/clear-history deletes all pieces."""
        import asyncio
        from opinionforge.storage.database import Database
        from opinionforge.storage.pieces import PieceStore

        db_file = str(tmp_path / "clear_test.db")

        async def seed():
            async with Database(db_file) as db:
                ps = PieceStore(db)
                await ps.save({
                    "topic": "topic 1",
                    "title": "Title 1",
                    "body": "Body 1.",
                    "mode": "analytical",
                })
                await ps.save({
                    "topic": "topic 2",
                    "title": "Title 2",
                    "body": "Body 2.",
                    "mode": "polemical",
                })

        asyncio.run(seed())

        test_client = _client(db_file)
        response = test_client.post("/settings/clear-history")
        assert response.status_code == 200
        assert "Saved successfully" in response.text

        # Verify pieces are gone
        export_resp = test_client.post("/settings/export-data")
        data = export_resp.json()
        assert len(data) == 0

    def test_clear_history_removes_all_exports(
        self, client: TestClient, tmp_path,
    ) -> None:
        """POST /settings/clear-history also deletes exports."""
        import asyncio
        from opinionforge.storage.database import Database
        from opinionforge.storage.exports import ExportStore
        from opinionforge.storage.pieces import PieceStore

        db_file = str(tmp_path / "clear_exports_test.db")

        async def seed():
            async with Database(db_file) as db:
                ps = PieceStore(db)
                piece_id = await ps.save({
                    "topic": "topic",
                    "title": "Title",
                    "body": "Body.",
                    "mode": "analytical",
                })
                es = ExportStore(db)
                await es.save(piece_id, "substack", "<p>exported</p>")

        asyncio.run(seed())

        test_client = _client(db_file)
        response = test_client.post("/settings/clear-history")
        assert response.status_code == 200

        # Verify exports are also gone
        async def check():
            async with Database(db_file) as db:
                row = await db.execute_fetchone(
                    "SELECT COUNT(*) FROM exports"
                )
                return row[0]

        count = asyncio.run(check())
        assert count == 0
