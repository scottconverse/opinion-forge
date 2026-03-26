"""Tests that settings templates render with expected structure.

Verifies the presence of all required sections, form elements, and
input types in the settings page HTML output.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path) -> TestClient:
    """Return a TestClient with a temporary database."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Test output."
    db_file = str(tmp_path / "template_test.db")
    app = create_app(client=mock_llm, db_path=db_file)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSettingsPageStructure:
    """Verify the settings page contains all required sections and elements."""

    def test_page_has_provider_config_section(self, client: TestClient) -> None:
        """Settings page contains the LLM Provider configuration section."""
        response = client.get("/settings")
        assert response.status_code == 200
        assert "provider-section" in response.text
        assert "LLM Provider" in response.text

    def test_page_has_search_config_section(self, client: TestClient) -> None:
        """Settings page contains the Search Provider section."""
        response = client.get("/settings")
        assert "search-section" in response.text
        assert "Search Provider" in response.text

    def test_page_has_preferences_section(self, client: TestClient) -> None:
        """Settings page contains the Preferences section."""
        response = client.get("/settings")
        assert "preferences-section" in response.text
        assert "Preferences" in response.text

    def test_page_has_data_management_section(self, client: TestClient) -> None:
        """Settings page contains the Data Management section."""
        response = client.get("/settings")
        assert "data-section" in response.text
        assert "Data Management" in response.text

    def test_api_key_inputs_are_password_type(self, client: TestClient) -> None:
        """API key inputs use type=password to mask the value."""
        response = client.get("/settings")
        html = response.text
        # Both API key fields should be password type
        assert 'type="password"' in html
        assert 'id="api_key"' in html
        assert 'id="search_api_key"' in html

    def test_export_and_clear_buttons_exist(self, client: TestClient) -> None:
        """Export and Clear History buttons are present."""
        response = client.get("/settings")
        html = response.text
        assert "Export All Pieces as JSON" in html
        assert "Clear History" in html

    def test_provider_selection_has_four_options(
        self, client: TestClient,
    ) -> None:
        """Provider selection includes all 4 provider types."""
        response = client.get("/settings")
        html = response.text
        assert 'value="ollama"' in html
        assert 'value="anthropic"' in html
        assert 'value="openai"' in html
        assert 'value="openai_compatible"' in html

    def test_mode_dropdown_has_12_modes(self, client: TestClient) -> None:
        """Default mode dropdown contains all 12 modes."""
        response = client.get("/settings")
        html = response.text
        expected_modes = [
            "analytical", "aphoristic", "data_driven", "dialectical",
            "forensic", "measured", "narrative", "oratorical",
            "polemical", "populist", "provocative", "satirical",
        ]
        for mode_id in expected_modes:
            assert f'value="{mode_id}"' in html, (
                f"Mode '{mode_id}' not found in preferences dropdown"
            )

    def test_theme_toggle_present(self, client: TestClient) -> None:
        """Theme toggle (light/dark) is present in preferences."""
        response = client.get("/settings")
        html = response.text
        assert 'value="light"' in html
        assert 'value="dark"' in html

    def test_clear_history_has_confirmation(self, client: TestClient) -> None:
        """Clear History button includes hx-confirm for confirmation."""
        response = client.get("/settings")
        assert "hx-confirm" in response.text

    def test_replay_tour_button_exists(self, client: TestClient) -> None:
        """Replay Tour button exists in the settings page."""
        response = client.get("/settings")
        assert "Replay Tour" in response.text
