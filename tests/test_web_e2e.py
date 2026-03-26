"""End-to-end tests for the OpinionForge web UI.

Validates that the web application serves correctly, all templates render
without errors, expected page elements are present, and the serve CLI
command is registered.

All LLM and search calls are mocked — zero real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_client() -> MagicMock:
    """Create a mock LLM client that returns canned text."""
    client = MagicMock()
    client.generate.return_value = (
        "The intersection of technology and governance presents "
        "one of the most consequential challenges of our era."
    )
    return client


def _make_mock_provider() -> MagicMock:
    """Create a mock LLM provider so the home route does not redirect to /setup."""
    provider = MagicMock()
    provider.model_name.return_value = "mock/test-model"
    return provider


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient with a mock LLM client injected."""
    mock_llm = _make_mock_client()
    app = create_app(client=mock_llm, provider=_make_mock_provider())
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: App creation
# ---------------------------------------------------------------------------

class TestAppCreation:
    """Tests for create_app() producing a valid FastAPI instance."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """create_app() returns a FastAPI instance without errors."""
        app = create_app(client=_make_mock_client())
        assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# Tests: Home page elements
# ---------------------------------------------------------------------------

class TestHomePage:
    """Tests for the home page containing expected UI elements."""

    def test_home_contains_all_12_mode_ids(self, client: TestClient) -> None:
        """GET / returns HTML containing the mode selector grid with all 12 mode IDs."""
        response = client.get("/")
        assert response.status_code == 200
        expected_ids = [
            "analytical", "aphoristic", "data_driven", "dialectical",
            "forensic", "measured", "narrative", "oratorical",
            "polemical", "populist", "provocative", "satirical",
        ]
        for mode_id in expected_ids:
            assert mode_id in response.text, f"Mode ID '{mode_id}' not found on home page"

    def test_home_contains_stance_slider(self, client: TestClient) -> None:
        """GET / returns HTML containing the stance slider element."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'name="stance"' in response.text
        assert 'type="range"' in response.text

    def test_home_contains_intensity_slider(self, client: TestClient) -> None:
        """GET / returns HTML containing the intensity slider element."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'name="intensity"' in response.text

    def test_home_contains_topic_input(self, client: TestClient) -> None:
        """GET / returns HTML containing the topic input area."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'name="topic"' in response.text

    def test_home_contains_generate_button(self, client: TestClient) -> None:
        """GET / returns HTML containing the generate button."""
        response = client.get("/")
        assert response.status_code == 200
        assert "generate-btn" in response.text or "Generate" in response.text


# ---------------------------------------------------------------------------
# Tests: Modes page
# ---------------------------------------------------------------------------

class TestModesPage:
    """Tests for the /modes page listing all modes."""

    def test_modes_page_lists_all_12_display_names(self, client: TestClient) -> None:
        """GET /modes returns HTML containing all 12 mode display names."""
        response = client.get("/modes")
        assert response.status_code == 200
        expected_names = [
            "Analytical", "Aphoristic", "Data-Driven", "Dialectical",
            "Forensic", "Measured", "Narrative", "Oratorical",
            "Polemical", "Populist", "Provocative", "Satirical",
        ]
        for name in expected_names:
            assert name in response.text, f"Mode display name '{name}' not found on /modes"


# ---------------------------------------------------------------------------
# Tests: Mode detail page
# ---------------------------------------------------------------------------

class TestModeDetailPage:
    """Tests for individual mode detail pages."""

    def test_mode_detail_analytical(self, client: TestClient) -> None:
        """GET /modes/analytical returns HTML containing 'Analytical' and the mode description."""
        response = client.get("/modes/analytical")
        assert response.status_code == 200
        assert "Analytical" in response.text
        # The description should appear on the detail page
        assert "evidence" in response.text.lower() or "rigorous" in response.text.lower()


# ---------------------------------------------------------------------------
# Tests: About page
# ---------------------------------------------------------------------------

class TestAboutPage:
    """Tests for the /about page."""

    def test_about_contains_editorial_craft_engine(self, client: TestClient) -> None:
        """GET /about returns HTML containing 'editorial craft engine'."""
        response = client.get("/about")
        assert response.status_code == 200
        assert "editorial craft engine" in response.text.lower()

    def test_about_contains_disclaimer_text(self, client: TestClient) -> None:
        """GET /about returns HTML containing the mandatory disclaimer text."""
        response = client.get("/about")
        assert response.status_code == 200
        assert "AI-assisted rhetorical controls" in response.text


# ---------------------------------------------------------------------------
# Tests: HTMX and static resources
# ---------------------------------------------------------------------------

class TestPageInfrastructure:
    """Tests for page infrastructure — HTMX script, CSS, no template errors."""

    def test_all_pages_include_htmx_script(self, client: TestClient) -> None:
        """All pages include the HTMX script tag."""
        pages = ["/", "/modes", "/about"]
        for page in pages:
            response = client.get(page)
            assert response.status_code == 200
            assert "htmx.org" in response.text, f"HTMX script not found on {page}"

    def test_static_css_served(self, client: TestClient) -> None:
        """GET /static/style.css returns 200 status."""
        response = client.get("/static/style.css")
        assert response.status_code == 200

    def test_no_jinja2_undefined_error(self, client: TestClient) -> None:
        """No template renders with a Jinja2 UndefinedError — all pages load without error."""
        pages = ["/", "/modes", "/modes/analytical", "/about"]
        for page in pages:
            response = client.get(page)
            assert response.status_code == 200, f"Route {page} returned {response.status_code}"
            # A Jinja2 UndefinedError would result in 500 or contain the error string
            assert "UndefinedError" not in response.text, f"Jinja2 error on {page}"


# ---------------------------------------------------------------------------
# Tests: Serve command registration
# ---------------------------------------------------------------------------

class TestServeCommand:
    """Tests for the serve command being registered in the Typer app."""

    def test_serve_command_registered(self) -> None:
        """The 'opinionforge serve' command is registered in the Typer app."""
        from opinionforge.cli import app as cli_app

        # Typer registers commands as registered_commands or in registered_groups
        command_names = []
        if hasattr(cli_app, "registered_commands"):
            for cmd in cli_app.registered_commands:
                if hasattr(cmd, "name") and cmd.name:
                    command_names.append(cmd.name)
                elif hasattr(cmd, "callback") and cmd.callback:
                    command_names.append(cmd.callback.__name__)
        # Also check registered groups
        if hasattr(cli_app, "registered_groups"):
            for grp in cli_app.registered_groups:
                if hasattr(grp, "name") and grp.name:
                    command_names.append(grp.name)

        assert "serve" in command_names, (
            f"'serve' command not found in CLI app. Found: {command_names}"
        )
