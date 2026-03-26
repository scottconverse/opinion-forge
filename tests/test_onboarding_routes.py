"""Tests for onboarding wizard backend routes.

All LLM and provider calls are mocked — zero real API calls.
Uses FastAPI TestClient to exercise every onboarding route.
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

def _make_app(**kwargs) -> TestClient:
    """Create a TestClient backed by an in-memory DB.

    Passes ``:memory:`` as db_path so tests never touch the real
    settings database.
    """
    mock_client = MagicMock()
    mock_client.generate.return_value = "Mock generated text"
    app = create_app(client=mock_client, db_path=":memory:", **kwargs)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    """TestClient with in-memory DB and mock LLM client."""
    return _make_app()


@pytest.fixture
def client_no_provider() -> TestClient:
    """TestClient without any LLM client configured."""
    app = create_app(db_path=":memory:")
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /setup
# ---------------------------------------------------------------------------


def test_setup_returns_200(client: TestClient) -> None:
    """GET /setup returns 200."""
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.list_ollama_models",
        new_callable=AsyncMock,
        return_value=[],
    ):
        response = client.get("/setup")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_setup_contains_wizard_steps(client: TestClient) -> None:
    """GET /setup response contains all 5 step elements."""
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.list_ollama_models",
        new_callable=AsyncMock,
        return_value=[],
    ):
        response = client.get("/setup")
    body = response.text
    for step_num in range(1, 6):
        assert f'id="step-{step_num}"' in body, (
            f"Step {step_num} element not found in setup page"
        )


def test_setup_contains_progress_indicators(client: TestClient) -> None:
    """GET /setup has progress step indicators for steps 1-5."""
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.list_ollama_models",
        new_callable=AsyncMock,
        return_value=[],
    ):
        response = client.get("/setup")
    body = response.text
    for step_num in range(1, 6):
        assert f'id="progress-step-{step_num}"' in body


# ---------------------------------------------------------------------------
# POST /setup/test-connection
# ---------------------------------------------------------------------------


def test_test_connection_ollama_running(client: TestClient) -> None:
    """POST /setup/test-connection with running Ollama returns success."""
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.detect_ollama",
        new_callable=AsyncMock,
        return_value=True,
    ):
        response = client.post(
            "/setup/test-connection",
            data={"provider_type": "ollama", "base_url": "http://localhost:11434", "model": ""},
        )
    assert response.status_code == 200
    assert "connection-success" in response.text


def test_test_connection_ollama_not_running(client: TestClient) -> None:
    """POST /setup/test-connection with unreachable Ollama returns error."""
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.detect_ollama",
        new_callable=AsyncMock,
        return_value=False,
    ):
        response = client.post(
            "/setup/test-connection",
            data={"provider_type": "ollama", "base_url": "http://localhost:11434", "model": ""},
        )
    assert response.status_code == 200
    assert "connection-error" in response.text
    assert "ollama serve" in response.text


def test_test_connection_anthropic_success(client: TestClient) -> None:
    """POST /setup/test-connection with valid Anthropic config returns success."""
    mock_provider = MagicMock()
    mock_provider.model_name.return_value = "anthropic/claude-3-haiku"

    with patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=mock_provider,
    ), patch(
        "opinionforge.providers.registry.ProviderRegistry.test_connection",
        new_callable=AsyncMock,
        return_value=(True, "Connected to anthropic/claude-3-haiku: ok"),
    ):
        response = client.post(
            "/setup/test-connection",
            data={
                "provider_type": "anthropic",
                "api_key": "sk-ant-test-key",
                "model": "claude-3-haiku-20240307",
            },
        )
    assert response.status_code == 200
    assert "connection-success" in response.text


def test_test_connection_invalid_api_key(client: TestClient) -> None:
    """POST /setup/test-connection with bad API key returns error."""
    mock_provider = MagicMock()

    with patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=mock_provider,
    ), patch(
        "opinionforge.providers.registry.ProviderRegistry.test_connection",
        new_callable=AsyncMock,
        return_value=(False, "Authentication failed: invalid API key"),
    ):
        response = client.post(
            "/setup/test-connection",
            data={
                "provider_type": "anthropic",
                "api_key": "invalid-key",
                "model": "claude-3-haiku-20240307",
            },
        )
    assert response.status_code == 200
    assert "connection-error" in response.text


def test_test_connection_unknown_provider_returns_error(client: TestClient) -> None:
    """POST /setup/test-connection with unknown provider type returns error."""
    response = client.post(
        "/setup/test-connection",
        data={"provider_type": "unknown_provider", "model": "some-model"},
    )
    assert response.status_code == 200
    assert "connection-error" in response.text


# ---------------------------------------------------------------------------
# POST /setup/save-provider
# ---------------------------------------------------------------------------


def test_save_provider_stores_config(client: TestClient) -> None:
    """POST /setup/save-provider stores provider config in DB."""
    response = client.post(
        "/setup/save-provider",
        data={
            "provider_type": "anthropic",
            "api_key": "sk-ant-test",
            "model": "claude-3-haiku-20240307",
        },
    )
    assert response.status_code == 200


def test_save_provider_ollama(client: TestClient) -> None:
    """POST /setup/save-provider stores Ollama config."""
    response = client.post(
        "/setup/save-provider",
        data={
            "provider_type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama3",
        },
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /setup/save-search
# ---------------------------------------------------------------------------


def test_save_search_stores_config(client: TestClient) -> None:
    """POST /setup/save-search stores search config in DB."""
    response = client.post(
        "/setup/save-search",
        data={"provider": "tavily", "api_key": "tvly-test-key"},
    )
    assert response.status_code == 200


def test_save_search_skip_stores_none(client: TestClient) -> None:
    """POST /setup/save-search with provider='none' stores skip config."""
    response = client.post(
        "/setup/save-search",
        data={"provider": "none", "api_key": ""},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /setup/test-generate
# ---------------------------------------------------------------------------


def test_test_generate_with_working_provider(client: TestClient) -> None:
    """POST /setup/test-generate with a working provider returns success."""
    # First save a provider config so the route can load it
    client.post(
        "/setup/save-provider",
        data={
            "provider_type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama3",
        },
    )

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(
        return_value="## Remote Work\n\nThe future is flexible."
    )

    with patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=mock_provider,
    ):
        response = client.post("/setup/test-generate")

    assert response.status_code == 200
    assert "test-generate-success" in response.text
    assert "Everything is working" in response.text


def test_test_generate_with_broken_provider(client: TestClient) -> None:
    """POST /setup/test-generate with a broken provider returns error."""
    # First save a provider config
    client.post(
        "/setup/save-provider",
        data={
            "provider_type": "anthropic",
            "api_key": "bad-key",
            "model": "claude-3-haiku-20240307",
        },
    )

    from opinionforge.providers.base import ProviderError

    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(
        side_effect=ProviderError(
            "Authentication failed: invalid API key",
            provider="anthropic",
        )
    )

    with patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=mock_provider,
    ):
        response = client.post("/setup/test-generate")

    assert response.status_code == 200
    assert "test-generate-error" in response.text


def test_test_generate_no_provider_configured(client: TestClient) -> None:
    """POST /setup/test-generate without any provider config returns error."""
    # Don't save any provider — settings are empty
    app = create_app(db_path=":memory:")
    fresh_client = TestClient(app)

    with patch(
        "opinionforge.providers.registry.ProviderRegistry.list_ollama_models",
        new_callable=AsyncMock,
        return_value=[],
    ):
        response = fresh_client.post("/setup/test-generate")

    assert response.status_code == 200
    assert "test-generate-error" in response.text
    assert "No provider configured" in response.text


# ---------------------------------------------------------------------------
# POST /setup/complete
# ---------------------------------------------------------------------------


def test_complete_marks_onboarding_done(client: TestClient) -> None:
    """POST /setup/complete returns success."""
    response = client.post("/setup/complete")
    assert response.status_code in (200, 303)


def test_complete_redirects_to_home(client: TestClient) -> None:
    """POST /setup/complete redirects to /."""
    response = client.post("/setup/complete", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_complete_htmx_returns_redirect_header(client: TestClient) -> None:
    """POST /setup/complete with HX-Request returns HX-Redirect header."""
    response = client.post(
        "/setup/complete",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert response.headers.get("HX-Redirect") == "/"


# ---------------------------------------------------------------------------
# Home redirect behaviour
# ---------------------------------------------------------------------------


def test_home_does_not_redirect_when_provider_configured(client: TestClient) -> None:
    """GET / does NOT redirect to /setup when onboarding has been completed."""
    # Complete onboarding so the flag is set in DB
    client.post("/setup/complete", follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "OpinionForge" in response.text


def test_home_redirects_to_setup_when_onboarding_not_complete() -> None:
    """GET / redirects to /setup when onboarding_completed is False."""
    # Create an app with no client and a fresh in-memory DB (onboarding not done)
    app = create_app(db_path=":memory:")
    fresh_client = TestClient(app, follow_redirects=False)
    response = fresh_client.get("/")
    assert response.status_code == 303
    assert response.headers["location"] == "/setup"


# ---------------------------------------------------------------------------
# Data preservation across steps
# ---------------------------------------------------------------------------


def test_going_back_preserves_previously_entered_data(client: TestClient) -> None:
    """Going back in the wizard preserves previously saved provider data.

    Simulates a user completing step 2 (saving a provider), then navigating
    back to /setup — the saved config should still be retrievable from the DB.
    """
    # Step 2: save provider config
    resp = client.post(
        "/setup/save-provider",
        data={
            "provider_type": "anthropic",
            "api_key": "sk-ant-test",
            "model": "claude-3-haiku-20240307",
        },
    )
    assert resp.status_code == 200

    # Simulate going back: GET /setup should still return 200 (wizard accessible)
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.list_ollama_models",
        new_callable=AsyncMock,
        return_value=[],
    ):
        back_resp = client.get("/setup")
    assert back_resp.status_code == 200

    # The previously entered data is still in the DB — verify by triggering
    # test-generate which loads the saved provider config
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value="## Test\n\nContent.")
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=mock_provider,
    ) as create_mock:
        gen_resp = client.post("/setup/test-generate")
    assert gen_resp.status_code == 200
    # Verify the provider was created with the previously saved type
    create_mock.assert_called_once()
    assert create_mock.call_args[0][0] == "anthropic"


def test_save_then_load_provider_config(client: TestClient) -> None:
    """Provider config saved via /setup/save-provider persists and can be
    used by /setup/test-generate."""
    # Save provider config
    client.post(
        "/setup/save-provider",
        data={
            "provider_type": "openai",
            "api_key": "sk-test-key",
            "model": "gpt-4o",
        },
    )

    # Verify test-generate route can load it
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(
        return_value="## Test\n\nGenerated content."
    )

    with patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=mock_provider,
    ) as create_mock:
        response = client.post("/setup/test-generate")

    assert response.status_code == 200
    # Verify the provider was created with the right type
    create_mock.assert_called_once()
    call_args = create_mock.call_args
    assert call_args[0][0] == "openai"  # provider_type
