"""Tests that onboarding templates render correctly with expected elements.

Verifies that the setup wizard HTML contains all required UI elements,
provider cards, form fields, progress indicators, and navigation buttons.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """TestClient with a mock LLM client and in-memory DB."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Mock output"
    app = create_app(client=mock_client, db_path=":memory:")
    return TestClient(app)


def _get_setup_html(client: TestClient) -> str:
    """Fetch /setup with Ollama detection mocked out."""
    with patch(
        "opinionforge.providers.registry.ProviderRegistry.list_ollama_models",
        new_callable=AsyncMock,
        return_value=["llama3", "mistral"],
    ):
        response = client.get("/setup")
    assert response.status_code == 200
    return response.text


# ---------------------------------------------------------------------------
# Tests: progress indicators
# ---------------------------------------------------------------------------


def test_setup_contains_all_5_step_indicators(client: TestClient) -> None:
    """The setup page contains progress indicators for all 5 steps."""
    html = _get_setup_html(client)
    for step_num in range(1, 6):
        assert f'id="progress-step-{step_num}"' in html, (
            f"Progress indicator for step {step_num} not found"
        )


def test_progress_shows_correct_step_numbers(client: TestClient) -> None:
    """Each progress step shows the correct step number text."""
    html = _get_setup_html(client)
    for step_num in range(1, 6):
        # The step number appears inside a span.step-number
        assert f">{step_num}<" in html, (
            f"Step number {step_num} text not found in progress indicator"
        )


# ---------------------------------------------------------------------------
# Tests: provider cards (Step 2)
# ---------------------------------------------------------------------------


def test_step2_contains_all_4_provider_cards(client: TestClient) -> None:
    """Step 2 contains cards for Ollama, Anthropic, OpenAI, and OpenAI-compatible."""
    html = _get_setup_html(client)
    assert 'data-provider="ollama"' in html
    assert 'data-provider="anthropic"' in html
    assert 'data-provider="openai"' in html
    assert 'data-provider="openai_compatible"' in html


def test_ollama_card_has_model_dropdown(client: TestClient) -> None:
    """Ollama card includes a model dropdown with detected models."""
    html = _get_setup_html(client)
    assert 'id="ollama-model"' in html
    assert "llama3" in html
    assert "mistral" in html


def test_anthropic_card_has_api_key_input(client: TestClient) -> None:
    """Anthropic card has an API key input field."""
    html = _get_setup_html(client)
    assert 'id="anthropic-key"' in html
    assert 'type="password"' in html


def test_anthropic_card_links_to_console(client: TestClient) -> None:
    """Anthropic card includes a link to console.anthropic.com."""
    html = _get_setup_html(client)
    assert "console.anthropic.com" in html


def test_openai_card_has_api_key_input(client: TestClient) -> None:
    """OpenAI card has an API key input field."""
    html = _get_setup_html(client)
    assert 'id="openai-key"' in html


def test_openai_card_links_to_platform(client: TestClient) -> None:
    """OpenAI card includes a link to platform.openai.com."""
    html = _get_setup_html(client)
    assert "platform.openai.com" in html


def test_test_connection_button_exists_for_each_provider(
    client: TestClient,
) -> None:
    """Each provider card has a 'Test Connection' button."""
    html = _get_setup_html(client)
    # Each provider has a testConnection('provider_type') call
    assert "testConnection('ollama')" in html
    assert "testConnection('anthropic')" in html
    assert "testConnection('openai')" in html
    assert "testConnection('openai_compatible')" in html


def test_openai_compatible_has_base_url_and_key(client: TestClient) -> None:
    """OpenAI-compatible card has base_url and optional api_key fields."""
    html = _get_setup_html(client)
    assert 'id="compat-url"' in html
    assert 'id="compat-key"' in html


# ---------------------------------------------------------------------------
# Tests: search step (Step 3)
# ---------------------------------------------------------------------------


def test_skip_button_on_search_step(client: TestClient) -> None:
    """Step 3 (Search) has a 'Skip for now' button."""
    html = _get_setup_html(client)
    assert "Skip for now" in html
    assert "skipSearch()" in html


def test_search_step_has_three_providers(client: TestClient) -> None:
    """Step 3 shows Tavily, Brave, and SerpAPI options."""
    html = _get_setup_html(client)
    assert 'data-search="tavily"' in html
    assert 'data-search="brave"' in html
    assert 'data-search="serpapi"' in html


# ---------------------------------------------------------------------------
# Tests: navigation
# ---------------------------------------------------------------------------


def test_back_button_hidden_on_step_1(client: TestClient) -> None:
    """Step 1 (Welcome) does not have a Back button since there is
    no previous step to go to — it only has Get Started."""
    html = _get_setup_html(client)
    # Step 1 content should have 'Get Started' but no 'Back' within it
    # We look at the step-1 div specifically
    step1_start = html.find('id="step-1"')
    step1_end = html.find('id="step-2"')
    step1_html = html[step1_start:step1_end]
    assert "Get Started" in step1_html
    assert "Back" not in step1_html


def test_welcome_step_has_get_started_button(client: TestClient) -> None:
    """Step 1 has a 'Get Started' button to advance to step 2."""
    html = _get_setup_html(client)
    assert "Get Started" in html


# ---------------------------------------------------------------------------
# Tests: tour step (Step 4)
# ---------------------------------------------------------------------------


def test_tour_step_has_skip_button(client: TestClient) -> None:
    """Step 4 (Quick Tour) has a 'Skip Tour' button."""
    html = _get_setup_html(client)
    assert "Skip Tour" in html


def test_tour_highlights_key_elements(client: TestClient) -> None:
    """The tour carousel highlights mode selector, stance slider,
    intensity slider, and generate button."""
    html = _get_setup_html(client)
    assert "mode-selector" in html
    assert "stance-slider" in html
    assert "intensity-slider" in html
    assert "generate-button" in html


# ---------------------------------------------------------------------------
# Tests: test generation step (Step 5)
# ---------------------------------------------------------------------------


def test_step5_shows_sample_topic(client: TestClient) -> None:
    """Step 5 displays the sample topic 'The Future of Remote Work'."""
    html = _get_setup_html(client)
    assert "The Future of Remote Work" in html


def test_step5_has_test_generate_button(client: TestClient) -> None:
    """Step 5 has a 'Run Test Generation' button."""
    html = _get_setup_html(client)
    assert "test-generate-btn" in html
    assert "Run Test Generation" in html
