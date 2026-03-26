"""Cross-provider integration tests ensuring all providers produce valid output.

Tests each of the 4 provider types (Anthropic, OpenAI, Ollama,
OpenAI-compatible) using mocked HTTP — no real API calls.
"""

from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opinionforge.providers.anthropic import AnthropicProvider
from opinionforge.providers.base import LLMProvider, ProviderError
from opinionforge.providers.ollama import OllamaProvider
from opinionforge.providers.openai_compatible import OpenAICompatibleProvider
from opinionforge.providers.openai_provider import OpenAIProvider
from opinionforge.providers.registry import ProviderRegistry, get_provider

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = (
    "## AI and Governance\n\n"
    "Artificial intelligence is reshaping democratic institutions in ways "
    "that demand urgent scrutiny."
)


# ---------------------------------------------------------------------------
# Tests: Each provider produces a generate response with model_name
# ---------------------------------------------------------------------------


class TestAnthropicProviderGenerate:
    """Anthropic provider generates text and records model_name."""

    @pytest.mark.asyncio
    async def test_anthropic_generate_returns_text(self) -> None:
        """AnthropicProvider.generate returns the mocked response text."""
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_MOCK_RESPONSE)]

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            MockAnthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="test-key", model="claude-test")
            provider._client = mock_client
            result = await provider.generate("system", "user prompt", 1000)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_anthropic_model_name(self) -> None:
        """AnthropicProvider.model_name returns 'anthropic/{model}'."""
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="key", model="claude-test")
        assert provider.model_name() == "anthropic/claude-test"


class TestOpenAIProviderGenerate:
    """OpenAI provider generates text and records model_name."""

    @pytest.mark.asyncio
    async def test_openai_generate_returns_text(self) -> None:
        """OpenAIProvider.generate returns the mocked response text."""
        mock_choice = MagicMock()
        mock_choice.message.content = _MOCK_RESPONSE
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            MockOpenAI.return_value = mock_client

            provider = OpenAIProvider(api_key="test-key", model="gpt-4o-test")
            provider._client = mock_client
            result = await provider.generate("system", "user prompt", 1000)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_openai_model_name(self) -> None:
        """OpenAIProvider.model_name returns 'openai/{model}'."""
        with patch("openai.AsyncOpenAI"):
            provider = OpenAIProvider(api_key="key", model="gpt-4o-test")
        assert provider.model_name() == "openai/gpt-4o-test"


class TestOllamaProviderGenerate:
    """Ollama provider generates text and records model_name."""

    @pytest.mark.asyncio
    async def test_ollama_generate_returns_text(self) -> None:
        """OllamaProvider.generate returns the mocked response text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": _MOCK_RESPONSE}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            provider = OllamaProvider(model="llama3")
            result = await provider.generate("system", "user prompt", 1000)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_ollama_model_name(self) -> None:
        """OllamaProvider.model_name returns 'ollama/{model}'."""
        provider = OllamaProvider(model="llama3")
        assert provider.model_name() == "ollama/llama3"


class TestOpenAICompatibleProviderGenerate:
    """OpenAI-compatible provider generates text and records model_name."""

    @pytest.mark.asyncio
    async def test_openai_compatible_generate_returns_text(self) -> None:
        """OpenAICompatibleProvider.generate returns text."""
        mock_choice = MagicMock()
        mock_choice.message.content = _MOCK_RESPONSE
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            MockOpenAI.return_value = mock_client

            provider = OpenAICompatibleProvider(
                base_url="http://localhost:1234/v1",
                model="local-model",
            )
            provider._client = mock_client
            result = await provider.generate("system", "user prompt", 1000)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_openai_compatible_model_name(self) -> None:
        """OpenAICompatibleProvider.model_name returns 'openai_compatible/{model}'."""
        with patch("openai.AsyncOpenAI"):
            provider = OpenAICompatibleProvider(
                base_url="http://localhost:1234/v1",
                model="local-model",
            )
        assert provider.model_name() == "openai_compatible/local-model"


# ---------------------------------------------------------------------------
# Tests: Streaming
# ---------------------------------------------------------------------------


class TestOllamaStreaming:
    """Ollama provider handles streaming correctly."""

    @pytest.mark.asyncio
    async def test_ollama_stream_produces_tokens(self) -> None:
        """OllamaProvider.stream yields token strings."""
        lines = [
            json.dumps({"response": "Hello ", "done": False}),
            json.dumps({"response": "world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_instance.stream = MagicMock(return_value=mock_stream_ctx)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            provider = OllamaProvider(model="llama3")
            tokens = []
            async for token in provider.stream("system", "prompt", 100):
                tokens.append(token)

        assert len(tokens) >= 1
        assert "Hello " in tokens


# ---------------------------------------------------------------------------
# Tests: Provider fallback when primary fails
# ---------------------------------------------------------------------------


class TestProviderFallback:
    """Provider registry handles provider creation errors gracefully."""

    def test_unknown_provider_raises_value_error(self) -> None:
        """ProviderRegistry.create_provider raises ValueError for unknown type."""
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="Unknown provider type"):
            registry.create_provider("nonexistent_provider")

    @pytest.mark.asyncio
    async def test_test_connection_reports_failure(self) -> None:
        """ProviderRegistry.test_connection returns (False, message) on error."""
        mock_provider = MagicMock()
        mock_provider.model_name.return_value = "test/model"
        mock_provider.generate = AsyncMock(
            side_effect=ProviderError("Connection refused", provider="test")
        )

        registry = ProviderRegistry()
        success, message = await registry.test_connection(mock_provider)

        assert success is False
        assert "Connection failed" in message
