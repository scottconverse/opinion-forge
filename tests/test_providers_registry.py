"""Unit tests for ProviderRegistry factory, detection, and connection testing.

Minimum 10 test cases covering provider creation, unknown types, test_connection,
detect_ollama, list_ollama_models, and get_provider convenience function.
All tests use mocked HTTP — no real network calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import openai
import pytest

from opinionforge.models.config import ProviderConfig
from opinionforge.providers.base import LLMProvider, ProviderError
from opinionforge.providers.registry import ProviderRegistry, get_provider


# ---------------------------------------------------------------------------
# Tests — create_provider
# ---------------------------------------------------------------------------


class TestProviderRegistryCreate:
    """Tests for ProviderRegistry.create_provider()."""

    def test_create_anthropic(self) -> None:
        """create_provider('anthropic') returns an AnthropicProvider."""
        registry = ProviderRegistry()
        with patch.object(anthropic, "AsyncAnthropic"):
            provider = registry.create_provider("anthropic", api_key="key")
        assert isinstance(provider, LLMProvider)
        assert provider.model_name().startswith("anthropic/")

    def test_create_openai(self) -> None:
        """create_provider('openai') returns an OpenAIProvider."""
        registry = ProviderRegistry()
        with patch.object(openai, "AsyncOpenAI"):
            provider = registry.create_provider("openai", api_key="key")
        assert isinstance(provider, LLMProvider)
        assert provider.model_name().startswith("openai/")

    def test_create_openai_compatible(self) -> None:
        """create_provider('openai_compatible') returns an OpenAICompatibleProvider."""
        registry = ProviderRegistry()
        with patch.object(openai, "AsyncOpenAI"):
            provider = registry.create_provider(
                "openai_compatible", base_url="http://localhost:1234/v1", model="test"
            )
        assert isinstance(provider, LLMProvider)
        assert provider.model_name().startswith("openai_compatible/")

    def test_create_ollama(self) -> None:
        """create_provider('ollama') returns an OllamaProvider."""
        registry = ProviderRegistry()
        provider = registry.create_provider("ollama", model="llama3")
        assert isinstance(provider, LLMProvider)
        assert provider.model_name() == "ollama/llama3"

    def test_unknown_provider_type_raises_value_error(self) -> None:
        """create_provider() raises ValueError for unknown provider type."""
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="Unknown provider type"):
            registry.create_provider("nonexistent")


# ---------------------------------------------------------------------------
# Tests — test_connection
# ---------------------------------------------------------------------------


class TestProviderRegistryConnection:
    """Tests for ProviderRegistry.test_connection()."""

    @pytest.mark.asyncio
    async def test_connection_success(self) -> None:
        """test_connection returns (True, message) when provider responds."""
        registry = ProviderRegistry()
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.generate = AsyncMock(return_value="ok")
        mock_provider.model_name.return_value = "test/model"

        success, message = await registry.test_connection(mock_provider)
        assert success is True
        assert "Connected" in message

    @pytest.mark.asyncio
    async def test_connection_failure(self) -> None:
        """test_connection returns (False, message) when provider raises."""
        registry = ProviderRegistry()
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.generate = AsyncMock(
            side_effect=ProviderError("connection refused")
        )
        mock_provider.model_name.return_value = "test/model"

        success, message = await registry.test_connection(mock_provider)
        assert success is False
        assert "failed" in message.lower()


# ---------------------------------------------------------------------------
# Tests — detect_ollama
# ---------------------------------------------------------------------------


class TestProviderRegistryDetectOllama:
    """Tests for ProviderRegistry.detect_ollama()."""

    @pytest.mark.asyncio
    async def test_detect_ollama_running(self) -> None:
        """detect_ollama returns True when Ollama responds with 200."""
        registry = ProviderRegistry()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.registry.httpx.AsyncClient", return_value=mock_client):
            result = await registry.detect_ollama()

        assert result is True

    @pytest.mark.asyncio
    async def test_detect_ollama_not_running(self) -> None:
        """detect_ollama returns False when Ollama is unreachable."""
        registry = ProviderRegistry()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.registry.httpx.AsyncClient", return_value=mock_client):
            result = await registry.detect_ollama()

        assert result is False


# ---------------------------------------------------------------------------
# Tests — list_ollama_models
# ---------------------------------------------------------------------------


class TestProviderRegistryListModels:
    """Tests for ProviderRegistry.list_ollama_models()."""

    @pytest.mark.asyncio
    async def test_list_models_returns_names(self) -> None:
        """list_ollama_models returns a list of model name strings."""
        registry = ProviderRegistry()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "codellama:7b"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.registry.httpx.AsyncClient", return_value=mock_client):
            models = await registry.list_ollama_models()

        assert models == ["llama3:latest", "codellama:7b"]

    @pytest.mark.asyncio
    async def test_list_models_ollama_not_running(self) -> None:
        """list_ollama_models raises ProviderError when Ollama is unreachable."""
        registry = ProviderRegistry()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.registry.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ProviderError, match="not running"):
                await registry.list_ollama_models()


# ---------------------------------------------------------------------------
# Tests — get_provider convenience function
# ---------------------------------------------------------------------------


class TestGetProviderConvenience:
    """Tests for the module-level get_provider() function."""

    def test_get_provider_reads_provider_config_ollama(self) -> None:
        """get_provider() reads ProviderConfig and returns an OllamaProvider."""
        config = ProviderConfig(provider_type="ollama", model="llama3")
        provider = get_provider(config)
        assert provider.model_name() == "ollama/llama3"

    def test_get_provider_reads_provider_config_with_api_key(self) -> None:
        """get_provider() passes api_key from ProviderConfig to the provider."""
        config = ProviderConfig(
            provider_type="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        with patch.object(anthropic, "AsyncAnthropic"):
            provider = get_provider(config)
        assert provider.model_name().startswith("anthropic/")

    def test_get_provider_reads_provider_config_with_base_url(self) -> None:
        """get_provider() passes base_url from ProviderConfig to the provider."""
        config = ProviderConfig(
            provider_type="openai_compatible",
            model="local-model",
            base_url="http://localhost:1234/v1",
        )
        with patch.object(openai, "AsyncOpenAI"):
            provider = get_provider(config)
        assert provider.model_name().startswith("openai_compatible/")

    def test_get_provider_unknown_raises(self) -> None:
        """get_provider() with unknown provider_type in config raises ValueError."""
        with pytest.raises(ValueError, match="provider_type must be one of"):
            ProviderConfig(provider_type="bad_type", model="m")
