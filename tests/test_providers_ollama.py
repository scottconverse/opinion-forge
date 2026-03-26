"""Unit tests for OllamaProvider with mocked httpx.

Minimum 10 test cases covering generate, stream, connection refused,
model not found, model_name format, POST body, custom base_url,
empty response, and timeout handling.
All tests use mocked httpx — no real Ollama calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from opinionforge.providers.ollama import OllamaProvider
from opinionforge.providers.base import ProviderError


# ---------------------------------------------------------------------------
# Tests — generate
# ---------------------------------------------------------------------------


class TestOllamaProviderGenerate:
    """Tests for OllamaProvider.generate()."""

    @pytest.mark.asyncio
    async def test_successful_generate(self) -> None:
        """generate() returns the response text from Ollama."""
        provider = OllamaProvider(model="llama3")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello from Ollama"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate("system", "user", 100)

        assert result == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_connection_refused_raises_provider_error(self) -> None:
        """generate() raises ProviderError when Ollama is not running."""
        provider = OllamaProvider(model="llama3")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ProviderError, match="not running"):
                await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_model_not_found_raises_provider_error(self) -> None:
        """generate() raises ProviderError when model is not pulled."""
        provider = OllamaProvider(model="nonexistent")

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ProviderError, match="not found"):
                await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_correct_post_body(self) -> None:
        """generate() sends the correct JSON body to /api/generate."""
        provider = OllamaProvider(model="llama3")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            await provider.generate("my system", "my prompt", 256)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["model"] == "llama3"
        assert body["system"] == "my system"
        assert body["prompt"] == "my prompt"
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_custom_base_url(self) -> None:
        """generate() uses the custom base_url in the POST request."""
        provider = OllamaProvider(model="llama3", base_url="http://remote:11434")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            await provider.generate("sys", "usr", 50)

        call_args = mock_client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "remote:11434" in url

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        """generate() returns empty string when Ollama returns empty response."""
        provider = OllamaProvider(model="llama3")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate("system", "user", 100)

        assert result == ""

    @pytest.mark.asyncio
    async def test_timeout_raises_provider_error(self) -> None:
        """generate() raises ProviderError on request timeout."""
        provider = OllamaProvider(model="llama3")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timed out")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ProviderError, match="timed out"):
                await provider.generate("system", "user", 100)


# ---------------------------------------------------------------------------
# Tests — stream
# ---------------------------------------------------------------------------


class TestOllamaProviderStream:
    """Tests for OllamaProvider.stream()."""

    @pytest.mark.asyncio
    async def test_successful_stream_multiple_chunks(self) -> None:
        """stream() yields multiple token strings from streamed JSON lines."""
        provider = OllamaProvider(model="llama3")

        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        mock_stream_response = AsyncMock()
        mock_stream_response.status_code = 200
        mock_stream_response.raise_for_status = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_stream_response.aiter_lines = _aiter_lines
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in provider.stream("system", "user", 100):
                tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_connection_refused(self) -> None:
        """stream() raises ProviderError when Ollama is not running."""
        provider = OllamaProvider(model="llama3")

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opinionforge.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ProviderError, match="not running"):
                async for _ in provider.stream("system", "user", 100):
                    pass


# ---------------------------------------------------------------------------
# Tests — model_name
# ---------------------------------------------------------------------------


class TestOllamaProviderModelName:
    """Tests for OllamaProvider.model_name()."""

    def test_model_name_format(self) -> None:
        """model_name() returns 'ollama/{model}'."""
        provider = OllamaProvider(model="llama3")
        assert provider.model_name() == "ollama/llama3"
