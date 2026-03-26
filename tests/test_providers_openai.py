"""Unit tests for OpenAIProvider with mocked HTTP.

Minimum 8 test cases covering generate, stream, auth errors, network errors,
model_name format, empty response, max_tokens, and message formatting.
All tests use mocked openai SDK — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from opinionforge.providers.openai_provider import OpenAIProvider
from opinionforge.providers.base import ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(model: str = "gpt-test") -> OpenAIProvider:
    """Create an OpenAIProvider with a mocked AsyncOpenAI client."""
    with patch.object(openai, "AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        provider = OpenAIProvider(api_key="test-key", model=model)
    return provider


def _mock_completion(text: str) -> MagicMock:
    """Build a mock chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderGenerate:
    """Tests for OpenAIProvider.generate()."""

    @pytest.mark.asyncio
    async def test_successful_generate(self) -> None:
        """generate() returns the message content from the API response."""
        provider = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("Hello from GPT")
        )

        result = await provider.generate("system", "user", 100)
        assert result == "Hello from GPT"

    @pytest.mark.asyncio
    async def test_auth_error_raises_provider_error(self) -> None:
        """generate() raises ProviderError on authentication failure."""
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "invalid key"}}
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="invalid key", response=mock_resp, body={}
            )
        )

        with pytest.raises(ProviderError, match="authentication failed"):
            await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_network_error_raises_provider_error(self) -> None:
        """generate() raises ProviderError on network failure."""
        provider = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(ProviderError, match="connect"):
            await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_empty_response_handling(self) -> None:
        """generate() returns empty string when message content is empty."""
        provider = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("")
        )

        result = await provider.generate("system", "user", 100)
        assert result == ""

    @pytest.mark.asyncio
    async def test_max_tokens_passed_correctly(self) -> None:
        """generate() forwards max_tokens to the API call."""
        provider = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        await provider.generate("sys", "usr", 777)

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 777

    @pytest.mark.asyncio
    async def test_system_and_user_messages_sent_correctly(self) -> None:
        """generate() sends system and user messages in the correct format."""
        provider = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        await provider.generate("my system", "my user", 50)

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "my system"},
            {"role": "user", "content": "my user"},
        ]


class TestOpenAIProviderStream:
    """Tests for OpenAIProvider.stream()."""

    @pytest.mark.asyncio
    async def test_successful_stream(self) -> None:
        """stream() yields token strings from stream chunks."""
        provider = _make_provider()

        chunks = []
        for text in ["Hello", " ", "world"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        # Add a final chunk with no content
        final_chunk = MagicMock()
        final_chunk.choices = [MagicMock()]
        final_chunk.choices[0].delta.content = None
        chunks.append(final_chunk)

        async def _aiter():
            for c in chunks:
                yield c

        provider._client.chat.completions.create = AsyncMock(return_value=_aiter())

        tokens = []
        async for token in provider.stream("system", "user", 100):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_stream_auth_error(self) -> None:
        """stream() raises ProviderError on authentication failure."""
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "bad key"}}
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="bad key", response=mock_resp, body={}
            )
        )

        with pytest.raises(ProviderError, match="authentication failed"):
            async for _ in provider.stream("system", "user", 100):
                pass


class TestOpenAIProviderModelName:
    """Tests for OpenAIProvider.model_name()."""

    def test_model_name_format(self) -> None:
        """model_name() returns 'openai/{model}'."""
        provider = _make_provider("gpt-test")
        assert provider.model_name() == "openai/gpt-test"

    def test_model_name_default(self) -> None:
        """model_name() with default model includes gpt-4o."""
        with patch.object(openai, "AsyncOpenAI"):
            provider = OpenAIProvider(api_key="key")
        assert provider.model_name() == "openai/gpt-4o"
