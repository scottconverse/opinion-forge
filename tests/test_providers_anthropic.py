"""Unit tests for AnthropicProvider with mocked HTTP.

Minimum 8 test cases covering generate, stream, auth errors, network errors,
model_name format, empty response, max_tokens, and prompt forwarding.
All tests use mocked anthropic SDK — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from opinionforge.providers.anthropic import AnthropicProvider
from opinionforge.providers.base import ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(model: str = "claude-test") -> AnthropicProvider:
    """Create an AnthropicProvider with a mocked AsyncAnthropic client."""
    with patch.object(anthropic, "AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        provider = AnthropicProvider(api_key="test-key", model=model)
    # The provider._client is already the mock_client
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicProviderGenerate:
    """Tests for AnthropicProvider.generate()."""

    @pytest.mark.asyncio
    async def test_successful_generate(self) -> None:
        """generate() returns the text content from the API response."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("system", "user", 100)
        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_auth_error_raises_provider_error(self) -> None:
        """generate() raises ProviderError on authentication failure."""
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "invalid key"}}
        provider._client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="invalid key", response=mock_resp, body={}
            )
        )

        with pytest.raises(ProviderError, match="authentication failed"):
            await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_network_error_raises_provider_error(self) -> None:
        """generate() raises ProviderError on network connection failure."""
        provider = _make_provider()
        provider._client.messages.create = AsyncMock(
            side_effect=anthropic.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(ProviderError, match="connect"):
            await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_empty_response_handling(self) -> None:
        """generate() returns empty string when content text is empty."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="")]
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("system", "user", 100)
        assert result == ""

    @pytest.mark.asyncio
    async def test_max_tokens_passed_correctly(self) -> None:
        """generate() forwards max_tokens to the API call."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        await provider.generate("sys", "usr", 999)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 999

    @pytest.mark.asyncio
    async def test_system_and_user_prompt_sent_correctly(self) -> None:
        """generate() sends system_prompt and user_prompt in the correct fields."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        await provider.generate("my system prompt", "my user prompt", 50)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "my system prompt"
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "my user prompt"}
        ]


class TestAnthropicProviderStream:
    """Tests for AnthropicProvider.stream()."""

    @pytest.mark.asyncio
    async def test_successful_stream(self) -> None:
        """stream() yields token strings from the text_stream."""
        provider = _make_provider()

        # Build an async context manager mock that yields tokens
        mock_stream_ctx = AsyncMock()

        async def _text_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.text_stream = _text_stream()

        provider._client.messages.stream = MagicMock(return_value=mock_stream_ctx)

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
        mock_resp.json.return_value = {"error": {"message": "invalid key"}}

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="invalid key", response=mock_resp, body={}
            )
        )
        provider._client.messages.stream = MagicMock(return_value=mock_stream_ctx)

        with pytest.raises(ProviderError, match="authentication failed"):
            async for _ in provider.stream("system", "user", 100):
                pass


class TestAnthropicProviderModelName:
    """Tests for AnthropicProvider.model_name()."""

    def test_model_name_format(self) -> None:
        """model_name() returns 'anthropic/{model}'."""
        provider = _make_provider()
        assert provider.model_name() == "anthropic/claude-test"

    def test_model_name_default(self) -> None:
        """model_name() with default model includes claude-sonnet-4-20250514."""
        with patch.object(anthropic, "AsyncAnthropic"):
            provider = AnthropicProvider(api_key="key")
        assert provider.model_name() == "anthropic/claude-sonnet-4-20250514"
