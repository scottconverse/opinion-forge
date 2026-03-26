"""Unit tests for OpenAICompatibleProvider with mocked HTTP.

Minimum 8 test cases covering generate with custom base_url, stream,
unreachable endpoint, optional api_key, model_name format, and base_url
forwarding.  All tests use mocked openai SDK — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from opinionforge.providers.openai_compatible import OpenAICompatibleProvider
from opinionforge.providers.base import ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    base_url: str = "http://localhost:1234/v1",
    model: str = "local-model",
    api_key: str | None = None,
) -> tuple[OpenAICompatibleProvider, MagicMock]:
    """Create an OpenAICompatibleProvider with mocked SDK client.

    Returns the provider and the mock class so tests can inspect init args.
    """
    with patch.object(openai, "AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        provider = OpenAICompatibleProvider(
            base_url=base_url, model=model, api_key=api_key
        )
    return provider, mock_cls


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


class TestOpenAICompatibleProviderGenerate:
    """Tests for OpenAICompatibleProvider.generate()."""

    @pytest.mark.asyncio
    async def test_successful_generate_with_custom_base_url(self) -> None:
        """generate() returns text from a custom endpoint."""
        provider, _ = _make_provider(base_url="http://my-server:8000/v1")
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("Local response")
        )

        result = await provider.generate("system", "user", 100)
        assert result == "Local response"

    @pytest.mark.asyncio
    async def test_unreachable_endpoint_raises_provider_error(self) -> None:
        """generate() raises ProviderError when the endpoint is unreachable."""
        provider, _ = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(ProviderError, match="Cannot reach"):
            await provider.generate("system", "user", 100)

    @pytest.mark.asyncio
    async def test_api_key_optional_none_works(self) -> None:
        """Provider can be created with api_key=None and still generate."""
        provider, _ = _make_provider(api_key=None)
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        result = await provider.generate("sys", "usr", 50)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_api_key_optional_string_works(self) -> None:
        """Provider can be created with an explicit api_key string."""
        provider, _ = _make_provider(api_key="my-local-key")
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        result = await provider.generate("sys", "usr", 50)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_max_tokens_forwarded(self) -> None:
        """generate() forwards max_tokens to the create call."""
        provider, _ = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        await provider.generate("sys", "usr", 512)
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512


class TestOpenAICompatibleProviderStream:
    """Tests for OpenAICompatibleProvider.stream()."""

    @pytest.mark.asyncio
    async def test_successful_stream(self) -> None:
        """stream() yields token strings from stream chunks."""
        provider, _ = _make_provider()

        chunks = []
        for text in ["token1", "token2"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        final = MagicMock()
        final.choices = [MagicMock()]
        final.choices[0].delta.content = None
        chunks.append(final)

        async def _aiter():
            for c in chunks:
                yield c

        provider._client.chat.completions.create = AsyncMock(return_value=_aiter())

        tokens = []
        async for token in provider.stream("system", "user", 100):
            tokens.append(token)

        assert tokens == ["token1", "token2"]

    @pytest.mark.asyncio
    async def test_stream_unreachable_raises_provider_error(self) -> None:
        """stream() raises ProviderError when the endpoint is unreachable."""
        provider, _ = _make_provider()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(ProviderError, match="Cannot reach"):
            async for _ in provider.stream("system", "user", 100):
                pass


class TestOpenAICompatibleProviderModelName:
    """Tests for OpenAICompatibleProvider.model_name()."""

    def test_model_name_format(self) -> None:
        """model_name() returns 'openai_compatible/{model}'."""
        provider, _ = _make_provider(model="my-local-llm")
        assert provider.model_name() == "openai_compatible/my-local-llm"

    def test_base_url_passed_to_openai_client(self) -> None:
        """The base_url is forwarded to AsyncOpenAI constructor."""
        _, mock_cls = _make_provider(base_url="http://custom:9999/v1")
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["base_url"] == "http://custom:9999/v1"
