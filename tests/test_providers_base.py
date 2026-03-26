"""Unit tests for the LLMProvider protocol and ProviderError exception.

Minimum 5 test cases covering protocol satisfaction, protocol violation,
ProviderError attributes, and ProviderError inheritance.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from opinionforge.providers.base import LLMProvider, ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ValidProvider:
    """A minimal class that satisfies the LLMProvider protocol."""

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        return "hello"

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        yield "hello"

    def model_name(self) -> str:
        return "test/model"


class _MissingGenerate:
    """A class that is missing generate() — violates LLMProvider."""

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        yield "hello"

    def model_name(self) -> str:
        return "test/model"


class _MissingStream:
    """A class that is missing stream() — violates LLMProvider."""

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        return "hello"

    def model_name(self) -> str:
        return "test/model"


class _MissingModelName:
    """A class that is missing model_name() — violates LLMProvider."""

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        return "hello"

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        yield "hello"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMProviderProtocol:
    """Tests for the LLMProvider protocol interface."""

    def test_valid_provider_satisfies_protocol(self) -> None:
        """A class implementing all three methods passes isinstance check."""
        provider = _ValidProvider()
        assert isinstance(provider, LLMProvider)

    def test_missing_generate_violates_protocol(self) -> None:
        """A class missing generate() fails the isinstance check."""
        obj = _MissingGenerate()
        assert not isinstance(obj, LLMProvider)

    def test_missing_stream_violates_protocol(self) -> None:
        """A class missing stream() fails the isinstance check."""
        obj = _MissingStream()
        assert not isinstance(obj, LLMProvider)

    def test_missing_model_name_violates_protocol(self) -> None:
        """A class missing model_name() fails the isinstance check."""
        obj = _MissingModelName()
        assert not isinstance(obj, LLMProvider)

    def test_protocol_is_runtime_checkable(self) -> None:
        """LLMProvider is decorated with @runtime_checkable."""
        # Attempting isinstance on a non-runtime_checkable protocol raises TypeError.
        # If this doesn't raise, the protocol is runtime_checkable.
        assert isinstance(_ValidProvider(), LLMProvider)


class TestProviderError:
    """Tests for the ProviderError exception class."""

    def test_inherits_from_exception(self) -> None:
        """ProviderError is a subclass of Exception."""
        assert issubclass(ProviderError, Exception)

    def test_message_stored(self) -> None:
        """The error message is accessible via str()."""
        err = ProviderError("something went wrong")
        assert str(err) == "something went wrong"

    def test_provider_attribute(self) -> None:
        """The provider attribute stores the provider name."""
        err = ProviderError("oops", provider="anthropic")
        assert err.provider == "anthropic"

    def test_original_error_attribute(self) -> None:
        """The original_error attribute stores the wrapped exception."""
        cause = RuntimeError("root cause")
        err = ProviderError("wrapped", original_error=cause)
        assert err.original_error is cause

    def test_default_attributes(self) -> None:
        """Default provider is '' and original_error is None."""
        err = ProviderError("msg")
        assert err.provider == ""
        assert err.original_error is None

    def test_can_be_caught_as_exception(self) -> None:
        """ProviderError can be caught with a bare 'except Exception'."""
        with pytest.raises(Exception, match="test"):
            raise ProviderError("test")
