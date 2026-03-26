"""LLMProvider protocol and ProviderError exception for pluggable LLM backends.

Defines the common interface that all provider adapters must implement,
along with the unified exception type for provider-specific errors.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


class ProviderError(Exception):
    """Unified exception for provider-specific errors.

    Wraps SDK-specific exceptions (authentication failures, network errors,
    missing models, etc.) with descriptive messages so callers do not need
    to handle raw SDK exception types.

    Attributes:
        provider: The provider name that raised the error (e.g. 'anthropic').
        original_error: The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for pluggable LLM provider adapters.

    All provider adapters must implement these three methods.  The protocol
    is ``@runtime_checkable`` so ``isinstance`` checks work at runtime.
    """

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Generate a complete text response from the LLM.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text string.

        Raises:
            ProviderError: On authentication, network, or API errors.
        """
        ...

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream token strings from the LLM.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Individual token strings as they arrive.

        Raises:
            ProviderError: On authentication, network, or API errors.
        """
        ...

    def model_name(self) -> str:
        """Return a human-readable identifier for the provider and model.

        Returns:
            A string in the form ``'{provider}/{model}'``.
        """
        ...
