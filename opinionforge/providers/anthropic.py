"""Anthropic Claude provider adapter using the anthropic SDK.

Implements the LLMProvider protocol for Anthropic's Claude models,
wrapping the ``anthropic.AsyncAnthropic`` client.
"""

from __future__ import annotations

from typing import AsyncIterator

import anthropic

from opinionforge.providers.base import ProviderError


class AnthropicProvider:
    """LLM provider backed by Anthropic Claude.

    Args:
        api_key: The Anthropic API key.
        model: The model identifier. Defaults to ``'claude-sonnet-4-20250514'``.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Generate a complete text response via the Anthropic messages API.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text string.

        Raises:
            ProviderError: On authentication or network errors.
        """
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except anthropic.AuthenticationError as exc:
            raise ProviderError(
                "Anthropic authentication failed — check your API key.",
                provider="anthropic",
                original_error=exc,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Failed to connect to Anthropic API: {exc}",
                provider="anthropic",
                original_error=exc,
            ) from exc
        except anthropic.APIError as exc:
            raise ProviderError(
                f"Anthropic API error: {exc}",
                provider="anthropic",
                original_error=exc,
            ) from exc

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream token strings from the Anthropic messages API.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Individual token strings as they arrive.

        Raises:
            ProviderError: On authentication or network errors.
        """
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError as exc:
            raise ProviderError(
                "Anthropic authentication failed — check your API key.",
                provider="anthropic",
                original_error=exc,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Failed to connect to Anthropic API: {exc}",
                provider="anthropic",
                original_error=exc,
            ) from exc
        except anthropic.APIError as exc:
            raise ProviderError(
                f"Anthropic API error: {exc}",
                provider="anthropic",
                original_error=exc,
            ) from exc

    def model_name(self) -> str:
        """Return the provider/model identifier.

        Returns:
            A string in the form ``'anthropic/{model}'``.
        """
        return f"anthropic/{self._model}"
