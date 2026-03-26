"""OpenAI provider adapter using the openai SDK.

Implements the LLMProvider protocol for OpenAI models,
wrapping the ``openai.AsyncOpenAI`` client.
"""

from __future__ import annotations

from typing import AsyncIterator

import openai

from opinionforge.providers.base import ProviderError


class OpenAIProvider:
    """LLM provider backed by the OpenAI API.

    Args:
        api_key: The OpenAI API key.
        model: The model identifier. Defaults to ``'gpt-4o'``.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Generate a complete text response via the OpenAI chat completions API.

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
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except openai.AuthenticationError as exc:
            raise ProviderError(
                "OpenAI authentication failed — check your API key.",
                provider="openai",
                original_error=exc,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(
                f"Failed to connect to OpenAI API: {exc}",
                provider="openai",
                original_error=exc,
            ) from exc
        except openai.APIError as exc:
            raise ProviderError(
                f"OpenAI API error: {exc}",
                provider="openai",
                original_error=exc,
            ) from exc

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream token strings from the OpenAI chat completions API.

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
            stream = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except openai.AuthenticationError as exc:
            raise ProviderError(
                "OpenAI authentication failed — check your API key.",
                provider="openai",
                original_error=exc,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(
                f"Failed to connect to OpenAI API: {exc}",
                provider="openai",
                original_error=exc,
            ) from exc
        except openai.APIError as exc:
            raise ProviderError(
                f"OpenAI API error: {exc}",
                provider="openai",
                original_error=exc,
            ) from exc

    def model_name(self) -> str:
        """Return the provider/model identifier.

        Returns:
            A string in the form ``'openai/{model}'``.
        """
        return f"openai/{self._model}"
