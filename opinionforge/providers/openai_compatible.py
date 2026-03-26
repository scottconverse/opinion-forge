"""OpenAI-compatible provider for LM Studio, vLLM, text-generation-inference, etc.

Implements the LLMProvider protocol using the ``openai.AsyncOpenAI`` client
with a custom ``base_url``, enabling communication with any OpenAI-compatible
endpoint.
"""

from __future__ import annotations

from typing import AsyncIterator

import openai

from opinionforge.providers.base import ProviderError


class OpenAICompatibleProvider:
    """LLM provider for any OpenAI-compatible API endpoint.

    Args:
        base_url: The base URL of the OpenAI-compatible server
                  (e.g. ``'http://localhost:1234/v1'``).
        model: The model identifier to request.
        api_key: Optional API key. Many local servers don't require one.
    """

    def __init__(
        self, base_url: str, model: str, api_key: str | None = None
    ) -> None:
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
        )
        self._model = model
        self._base_url = base_url

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Generate a complete text response from the compatible endpoint.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text string.

        Raises:
            ProviderError: On connection or API errors.
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
        except openai.APIConnectionError as exc:
            raise ProviderError(
                f"Cannot reach OpenAI-compatible endpoint at {self._base_url}: {exc}",
                provider="openai_compatible",
                original_error=exc,
            ) from exc
        except openai.AuthenticationError as exc:
            raise ProviderError(
                "Authentication failed for OpenAI-compatible endpoint.",
                provider="openai_compatible",
                original_error=exc,
            ) from exc
        except openai.APIError as exc:
            raise ProviderError(
                f"OpenAI-compatible API error: {exc}",
                provider="openai_compatible",
                original_error=exc,
            ) from exc

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream token strings from the compatible endpoint.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Individual token strings as they arrive.

        Raises:
            ProviderError: On connection or API errors.
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
        except openai.APIConnectionError as exc:
            raise ProviderError(
                f"Cannot reach OpenAI-compatible endpoint at {self._base_url}: {exc}",
                provider="openai_compatible",
                original_error=exc,
            ) from exc
        except openai.AuthenticationError as exc:
            raise ProviderError(
                "Authentication failed for OpenAI-compatible endpoint.",
                provider="openai_compatible",
                original_error=exc,
            ) from exc
        except openai.APIError as exc:
            raise ProviderError(
                f"OpenAI-compatible API error: {exc}",
                provider="openai_compatible",
                original_error=exc,
            ) from exc

    def model_name(self) -> str:
        """Return the provider/model identifier.

        Returns:
            A string in the form ``'openai_compatible/{model}'``.
        """
        return f"openai_compatible/{self._model}"
