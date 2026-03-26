"""Ollama local LLM provider adapter using the Ollama REST API via httpx.

Implements the LLMProvider protocol by communicating directly with the
Ollama ``/api/generate`` endpoint, avoiding the less-mature Ollama Python SDK.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from opinionforge.providers.base import ProviderError


class OllamaProvider:
    """LLM provider backed by a local Ollama server.

    Args:
        model: The Ollama model name (e.g. ``'llama3'``).
        base_url: The Ollama server URL. Defaults to ``'http://localhost:11434'``.
    """

    def __init__(
        self, model: str, base_url: str = "http://localhost:11434"
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        """Generate a complete text response from Ollama.

        Posts to ``/api/generate`` with ``stream=false`` and returns the
        full response text.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text string.

        Raises:
            ProviderError: On connection errors or when the model is not found.
        """
        payload = {
            "model": self._model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=120.0,
                )
                if response.status_code == 404:
                    raise ProviderError(
                        f"Ollama model '{self._model}' not found — "
                        f"pull it with 'ollama pull {self._model}'.",
                        provider="ollama",
                    )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.ConnectError as exc:
            raise ProviderError(
                "Ollama is not running — start it with 'ollama serve'.",
                provider="ollama",
                original_error=exc,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama request timed out: {exc}",
                provider="ollama",
                original_error=exc,
            ) from exc
        except ProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama API error (HTTP {exc.response.status_code}): {exc}",
                provider="ollama",
                original_error=exc,
            ) from exc

    async def stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream token strings from Ollama.

        Posts to ``/api/generate`` with ``stream=true`` and yields token
        strings parsed from the streamed JSON lines.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user-level prompt.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Individual token strings as they arrive.

        Raises:
            ProviderError: On connection errors or when the model is not found.
        """
        payload = {
            "model": self._model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=120.0,
                ) as response:
                    if response.status_code == 404:
                        raise ProviderError(
                            f"Ollama model '{self._model}' not found — "
                            f"pull it with 'ollama pull {self._model}'.",
                            provider="ollama",
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
        except httpx.ConnectError as exc:
            raise ProviderError(
                "Ollama is not running — start it with 'ollama serve'.",
                provider="ollama",
                original_error=exc,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama request timed out: {exc}",
                provider="ollama",
                original_error=exc,
            ) from exc
        except ProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama API error (HTTP {exc.response.status_code}): {exc}",
                provider="ollama",
                original_error=exc,
            ) from exc

    def model_name(self) -> str:
        """Return the provider/model identifier.

        Returns:
            A string in the form ``'ollama/{model}'``.
        """
        return f"ollama/{self._model}"
