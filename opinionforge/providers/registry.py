"""Provider registry for discovering, instantiating, and testing LLM providers.

Provides a factory for creating provider instances by type string, plus
helper utilities for detecting Ollama, listing its models, and testing
provider connectivity.
"""

from __future__ import annotations

import httpx

from opinionforge.providers.base import LLMProvider, ProviderError
from opinionforge.providers.anthropic import AnthropicProvider
from opinionforge.providers.ollama import OllamaProvider
from opinionforge.providers.openai_compatible import OpenAICompatibleProvider
from opinionforge.providers.openai_provider import OpenAIProvider
from opinionforge.models.config import ProviderConfig


_PROVIDER_TYPES = {
    "anthropic",
    "openai",
    "openai_compatible",
    "ollama",
}


class ProviderRegistry:
    """Registry for creating and managing LLM provider instances.

    Supports four provider types: ``'anthropic'``, ``'openai'``,
    ``'openai_compatible'``, and ``'ollama'``.
    """

    def create_provider(self, provider_type: str, **kwargs: object) -> LLMProvider:
        """Create an LLM provider instance by type.

        Args:
            provider_type: One of ``'anthropic'``, ``'openai'``,
                ``'openai_compatible'``, or ``'ollama'``.
            **kwargs: Arguments forwarded to the provider constructor.

        Returns:
            An instantiated provider implementing LLMProvider.

        Raises:
            ValueError: If ``provider_type`` is not recognized.
        """
        if provider_type == "anthropic":
            return AnthropicProvider(**kwargs)  # type: ignore[arg-type]
        elif provider_type == "openai":
            return OpenAIProvider(**kwargs)  # type: ignore[arg-type]
        elif provider_type == "openai_compatible":
            return OpenAICompatibleProvider(**kwargs)  # type: ignore[arg-type]
        elif provider_type == "ollama":
            return OllamaProvider(**kwargs)  # type: ignore[arg-type]
        else:
            raise ValueError(
                f"Unknown provider type '{provider_type}'. "
                f"Supported types: {sorted(_PROVIDER_TYPES)}"
            )

    async def test_connection(
        self, provider: LLMProvider
    ) -> tuple[bool, str]:
        """Send a minimal request to verify provider connectivity.

        Args:
            provider: An instantiated LLMProvider to test.

        Returns:
            A ``(success, message)`` tuple.  ``success`` is True when the
            provider responds, False on error.
        """
        try:
            response = await provider.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'ok'.",
                max_tokens=10,
            )
            return True, f"Connected to {provider.model_name()}: {response[:50]}"
        except ProviderError as exc:
            return False, f"Connection failed for {provider.model_name()}: {exc}"

    async def detect_ollama(
        self, base_url: str = "http://localhost:11434"
    ) -> bool:
        """Check whether an Ollama server is reachable.

        Args:
            base_url: The Ollama server URL to probe.

        Returns:
            True if Ollama responds, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(base_url, timeout=5.0)
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_ollama_models(
        self, base_url: str = "http://localhost:11434"
    ) -> list[str]:
        """Query Ollama for available model names.

        Args:
            base_url: The Ollama server URL.

        Returns:
            A list of model name strings.

        Raises:
            ProviderError: If Ollama is unreachable.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/api/tags", timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except httpx.ConnectError as exc:
            raise ProviderError(
                "Ollama is not running — start it with 'ollama serve'.",
                provider="ollama",
                original_error=exc,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "Timed out connecting to Ollama.",
                provider="ollama",
                original_error=exc,
            ) from exc


def get_provider(config: ProviderConfig) -> LLMProvider:
    """Convenience factory that reads a ProviderConfig and returns an
    instantiated provider.

    Args:
        config: A :class:`~opinionforge.models.config.ProviderConfig`
            describing the provider type, model, and optional credentials.

    Returns:
        An instantiated provider implementing LLMProvider.

    Raises:
        ValueError: If ``config.provider_type`` is not recognised.
    """
    kwargs: dict[str, object] = {"model": config.model}
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url

    registry = ProviderRegistry()
    return registry.create_provider(config.provider_type, **kwargs)
