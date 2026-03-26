"""Tone preview generator that produces a 2-3 sentence preview using the composed voice prompt.

The preview captures the opening hook or thesis statement style without
performing a full research cycle.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from opinionforge.config import Settings, get_settings
from opinionforge.models.config import StanceConfig
from opinionforge.models.topic import TopicContext

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client abstraction, enabling dependency injection for tests."""

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text from the LLM.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        ...


class AnthropicLLMClient:
    """LLM client backed by the Anthropic Claude API.

    Args:
        api_key: The Anthropic API key.
        model: The model to use. Defaults to 'claude-sonnet-4-20250514'.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text using the Anthropic Claude API.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class OpenAILLMClient:
    """LLM client backed by the OpenAI API.

    Args:
        api_key: The OpenAI API key.
        model: The model to use. Defaults to 'gpt-4o'.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text using the OpenAI API.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class ProviderLLMClient:
    """Sync LLMClient wrapper around an async LLMProvider.

    Bridges the async provider interface into the existing sync LLMClient
    protocol by running async calls in an event loop.  This preserves
    backward compatibility with all code that consumes LLMClient.

    Args:
        provider: An LLMProvider instance to wrap.
    """

    def __init__(self, provider: object) -> None:
        self._provider = provider

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text by delegating to the async provider's generate method.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: If the provider raises a ProviderError.
        """
        from opinionforge.providers.base import ProviderError

        coro = self._provider.generate(  # type: ignore[union-attr]
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        try:
            # If we're already inside an event loop (e.g. FastAPI), use
            # a new thread to avoid "cannot run nested event loop" errors.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            else:
                result = asyncio.run(coro)

            return result
        except ProviderError as exc:
            raise RuntimeError(str(exc)) from exc

    @property
    def provider(self) -> object:
        """Return the underlying LLMProvider instance.

        Returns:
            The wrapped async LLMProvider.
        """
        return self._provider


def create_llm_client_from_provider(provider: object) -> LLMClient:
    """Wrap an async LLMProvider into the sync LLMClient interface.

    This is the bridge function that allows async providers from the
    provider layer (Sprint 011) to be used anywhere the existing sync
    LLMClient protocol is expected.

    Args:
        provider: An LLMProvider instance.

    Returns:
        An LLMClient that delegates to the provider.
    """
    return ProviderLLMClient(provider)


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create an LLM client based on configuration.

    Reads ProviderConfig from storage (via Settings.get_provider_config)
    when possible, falling back to env vars for backward compatibility.
    For providers that require API keys (anthropic, openai), falls back
    to the legacy direct-client path when the key is available.

    Args:
        settings: Application settings. Uses default settings if None.

    Returns:
        An LLMClient implementation for the configured provider.
    """
    if settings is None:
        settings = get_settings()

    provider_type = settings.opinionforge_llm_provider

    # For the legacy anthropic/openai providers with API keys set,
    # use the direct sync clients for backward compatibility
    if provider_type == "anthropic" and settings.anthropic_api_key:
        if settings.opinionforge_model:
            return AnthropicLLMClient(api_key=settings.anthropic_api_key, model=settings.opinionforge_model)
        return AnthropicLLMClient(api_key=settings.anthropic_api_key)
    elif provider_type == "openai" and settings.openai_api_key:
        if settings.opinionforge_model:
            return OpenAILLMClient(api_key=settings.openai_api_key, model=settings.opinionforge_model)
        return OpenAILLMClient(api_key=settings.openai_api_key)

    # For all provider types (including ollama, openai_compatible, or
    # anthropic/openai without direct keys), use the provider layer
    try:
        config = settings._provider_config_from_env()
        from opinionforge.providers import get_provider

        provider = get_provider(config)
        return create_llm_client_from_provider(provider)
    except Exception:
        # Final fallback: try the legacy path with require_llm_api_key
        api_key = settings.require_llm_api_key()
        if provider_type == "anthropic":
            return AnthropicLLMClient(api_key=api_key)
        else:
            return OpenAILLMClient(api_key=api_key)


def generate_preview(
    topic: TopicContext,
    voice_prompt: str,
    spectrum: StanceConfig,
    *,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> str:
    """Generate a 2-3 sentence tone preview in the selected voice.

    Captures the opening hook or thesis statement style without performing
    a full research cycle. Uses a single short LLM call.

    Args:
        topic: The normalized topic context.
        voice_prompt: The composed voice prompt (after blending and spectrum).
        spectrum: The stance configuration for context.
        client: Optional LLM client for dependency injection (used in tests).
        settings: Optional settings override.

    Returns:
        A 2-3 sentence preview string.

    Raises:
        RuntimeError: If the LLM API call fails.
    """
    if client is None:
        client = create_llm_client(settings)

    user_prompt = (
        f"Write a 2-3 sentence preview (opening hook or thesis statement) for an opinion piece "
        f"about the following topic. This is a tone preview only -- do not write the full piece.\n\n"
        f"Topic: {topic.title}\n"
        f"Summary: {topic.summary}\n"
    )
    if topic.key_claims:
        user_prompt += f"Key claims: {'; '.join(topic.key_claims[:3])}\n"

    try:
        return client.generate(
            system_prompt=voice_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate preview: {exc}. "
            "Check your API key configuration and network connection."
        ) from exc
