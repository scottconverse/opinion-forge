"""Application configuration using pydantic-settings, loading from environment variables."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """OpinionForge application settings loaded from environment variables.

    Attributes:
        opinionforge_llm_provider: LLM provider to use.
        anthropic_api_key: API key for Anthropic Claude models.
        openai_api_key: API key for OpenAI models.
        opinionforge_search_api_key: API key for the web search provider.
        opinionforge_search_provider: Search provider for source research.
        opinionforge_ollama_base_url: Base URL for the Ollama server.
        opinionforge_model: Optional model name override.
        opinionforge_host: Host to bind the web server to.
        opinionforge_port: Port to bind the web server to (default 8484 per PRD).
    """

    opinionforge_llm_provider: Literal[
        "anthropic", "openai", "ollama", "openai_compatible"
    ] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    opinionforge_search_api_key: str | None = None
    opinionforge_search_provider: Literal["tavily", "brave", "serpapi"] = "tavily"
    opinionforge_ollama_base_url: str = "http://localhost:11434"
    opinionforge_model: str | None = None
    opinionforge_host: str = "127.0.0.1"
    opinionforge_port: int = 8484

    def require_llm_api_key(self) -> str:
        """Return the active LLM API key or exit with code 5 if not configured.

        Returns:
            The API key string for the configured LLM provider.

        Raises:
            SystemExit: With exit code 5 if the required API key is missing.
        """
        if self.opinionforge_llm_provider == "anthropic":
            if not self.anthropic_api_key:
                print(
                    "Error: ANTHROPIC_API_KEY is not set. "
                    "Please set it in your environment or .env file.",
                    file=sys.stderr,
                )
                raise SystemExit(5)
            return self.anthropic_api_key
        elif self.opinionforge_llm_provider == "openai":
            if not self.openai_api_key:
                print(
                    "Error: OPENAI_API_KEY is not set. "
                    "Please set it in your environment or .env file.",
                    file=sys.stderr,
                )
                raise SystemExit(5)
            return self.openai_api_key
        else:
            # ollama and openai_compatible don't always need API keys
            return ""

    def require_search_api_key(self) -> str:
        """Return the search API key or exit with code 5 if not configured.

        Returns:
            The search API key string.

        Raises:
            SystemExit: With exit code 5 if the search API key is missing.
        """
        if not self.opinionforge_search_api_key:
            print(
                "Error: OPINIONFORGE_SEARCH_API_KEY is not set. "
                "Please set it in your environment or .env file.",
                file=sys.stderr,
            )
            raise SystemExit(5)
        return self.opinionforge_search_api_key

    async def get_provider_config(self) -> "ProviderConfig":
        """Read provider configuration from storage first, env vars as fallback.

        Attempts to read a ProviderConfig from the SQLite settings store.
        If storage is unavailable or has no config, builds a ProviderConfig
        from environment variable settings.

        Returns:
            A ProviderConfig instance describing the active LLM provider.
        """
        from opinionforge.models.config import ProviderConfig

        # Try reading from storage first
        try:
            from opinionforge.storage import Database, SettingsStore, get_db_path

            db_path = get_db_path()
            if db_path.exists():
                async with Database(db_path) as db:
                    store = SettingsStore(db)
                    stored = await store.get_provider_config()
                    if stored is not None:
                        return stored
        except Exception:
            logger.debug("Could not read provider config from storage, using env vars")

        # Fallback to env vars
        return self._provider_config_from_env()

    def _provider_config_from_env(self) -> "ProviderConfig":
        """Build a ProviderConfig from current environment variable settings.

        Returns:
            A ProviderConfig derived from env-var-based Settings fields.
        """
        from opinionforge.models.config import ProviderConfig

        provider_type = self.opinionforge_llm_provider

        if provider_type == "anthropic":
            return ProviderConfig(
                provider_type="anthropic",
                model=self.opinionforge_model or "claude-sonnet-4-20250514",
                api_key=self.anthropic_api_key,
            )
        elif provider_type == "openai":
            return ProviderConfig(
                provider_type="openai",
                model=self.opinionforge_model or "gpt-4o",
                api_key=self.openai_api_key,
            )
        elif provider_type == "ollama":
            return ProviderConfig(
                provider_type="ollama",
                model=self.opinionforge_model or "llama3",
                base_url=self.opinionforge_ollama_base_url,
            )
        else:
            # openai_compatible
            return ProviderConfig(
                provider_type="openai_compatible",
                model=self.opinionforge_model or "default",
                base_url=self.opinionforge_ollama_base_url,
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance.

    Returns:
        The application Settings, loaded from environment variables.
    """
    return Settings()
