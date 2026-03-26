"""Application configuration using pydantic-settings, loading from environment variables."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """OpinionForge application settings loaded from environment variables.

    Attributes:
        opinionforge_llm_provider: LLM provider to use ('anthropic' or 'openai').
        anthropic_api_key: API key for Anthropic Claude models.
        openai_api_key: API key for OpenAI models.
        opinionforge_search_api_key: API key for the web search provider.
        opinionforge_search_provider: Search provider for source research.
    """

    opinionforge_llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    opinionforge_search_api_key: str | None = None
    opinionforge_search_provider: Literal["tavily", "brave", "serpapi"] = "tavily"
    opinionforge_host: str = "127.0.0.1"
    opinionforge_port: int = 8000

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
        else:
            if not self.openai_api_key:
                print(
                    "Error: OPENAI_API_KEY is not set. "
                    "Please set it in your environment or .env file.",
                    file=sys.stderr,
                )
                raise SystemExit(5)
            return self.openai_api_key

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance.

    Returns:
        The application Settings, loaded from environment variables.
    """
    return Settings()
