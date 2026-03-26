"""Pluggable LLM provider backends for OpinionForge.

Exports the core protocol, registry, convenience factory, and all concrete
provider adapters.
"""

from opinionforge.providers.base import LLMProvider, ProviderError
from opinionforge.providers.registry import ProviderRegistry, get_provider

__all__ = [
    "LLMProvider",
    "ProviderError",
    "ProviderRegistry",
    "get_provider",
]
