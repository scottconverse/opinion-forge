"""ModeBlendConfig, StanceConfig, ImagePromptConfig, ProviderConfig, SearchConfig, and UserPreferences models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModeBlendConfig(BaseModel):
    """Configuration for blending multiple rhetorical modes.

    Attributes:
        modes: List of (mode_id, weight) pairs where weights sum to 100.
            Maximum 3 modes per blend.
    """


    modes: list[tuple[str, float]]

    @model_validator(mode="after")
    def validate_blend(self) -> ModeBlendConfig:
        """Validate that the mode blend configuration is well-formed.

        Checks that there is at least one mode, at most three modes,
        and that weights sum to 100 (within 0.01 tolerance).

        Returns:
            The validated ModeBlendConfig instance.

        Raises:
            ValueError: If validation constraints are violated.
        """
        if len(self.modes) == 0:
            raise ValueError("At least one mode is required")
        if len(self.modes) > 3:
            raise ValueError("Maximum 3 modes in a blend")
        total_weight = sum(w for _, w in self.modes)
        if abs(total_weight - 100) > 0.01:
            raise ValueError(
                f"Blend weights must sum to 100, got {total_weight}"
            )
        return self


class StanceConfig(BaseModel):
    """Stance and intensity controls for rhetorical positioning.

    Decouples argumentative emphasis direction (position) from rhetorical heat
    (intensity), allowing independent control over both dimensions.

    Attributes:
        position: Argumentative emphasis direction from -100 (equity-focused)
            to +100 (liberty-focused). Zero is balanced.
        intensity: Rhetorical heat from 0.0 (deliberative/measured) to
            1.0 (maximum conviction and force). Default is 0.5.
    """

    position: int = Field(default=0, ge=-100, le=100)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def label(self) -> str:
        """Return a non-political descriptive label for the current position.

        Uses equity/liberty vocabulary rather than left/right political labels.

        Returns:
            One of 'strongly_equity_focused', 'equity_leaning', 'balanced',
            'liberty_leaning', or 'strongly_liberty_focused'.
        """
        if self.position <= -50:
            return "strongly_equity_focused"
        if self.position < -25:
            return "equity_leaning"
        if self.position <= 25:
            return "balanced"
        if self.position < 50:
            return "liberty_leaning"
        return "strongly_liberty_focused"


# Platform aspect ratios per PRD specification
_PLATFORM_ASPECT_RATIOS: dict[str, str] = {
    "substack": "16:9",
    "medium": "16:9",
    "wordpress": "16:9",
    "facebook": "1.91:1",
    "twitter": "16:9",
    "instagram": "1:1",
}

# Platform pixel dimensions per PRD specification: (width, height)
_PLATFORM_DIMENSIONS: dict[str, tuple[int, int]] = {
    "substack": (1456, 819),
    "medium": (1400, 788),
    "wordpress": (1200, 675),
    "facebook": (1200, 628),
    "twitter": (1600, 900),
    "instagram": (1080, 1080),
}


class ImagePromptConfig(BaseModel):
    """Configuration for header image prompt generation.

    Attributes:
        style: Visual style directive for the generated image.
        platform: Target publishing platform, which determines aspect ratio and dimensions.
        custom_keywords: Optional user-provided keywords to include in the image prompt.
    """

    style: Literal[
        "photorealistic", "editorial", "cartoon", "minimalist", "vintage", "abstract"
    ] = "editorial"
    platform: Literal[
        "substack", "medium", "wordpress", "facebook", "twitter", "instagram"
    ] = "substack"
    custom_keywords: list[str] = Field(default_factory=list)

    @property
    def aspect_ratio(self) -> str:
        """Return the aspect ratio string for the configured platform.

        Returns:
            Aspect ratio string (e.g. '16:9', '1:1', '1.91:1').
        """
        return _PLATFORM_ASPECT_RATIOS[self.platform]

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the pixel dimensions (width, height) for the configured platform.

        Returns:
            A tuple of (width, height) in pixels.
        """
        return _PLATFORM_DIMENSIONS[self.platform]


# ---------------------------------------------------------------------------
# Provider, Search, and User-preferences models (Sprint 12)
# ---------------------------------------------------------------------------

_VALID_PROVIDER_TYPES = ("ollama", "anthropic", "openai", "openai_compatible")


class ProviderConfig(BaseModel):
    """LLM provider configuration for OpinionForge.

    Attributes:
        provider_type: The LLM provider backend.  Must be one of
            ``'ollama'``, ``'anthropic'``, ``'openai'``, or
            ``'openai_compatible'``.
        model: Model identifier (e.g. ``'claude-3-opus-20240229'``).
        api_key: Optional API key for the provider.
        base_url: Optional custom endpoint URL.
        max_tokens: Maximum tokens for generation.  Defaults to 4096.
    """

    provider_type: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = Field(default=4096)

    @model_validator(mode="after")
    def validate_provider_type(self) -> ProviderConfig:
        """Validate that provider_type is one of the supported backends.

        Returns:
            The validated ProviderConfig instance.

        Raises:
            ValueError: If provider_type is not a recognised backend.
        """
        if self.provider_type not in _VALID_PROVIDER_TYPES:
            raise ValueError(
                f"provider_type must be one of {_VALID_PROVIDER_TYPES}, "
                f"got '{self.provider_type}'"
            )
        return self


class SearchConfig(BaseModel):
    """Search provider configuration.

    Attributes:
        provider: Search backend identifier (e.g. ``'tavily'``).
        api_key: Optional API key for the search provider.
    """

    provider: str
    api_key: str | None = None


class UserPreferences(BaseModel):
    """User preference defaults persisted to the settings store.

    Attributes:
        default_mode: Default rhetorical mode id.
        default_stance: Default stance position (-100 to 100).
        default_intensity: Default rhetorical intensity (0.0 to 1.0).
        default_length: Default output length preset.
        theme: UI colour theme.
        auto_launch: Whether to auto-launch the web UI on start.
        onboarding_completed: Whether the user has finished onboarding.
    """

    default_mode: str = "analytical"
    default_stance: int = 0
    default_intensity: float = 0.5
    default_length: str = "standard"
    theme: str = "light"
    auto_launch: bool = False
    onboarding_completed: bool = False

