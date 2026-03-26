"""Main LLM-based piece generation engine.

Composes the full system prompt from mode, stance, length, and research
context, then generates the complete opinion piece.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from opinionforge.core.preview import LLMClient, create_llm_client
from opinionforge.core.stance import apply_stance
from opinionforge.config import Settings
from opinionforge.models.config import ImagePromptConfig, ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece
from opinionforge.models.topic import TopicContext

logger = logging.getLogger(__name__)


# Mandatory fixed disclaimer — not constructed dynamically from any profile data.
MANDATORY_DISCLAIMER: str = (
    "This piece was generated with AI-assisted rhetorical controls. "
    "It is original content and is not written by, endorsed by, or affiliated with any real person."
)

# Length presets from the PRD
LENGTH_PRESETS: dict[str, int] = {
    "short": 500,
    "standard": 800,
    "long": 1200,
    "essay": 2500,
    "feature": 5000,
}

MIN_WORD_COUNT = 200
MAX_WORD_COUNT = 8000


def resolve_length(length_input: str | int) -> int:
    """Resolve a length preset name or custom word count to an integer.

    Args:
        length_input: Either a preset name ('short', 'standard', 'long',
            'essay', 'feature') or a custom word count integer.

    Returns:
        The target word count as an integer.

    Raises:
        ValueError: If the preset name is unknown or the word count is
            outside the 200-8000 range.
    """
    if isinstance(length_input, str):
        key = length_input.lower().strip()
        if key in LENGTH_PRESETS:
            return LENGTH_PRESETS[key]
        # Try parsing as integer string
        try:
            count = int(key)
        except ValueError:
            valid = ", ".join(sorted(LENGTH_PRESETS.keys()))
            raise ValueError(
                f"Unknown length preset '{length_input}'. "
                f"Valid presets: {valid}. Or provide a word count between "
                f"{MIN_WORD_COUNT} and {MAX_WORD_COUNT}."
            ) from None
        length_input = count

    if not isinstance(length_input, int):
        length_input = int(length_input)

    if length_input < MIN_WORD_COUNT or length_input > MAX_WORD_COUNT:
        raise ValueError(
            f"Word count {length_input} is outside the allowed range "
            f"({MIN_WORD_COUNT}-{MAX_WORD_COUNT})."
        )

    return length_input


def _length_instructions(target_length: int) -> str:
    """Generate length-specific writing instructions.

    Args:
        target_length: The target word count.

    Returns:
        Instruction string for length targeting.
    """
    tolerance = int(target_length * 0.1)

    if target_length <= 500:
        structure = (
            "This is a short op-ed. Prioritize the single strongest argument "
            "with minimal preamble. Get to the thesis immediately."
        )
    elif target_length <= 800:
        structure = (
            "This is a standard op-ed. Present the thesis clearly, develop "
            "2-3 supporting arguments, and close with a strong conclusion."
        )
    elif target_length <= 1200:
        structure = (
            "This is an extended column. Develop multiple arguments with "
            "supporting evidence. Include a clear thesis and substantive conclusion."
        )
    elif target_length <= 2500:
        structure = (
            "This is a magazine essay. Include section development, nuanced argumentation, "
            "and detailed evidence. Use subheadings if stylistically appropriate."
        )
    else:
        structure = (
            "This is a long-form feature. Include section breaks, subheadings "
            "(when stylistically appropriate), deeply developed arguments, "
            "rich evidence, and narrative structure."
        )

    return (
        f"\n\n--- Length Instructions ---\n"
        f"Target word count: {target_length} words (acceptable range: "
        f"{target_length - tolerance}-{target_length + tolerance}).\n"
        f"{structure}"
    )


def compose_system_prompt(
    mode_prompt: str,
    stance: StanceConfig,
    length_target: int,
    research_context: str | None = None,
) -> str:
    """Assemble the full system prompt in PRD order.

    Order: base role -> mode fragment (with stance modifier already applied)
    -> length instructions -> research context -> output format.

    Args:
        mode_prompt: The composed mode prompt (already includes stance modifier).
        stance: The stance configuration.
        length_target: Target word count.
        research_context: Optional research findings to include.

    Returns:
        The complete system prompt string.
    """
    parts: list[str] = []

    # 1. Base role
    parts.append(
        "You are OpinionForge, an expert opinion writing engine. "
        "You generate publication-ready opinion pieces -- op-eds, columns, "
        "and long-form opinion content. You write with conviction, rhetorical "
        "skill, and argumentative coherence. Every piece you produce is original "
        "AI-generated content."
    )

    # 2. Mode fragment (with stance modifier already applied)
    parts.append(f"\n\n--- Rhetorical Mode ---\n{mode_prompt}")

    # 3. Length instructions
    parts.append(_length_instructions(length_target))

    # 4. Research context (if available)
    if research_context:
        parts.append(
            f"\n\n--- Research Context ---\n"
            f"Use the following research findings to support your arguments "
            f"with factual claims and evidence:\n\n{research_context}"
        )

    # 5. Output format instructions
    parts.append(
        "\n\n--- Output Format ---\n"
        "Produce the opinion piece with:\n"
        "1. A compelling headline/title\n"
        "2. The full body text in markdown\n"
        "3. Use '## Title' for the headline on the first line\n"
        "4. Separate the title from the body with a blank line"
    )

    return "".join(parts)


def _parse_generated_output(raw_output: str) -> tuple[str, str]:
    """Parse the LLM output into title and body.

    Args:
        raw_output: The raw text from the LLM.

    Returns:
        A tuple of (title, body).
    """
    lines = raw_output.strip().split("\n")

    title = ""
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            body_start = i + 1
            break
        elif stripped.startswith("# "):
            title = stripped[2:].strip()
            body_start = i + 1
            break
        elif stripped and not title:
            title = stripped
            body_start = i + 1
            break

    body = "\n".join(lines[body_start:]).strip()

    if not title:
        title = "Untitled Opinion Piece"

    return title, body


def _get_provider_name(client: LLMClient | None) -> str:
    """Extract the provider/model name from an LLMClient if available.

    Args:
        client: The LLM client used for generation.

    Returns:
        A provider name string, or 'unknown' if not determinable.
    """
    from opinionforge.core.preview import ProviderLLMClient

    if isinstance(client, ProviderLLMClient):
        try:
            return client.provider.model_name()  # type: ignore[union-attr]
        except Exception:
            return "unknown"
    # Legacy clients
    if hasattr(client, "_model"):
        model = getattr(client, "_model", "unknown")
        if "AnthropicLLMClient" in type(client).__name__:
            return f"anthropic/{model}"
        elif "OpenAILLMClient" in type(client).__name__:
            return f"openai/{model}"
    return "unknown"


def _save_piece_to_storage(piece: GeneratedPiece, provider_name: str) -> None:
    """Best-effort save of a generated piece to SQLite storage.

    This function is non-blocking in spirit: if storage is unavailable
    (e.g. first run, no DB file), the save silently fails and generation
    still succeeds.

    Args:
        piece: The generated piece to persist.
        provider_name: The provider/model identifier string.
    """
    try:
        from opinionforge.storage import Database, PieceStore, get_db_path

        db_path = get_db_path()

        piece_data = {
            "id": piece.id,
            "topic": piece.topic.title,
            "title": piece.title,
            "subtitle": getattr(piece, "subtitle", None),
            "body": piece.body,
            "preview_text": piece.preview_text,
            "mode": provider_name,
            "mode_config": [
                {"mode": m, "weight": w} for m, w in piece.mode_config.modes
            ],
            "stance_position": piece.stance.position,
            "stance_intensity": piece.stance.intensity,
            "target_length": piece.target_length,
            "actual_length": piece.actual_length,
            "sources": [s.model_dump() for s in piece.sources] if piece.sources else [],
            "research_queries": piece.research_queries or [],
            "disclaimer": piece.disclaimer,
        }

        async def _do_save() -> None:
            async with Database(db_path) as db:
                store = PieceStore(db)
                await store.save(piece_data)

        # Run the async save — handle the case where we may or may not
        # be inside an existing event loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an event loop (e.g. web server) — schedule it
            # as a fire-and-forget task so it doesn't block.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, _do_save()).result(timeout=5)
        else:
            asyncio.run(_do_save())

    except Exception as exc:
        logger.debug("Best-effort storage save failed: %s", exc)


def generate_piece(
    topic: TopicContext,
    mode_config: ModeBlendConfig,
    stance: StanceConfig,
    target_length: int | str,
    research_context: str | None = None,
    *,
    image_config: ImagePromptConfig | None = None,
    client: LLMClient | None = None,
    settings: Settings | None = None,
    provider_type: str | None = None,
    model: str | None = None,
) -> GeneratedPiece:
    """Orchestrate full opinion piece generation.

    Loads mode profiles, applies stance, composes system prompt,
    calls the LLM, and returns a structured GeneratedPiece with the mandatory
    fixed disclaimer. Optionally generates an image prompt when image_config
    is provided.

    After successful generation, the piece is saved to SQLite storage
    (best-effort -- if the database is unavailable, generation still succeeds).

    Args:
        topic: The normalized topic context.
        mode_config: Rhetorical mode blend configuration.
        stance: Stance and intensity controls.
        target_length: Target word count (preset name or integer).
        research_context: Optional research findings.
        image_config: Optional image prompt configuration. When provided,
            generate_image_prompt() is called after piece generation and the
            result is stored in the returned GeneratedPiece.
        client: Optional LLM client for dependency injection.
        settings: Optional settings override.
        provider_type: Optional provider type override (e.g. 'ollama', 'anthropic').
            When set, overrides the configured default provider for this run.
        model: Optional model name override. When set, overrides the
            configured default model for this run.

    Returns:
        A GeneratedPiece with title, body, preview_text, and the mandatory
        fixed disclaimer string. If image_config is provided, image_prompt
        and image_platform are also populated.

    Raises:
        ValueError: If mode or length configuration is invalid.
        RuntimeError: If LLM generation fails.
    """
    # Resolve length
    resolved_length = resolve_length(target_length)

    # Build mode prompt
    from opinionforge.core.mode_engine import blend_modes
    mode_prompt = blend_modes(mode_config)

    # Apply stance modifier (no ideological_baseline — modes are ideologically neutral)
    modified_prompt = apply_stance(mode_prompt, stance)

    # Compose full system prompt
    system_prompt = compose_system_prompt(
        modified_prompt, stance, resolved_length, research_context
    )

    # Create LLM client if not injected
    if client is None:
        if provider_type is not None:
            # Provider override: create a provider via the registry
            from opinionforge.core.preview import create_llm_client_from_provider
            from opinionforge.providers import get_provider

            if settings is None:
                from opinionforge.config import get_settings
                settings = get_settings()

            from opinionforge.models.config import ProviderConfig

            resolved_model: str
            resolved_api_key: str | None = None
            resolved_base_url: str | None = None

            if provider_type == "anthropic":
                resolved_model = model or settings.opinionforge_model or "claude-sonnet-4-20250514"
                resolved_api_key = settings.anthropic_api_key or ""
            elif provider_type == "openai":
                resolved_model = model or settings.opinionforge_model or "gpt-4o"
                resolved_api_key = settings.openai_api_key or ""
            elif provider_type == "ollama":
                resolved_model = model or settings.opinionforge_model or "llama3"
                resolved_base_url = settings.opinionforge_ollama_base_url
            elif provider_type == "openai_compatible":
                resolved_model = model or settings.opinionforge_model or "default"
                resolved_base_url = settings.opinionforge_ollama_base_url
            else:
                resolved_model = model or settings.opinionforge_model or "default"

            provider_config = ProviderConfig(
                provider_type=provider_type,
                model=resolved_model,
                api_key=resolved_api_key or None,
                base_url=resolved_base_url,
            )
            provider = get_provider(provider_config)
            client = create_llm_client_from_provider(provider)
        else:
            client = create_llm_client(settings)

    # Build user prompt
    user_prompt = (
        f"Write an opinion piece about the following topic.\n\n"
        f"Topic: {topic.title}\n"
        f"Summary: {topic.summary}\n"
    )
    if topic.key_claims:
        user_prompt += f"Key claims: {'; '.join(topic.key_claims)}\n"
    if topic.key_entities:
        user_prompt += f"Key entities: {', '.join(topic.key_entities)}\n"
    if topic.subject_domain:
        user_prompt += f"Domain: {topic.subject_domain}\n"

    # Estimate max tokens (rough: 1.5 tokens per word + buffer)
    max_tokens = int(resolved_length * 1.5) + 500

    try:
        raw_output = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate opinion piece: {exc}. "
            "Check your API key configuration and network connection."
        ) from exc

    # Parse output
    title, body = _parse_generated_output(raw_output)

    # --- Similarity Screening (BEFORE disclaimer injection per PRD pipeline order) ---
    from opinionforge.core.similarity import screen_output

    research_texts: list[str] = []
    if research_context:
        research_texts = [research_context]

    screening_result = screen_output(
        text=body,
        research_texts=research_texts,
        mode_config=mode_config,
        client=client,
    )

    if not screening_result.passed:
        raise RuntimeError(
            "Similarity screening failed — output blocked. "
            f"Details: {screening_result.warning or 'unresolved similarity violations'}. "
            "Please regenerate or revise your request."
        )

    # Build preview text (first 2-3 sentences of body)
    sentences = body.replace("\n", " ").split(". ")
    preview_sentences = sentences[:3]
    preview_text = ". ".join(preview_sentences)
    if not preview_text.endswith("."):
        preview_text += "."

    # Count words
    actual_length = len(body.split())

    # Determine provider name for metadata
    provider_name = _get_provider_name(client)

    piece = GeneratedPiece(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        topic=topic,
        mode_config=mode_config,
        stance=stance,
        target_length=resolved_length,
        actual_length=actual_length,
        title=title,
        body=body,
        preview_text=preview_text,
        sources=[],
        research_queries=[],
        disclaimer=MANDATORY_DISCLAIMER,
        screening_result=screening_result,
    )

    # Optionally generate the image prompt after piece creation
    if image_config is not None:
        from opinionforge.core.image_prompt import generate_image_prompt

        generated_image_prompt = generate_image_prompt(
            piece,
            image_config,
            client=client,
            settings=settings,
        )
        piece = piece.model_copy(
            update={
                "image_prompt": generated_image_prompt,
                "image_platform": image_config.platform,
            }
        )

    # Best-effort save to storage (non-blocking, never fails the generation)
    _save_piece_to_storage(piece, provider_name)

    return piece
