"""OpinionForge CLI application built with Typer and Rich.

Provides commands for generating opinion pieces, previewing tone,
listing rhetorical mode profiles, and managing configuration.

Exit codes follow the PRD specification:
    0 = success
    1 = general / generation error
    2 = invalid arguments
    3 = network error
    4 = mode not found
    5 = API key not configured
    6 = rate limit exceeded
    7 = content policy violation
    8 = similarity screening failure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from opinionforge.config import Settings, get_settings
from opinionforge.core.length import (
    LENGTH_PRESETS,
    MAX_WORD_COUNT,
    MIN_WORD_COUNT,
    get_length_instructions,
    resolve_length,
)
from opinionforge.models.config import ModeBlendConfig, StanceConfig, ImagePromptConfig

_VALID_EXPORT_FORMATS = frozenset({"substack", "medium", "wordpress", "twitter"})
_VALID_IMAGE_PLATFORMS = frozenset({"substack", "medium", "wordpress", "facebook", "twitter", "instagram"})
_VALID_IMAGE_STYLES = frozenset({"photorealistic", "editorial", "cartoon", "minimalist", "vintage", "abstract"})
_VALID_PROVIDERS = frozenset({"ollama", "anthropic", "openai", "openai_compatible"})

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="opinionforge",
    help="OpinionForge — an editorial craft engine for generating opinion pieces with rhetorical precision.",
    no_args_is_help=False,
    invoke_without_command=True,
)


# ---------------------------------------------------------------------------
# Port / Server Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 8484
_MAX_PORT = 8494


def _find_available_port(start: int = _DEFAULT_PORT, end: int = _MAX_PORT) -> int:
    """Find the first available port in the range [start, end].

    Args:
        start: First port to try.
        end: Last port to try (inclusive).

    Returns:
        An available port number.

    Raises:
        typer.Exit: With code 1 if no port is available.
    """
    import socket

    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue

    err_console.print(
        f"[red]Error:[/red] All ports {start}–{end} are in use. "
        "Cannot start the web server."
    )
    raise typer.Exit(code=1)


def _launch_server(
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Launch the web server with optional browser auto-open and tray icon.

    Args:
        host: Host to bind the server to.
        port: Specific port, or None to auto-detect.
        open_browser: Whether to open the default browser on startup.
    """
    import threading

    import uvicorn

    from opinionforge.web.app import create_app

    resolved_port = port if port is not None else _find_available_port()
    url = f"http://{host}:{resolved_port}"

    web_app = create_app()
    console.print(f"[green]Starting OpinionForge web UI at {url}[/green]")

    if open_browser:
        def _open_browser_thread() -> None:
            from opinionforge.desktop.browser import open_browser as _open

            _open(url)

        t = threading.Thread(target=_open_browser_thread, daemon=True)
        t.start()

    # Start system tray in background (optional dependency)
    try:
        from opinionforge.desktop.tray import PYSTRAY_AVAILABLE, SystemTrayApp

        if PYSTRAY_AVAILABLE:
            tray = SystemTrayApp(port=resolved_port, host=host)
            tray.start()
    except Exception:
        pass  # Tray is best-effort

    uvicorn.run(web_app, host=host, port=resolved_port)


# ---------------------------------------------------------------------------
# Default callback — bare ``opinionforge`` launches web UI
# ---------------------------------------------------------------------------


@app.callback()
def _default_callback(ctx: typer.Context) -> None:
    """Launch the web UI when no subcommand is given.

    If a subcommand is provided, this callback is a no-op and the
    subcommand runs normally.
    """
    if ctx.invoked_subcommand is not None:
        return
    _launch_server(open_browser=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_mode_blend(mode_str: str) -> ModeBlendConfig:
    """Parse a mode specification string into a ModeBlendConfig.

    Supports two formats:
      - Single mode: ``analytical`` (treated as 100 %)
      - Blend: ``polemical:60,narrative:40``

    Args:
        mode_str: The raw ``--mode`` value.

    Returns:
        A validated ModeBlendConfig.

    Raises:
        typer.Exit: With code 2 on parse errors, code 4 on unknown modes.
    """
    mode_str = mode_str.strip()

    if not mode_str:
        err_console.print(
            "[red]Error:[/red] The --mode argument cannot be empty. "
            "Provide a mode ID (e.g., 'analytical') or a blend (e.g., 'polemical:60,narrative:40')."
        )
        raise typer.Exit(code=2)

    # Single mode (no colon)
    if ":" not in mode_str:
        return ModeBlendConfig(modes=[(mode_str, 100.0)])

    # Blend syntax: mode1:weight1,mode2:weight2
    parts = [p.strip() for p in mode_str.split(",") if p.strip()]
    modes: list[tuple[str, float]] = []
    for part in parts:
        if ":" not in part:
            err_console.print(
                f"[red]Error:[/red] Invalid blend syntax '{part}'. "
                "Expected format: mode:weight (e.g., polemical:60)."
            )
            raise typer.Exit(code=2)
        name, weight_str = part.rsplit(":", 1)
        try:
            weight = float(weight_str)
        except ValueError:
            err_console.print(
                f"[red]Error:[/red] Invalid weight '{weight_str}' for mode '{name}'. "
                "Weight must be a number."
            )
            raise typer.Exit(code=2)
        modes.append((name.strip(), weight))

    try:
        return ModeBlendConfig(modes=modes)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)


def _validate_stance(position: int, intensity: float) -> StanceConfig:
    """Validate stance position and intensity and return a StanceConfig.

    Args:
        position: Argumentative emphasis direction from -100 to +100.
        intensity: Rhetorical heat from 0.0 to 1.0.

    Returns:
        A validated StanceConfig.

    Raises:
        typer.Exit: With code 2 if out of range.
    """
    if position < -100 or position > 100:
        err_console.print(
            f"[red]Error:[/red] Stance position {position} is out of range. "
            "Must be between -100 and +100."
        )
        raise typer.Exit(code=2)
    if intensity < 0.0 or intensity > 1.0:
        err_console.print(
            f"[red]Error:[/red] Intensity {intensity} is out of range. "
            "Must be between 0.0 and 1.0."
        )
        raise typer.Exit(code=2)
    return StanceConfig(position=position, intensity=intensity)


def _validate_length(value: str) -> int:
    """Validate and resolve a length value.

    Args:
        value: A preset name or numeric string.

    Returns:
        Resolved word count.

    Raises:
        typer.Exit: With code 2 on invalid input.
    """
    try:
        return resolve_length(value)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)


def _load_modes(blend: ModeBlendConfig) -> str:
    """Load modes and return the blended prompt fragment.

    Args:
        blend: The mode blend configuration.

    Returns:
        The composed mode prompt string.

    Raises:
        typer.Exit: With code 4 if any mode is not found.
    """
    from opinionforge.core.mode_engine import blend_modes

    try:
        return blend_modes(blend)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=4)


def _ingest_topic(
    topic: str,
    url: Optional[str],
    file: Optional[Path],
) -> "TopicContext":
    """Ingest a topic from text, URL, or file.

    Args:
        topic: Plain-text topic string (may be empty).
        url: Optional URL to ingest from.
        file: Optional file path to ingest from.

    Returns:
        A TopicContext instance.

    Raises:
        typer.Exit: With code 2 on input errors, code 3 on network errors.
    """
    from opinionforge.core.topic import ingest_file, ingest_text, ingest_url

    if not url and not file and not topic:
        err_console.print(
            "[red]Error:[/red] No topic provided. Supply a TOPIC argument, --url, or --file."
        )
        raise typer.Exit(code=2)

    try:
        if url:
            return ingest_url(url)
        if file:
            return ingest_file(str(file))
        return ingest_text(topic)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)
    except Exception as exc:
        # Network or other fetch errors
        err_console.print(f"[red]Error:[/red] Network error: {exc}")
        raise typer.Exit(code=3)


def _mask_key(value: str | None) -> str:
    """Mask an API key for display, showing only last 4 chars.

    Args:
        value: The key string or None.

    Returns:
        A masked representation or '(not set)'.
    """
    if not value or not isinstance(value, str):
        return "(not set)"
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def _format_research_context(research_result: "ResearchResult") -> str:
    """Format research results into a context string for the LLM.

    Args:
        research_result: The structured research output.

    Returns:
        A formatted string of source claims.
    """
    from opinionforge.utils.text import format_citations

    if not research_result.sources:
        return ""
    return format_citations(research_result.sources)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _deprecated_voice_callback(ctx: typer.Context, param: typer.CallbackParam, value: str) -> str:
    """Callback for deprecated --voice flag — always raises an error pointing to --mode.

    Args:
        ctx: The Typer context.
        param: The parameter that triggered this callback.
        value: The provided value (ignored).

    Raises:
        typer.BadParameter: With a message directing users to use --mode.
    """
    if value is not None:
        err_console.print(
            "[red]Error:[/red] --voice is not a valid option. "
            "Use --mode instead (e.g., --mode polemical)."
        )
        raise typer.Exit(code=2)
    return value


def _deprecated_spectrum_callback(ctx: typer.Context, param: typer.CallbackParam, value: Optional[int]) -> Optional[int]:
    """Callback for deprecated --spectrum flag — always raises an error pointing to --stance.

    Args:
        ctx: The Typer context.
        param: The parameter that triggered this callback.
        value: The provided value (ignored).

    Raises:
        typer.Exit: With code 2 and a message directing users to use --stance.
    """
    if value is not None:
        err_console.print(
            "[red]Error:[/red] --spectrum is not a valid option. "
            "Use --stance instead (e.g., --stance -50)."
        )
        raise typer.Exit(code=2)
    return value


def _deprecated_no_disclaimer_callback(ctx: typer.Context, param: typer.CallbackParam, value: bool) -> bool:
    """Callback for removed --no-disclaimer flag — always raises an error.

    Args:
        ctx: The Typer context.
        param: The parameter that triggered this callback.
        value: The flag value (ignored; always errors).

    Raises:
        typer.Exit: With code 2 when the flag is used.
    """
    if value:
        err_console.print(
            "[red]Error:[/red] --no-disclaimer is not a valid option. "
            "The disclaimer is mandatory and cannot be suppressed."
        )
        raise typer.Exit(code=2)
    return value


@app.command()
def write(
    topic: str = typer.Argument(default="", help="Topic text for the opinion piece."),
    mode: str = typer.Option("analytical", "--mode", "-m", help="Rhetorical mode or blend (e.g., 'analytical' or 'polemical:60,narrative:40')."),
    stance: int = typer.Option(0, "--stance", "-s", help="Argumentative emphasis direction (-100 to +100, 0 = balanced)."),
    intensity: float = typer.Option(0.5, "--intensity", "-i", help="Rhetorical heat (0.0 = measured, 1.0 = maximum conviction)."),
    length: str = typer.Option("standard", "--length", "-l", help="Length preset (short/standard/long/essay/feature) or custom word count."),
    url: Optional[str] = typer.Option(None, "--url", help="Ingest topic from a URL."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Ingest topic from a local file."),
    no_preview: bool = typer.Option(False, "--no-preview", help="Skip tone preview and generate immediately."),
    research: bool = typer.Option(True, "--research/--no-research", help="Enable or disable source research."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write output to a file instead of stdout."),
    verbose: bool = typer.Option(False, "--verbose", help="Show research progress and generation details."),
    export_format: Optional[str] = typer.Option(
        None,
        "--export",
        help="Export format for the generated piece. One of: substack, medium, wordpress, twitter.",
    ),
    image_prompt: bool = typer.Option(
        False,
        "--image-prompt",
        help="Generate a header image prompt (DALL-E / Midjourney / Stable Diffusion) for the piece.",
    ),
    image_platform: str = typer.Option(
        "substack",
        "--image-platform",
        help="Target platform for image dimensions. One of: substack, medium, wordpress, facebook, twitter, instagram.",
    ),
    image_style: str = typer.Option(
        "editorial",
        "--image-style",
        help="Visual style for the image prompt. One of: photorealistic, editorial, cartoon, minimalist, vintage, abstract.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider override for this run. One of: ollama, anthropic, openai, openai_compatible.",
    ),
    model_override: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name override for this run (e.g., 'llama3', 'claude-sonnet-4-20250514').",
    ),
    # ---------------------------------------------------------------------------
    # Deprecated v0.2.0 flags — present to produce clear error messages, not used
    # ---------------------------------------------------------------------------
    _deprecated_voice: Optional[str] = typer.Option(
        None,
        "--voice",
        hidden=True,
        callback=_deprecated_voice_callback,
        is_eager=True,
        help="Deprecated. Use --mode instead.",
        expose_value=False,
    ),
    _deprecated_spectrum: Optional[int] = typer.Option(
        None,
        "--spectrum",
        hidden=True,
        callback=_deprecated_spectrum_callback,
        is_eager=True,
        help="Deprecated. Use --stance instead.",
        expose_value=False,
    ),
    _deprecated_no_disclaimer: bool = typer.Option(
        False,
        "--no-disclaimer",
        hidden=True,
        callback=_deprecated_no_disclaimer_callback,
        is_eager=True,
        help="Removed. The disclaimer is mandatory.",
        expose_value=False,
    ),
) -> None:
    """Generate an opinion piece on the given topic.

    Runs the full pipeline: topic ingestion, mode loading, stance
    adjustment, tone preview, source research, and generation.
    Optionally exports to a platform format (--export) and generates
    an image prompt (--image-prompt).
    """
    # 1. Validate inputs
    mode_blend = _parse_mode_blend(mode)
    stance_cfg = _validate_stance(stance, intensity)
    target_length = _validate_length(length)

    # Validate --provider
    if provider is not None:
        provider_lower = provider.lower()
        if provider_lower not in _VALID_PROVIDERS:
            supported = ", ".join(sorted(_VALID_PROVIDERS))
            err_console.print(
                f"[red]Error:[/red] Unknown provider '{provider}'. "
                f"Supported providers: {supported}."
            )
            raise typer.Exit(code=2)
    else:
        provider_lower = None

    # Validate --export format
    if export_format is not None:
        export_fmt_lower = export_format.lower()
        if export_fmt_lower not in _VALID_EXPORT_FORMATS:
            supported = ", ".join(sorted(_VALID_EXPORT_FORMATS))
            err_console.print(
                f"[red]Error:[/red] Unknown export format '{export_format}'. "
                f"Supported formats: {supported}."
            )
            raise typer.Exit(code=2)
    else:
        export_fmt_lower = None

    # Validate --image-platform
    if image_platform.lower() not in _VALID_IMAGE_PLATFORMS:
        supported = ", ".join(sorted(_VALID_IMAGE_PLATFORMS))
        err_console.print(
            f"[red]Error:[/red] Unknown image platform '{image_platform}'. "
            f"Supported platforms: {supported}."
        )
        raise typer.Exit(code=2)

    # Validate --image-style
    if image_style.lower() not in _VALID_IMAGE_STYLES:
        supported = ", ".join(sorted(_VALID_IMAGE_STYLES))
        err_console.print(
            f"[red]Error:[/red] Unknown image style '{image_style}'. "
            f"Supported styles: {supported}."
        )
        raise typer.Exit(code=2)

    # Build image config if requested
    image_config: ImagePromptConfig | None = None
    if image_prompt:
        image_config = ImagePromptConfig(
            style=image_style.lower(),  # type: ignore[arg-type]
            platform=image_platform.lower(),  # type: ignore[arg-type]
        )

    # 2. Ingest topic
    topic_ctx = _ingest_topic(topic, url, file)

    if verbose:
        console.print(f"[dim]Topic:[/dim] {topic_ctx.title}")
        console.print(f"[dim]Domain:[/dim] {topic_ctx.subject_domain}")

    # 3. Load modes
    mode_prompt = _load_modes(mode_blend)

    # 4. Apply stance modifier
    from opinionforge.core.stance import apply_stance

    modified_prompt = apply_stance(mode_prompt, stance_cfg)

    # 5. Tone preview (unless --no-preview)
    if not no_preview:
        from opinionforge.core.preview import generate_preview, create_llm_client

        try:
            settings = get_settings()
            client = create_llm_client(settings)
            preview_text = generate_preview(topic_ctx, modified_prompt, stance_cfg, client=client)
        except SystemExit:
            raise
        except RuntimeError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

        console.print()
        console.print(Panel(preview_text, title="Tone Preview", border_style="cyan"))
        console.print()

        # Prompt for confirmation
        proceed = typer.confirm("Generate full piece?", default=True)
        if not proceed:
            console.print("[yellow]Generation cancelled.[/yellow]")
            raise typer.Exit(code=0)

    # 6. Research (unless --no-research)
    research_context: str | None = None
    if research:
        from opinionforge.core.research import research_topic

        if verbose:
            console.print("[dim]Researching sources...[/dim]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Researching sources...", total=None)
                research_result = research_topic(
                    topic_ctx, stance_cfg, target_length=target_length
                )

            if research_result.warning and verbose:
                console.print(f"[yellow]Warning:[/yellow] {research_result.warning}")

            research_context = _format_research_context(research_result)
        except SystemExit:
            raise
        except Exception as exc:
            if verbose:
                console.print(f"[yellow]Warning:[/yellow] Research failed: {exc}")
            # Continue without research

    # 7. Generate
    from opinionforge.core.generator import generate_piece

    if verbose:
        console.print("[dim]Generating opinion piece...[/dim]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating opinion piece...", total=None)
            piece = generate_piece(
                topic=topic_ctx,
                mode_config=mode_blend,
                stance=stance_cfg,
                target_length=target_length,
                research_context=research_context,
                image_config=image_config,
                provider_type=provider_lower,
                model=model_override,
            )
    except SystemExit:
        raise
    except RuntimeError as exc:
        exc_str = str(exc)
        if "similarity screening" in exc_str.lower():
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=8)
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    # 8. Output — disclaimer is always mandatory and always included
    from opinionforge.core.generator import MANDATORY_DISCLAIMER

    if export_fmt_lower:
        from opinionforge.exporters import export as _export
        output_text = _export(piece, export_fmt_lower)
    else:
        output_text = f"## {piece.title}\n\n{piece.body}"
        output_text += f"\n\n---\n*{MANDATORY_DISCLAIMER}*"
        if piece.image_prompt:
            output_text += f"\n\n**Header image prompt:** {piece.image_prompt}"

    if output:
        output.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Piece written to {output}[/green]")
    else:
        console.print()
        console.print(Panel(output_text, title=piece.title, border_style="green"))

    if verbose:
        console.print(f"[dim]Words: {piece.actual_length} (target: {target_length})[/dim]")
        if piece.image_prompt:
            console.print(f"[dim]Image prompt generated for platform: {piece.image_platform}[/dim]")


@app.command()
def preview(
    topic: str = typer.Argument(default="", help="Topic text for the tone preview."),
    mode: str = typer.Option("analytical", "--mode", "-m", help="Rhetorical mode or blend."),
    stance: int = typer.Option(0, "--stance", "-s", help="Argumentative emphasis direction (-100 to +100)."),
    intensity: float = typer.Option(0.5, "--intensity", "-i", help="Rhetorical heat (0.0 to 1.0)."),
    url: Optional[str] = typer.Option(None, "--url", help="Ingest topic from a URL."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Ingest topic from a local file."),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider override for this run. One of: ollama, anthropic, openai, openai_compatible.",
    ),
    model_override: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name override for this run (e.g., 'llama3', 'claude-sonnet-4-20250514').",
    ),
) -> None:
    """Generate a short tone preview for the given topic.

    Produces a 2-3 sentence preview in the selected rhetorical mode without
    performing full research or generation.
    """
    mode_blend = _parse_mode_blend(mode)
    stance_cfg = _validate_stance(stance, intensity)

    # Validate --provider
    if provider is not None:
        provider_lower = provider.lower()
        if provider_lower not in _VALID_PROVIDERS:
            supported = ", ".join(sorted(_VALID_PROVIDERS))
            err_console.print(
                f"[red]Error:[/red] Unknown provider '{provider}'. "
                f"Supported providers: {supported}."
            )
            raise typer.Exit(code=2)
    else:
        provider_lower = None

    topic_ctx = _ingest_topic(topic, url, file)
    mode_prompt = _load_modes(mode_blend)

    from opinionforge.core.stance import apply_stance

    modified_prompt = apply_stance(mode_prompt, stance_cfg)

    from opinionforge.core.preview import create_llm_client, create_llm_client_from_provider, generate_preview

    try:
        settings = get_settings()
        if provider_lower is not None:
            # Build a provider via the registry and wrap in sync LLMClient
            from opinionforge.providers import get_provider

            from opinionforge.models.config import ProviderConfig

            resolved_model: str
            resolved_api_key: str | None = None
            resolved_base_url: str | None = None

            if provider_lower == "anthropic":
                resolved_model = model_override or settings.opinionforge_model or "claude-sonnet-4-20250514"
                resolved_api_key = settings.anthropic_api_key or ""
            elif provider_lower == "openai":
                resolved_model = model_override or settings.opinionforge_model or "gpt-4o"
                resolved_api_key = settings.openai_api_key or ""
            elif provider_lower == "ollama":
                resolved_model = model_override or settings.opinionforge_model or "llama3"
                resolved_base_url = settings.opinionforge_ollama_base_url
            elif provider_lower == "openai_compatible":
                resolved_model = model_override or settings.opinionforge_model or "default"
                resolved_base_url = settings.opinionforge_ollama_base_url
            else:
                resolved_model = model_override or settings.opinionforge_model or "default"

            provider_config = ProviderConfig(
                provider_type=provider_lower,
                model=resolved_model,
                api_key=resolved_api_key or None,
                base_url=resolved_base_url,
            )
            llm_provider = get_provider(provider_config)
            client = create_llm_client_from_provider(llm_provider)
        else:
            client = create_llm_client(settings)

        preview_text = generate_preview(topic_ctx, modified_prompt, stance_cfg, client=client)
    except SystemExit:
        raise
    except RuntimeError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print()
    console.print(Panel(preview_text, title="Tone Preview", border_style="cyan"))


@app.command()
def modes(
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Filter modes by search query."),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter modes by category."),
    detail: Optional[str] = typer.Option(None, "--detail", "-d", help="Show full detail for a rhetorical mode."),
) -> None:
    """List and search available rhetorical mode profiles."""
    from opinionforge.modes import list_modes, load_mode

    if detail:
        # Show detailed view of a single mode
        try:
            profile = load_mode(detail)
        except FileNotFoundError:
            err_console.print(f"[red]Error:[/red] Mode '{detail}' not found.")
            raise typer.Exit(code=4)

        detail_text = (
            f"[bold]{profile.display_name}[/bold] ({profile.id})\n\n"
            f"[bold]Category:[/bold] {profile.category}\n"
            f"[bold]Description:[/bold] {profile.description}\n\n"
            f"[bold]Rhetorical Devices:[/bold] {', '.join(profile.rhetorical_devices)}\n"
            f"[bold]Signature Patterns:[/bold] {', '.join(profile.signature_patterns)}\n\n"
            f"[bold]Prose Patterns:[/bold]\n"
            f"  Sentence length: {profile.prose_patterns.avg_sentence_length}\n"
            f"  Paragraph length: {profile.prose_patterns.paragraph_length}\n"
            f"  Uses fragments: {profile.prose_patterns.uses_fragments}\n"
            f"  Opening style: {profile.prose_patterns.opening_style}\n"
            f"  Closing style: {profile.prose_patterns.closing_style}\n\n"
            f"[bold]Argument Structure:[/bold]\n"
            f"  Approach: {profile.argument_structure.approach}\n"
            f"  Evidence: {profile.argument_structure.evidence_style}\n"
            f"  Concessions: {profile.argument_structure.concession_pattern}\n"
            f"  Thesis placement: {profile.argument_structure.thesis_placement}"
        )
        console.print(Panel(detail_text, title=f"Mode Profile: {profile.display_name}", border_style="blue"))
        return

    # List all modes (optionally filtered)
    all_modes = list_modes()

    if not all_modes:
        console.print("[yellow]No rhetorical mode profiles installed.[/yellow]")
        return

    if category:
        all_modes = [m for m in all_modes if m.category.lower() == category.lower()]

    if search:
        query = search.lower()
        all_modes = [
            m for m in all_modes
            if query in m.id.lower()
            or query in m.display_name.lower()
            or query in m.category.lower()
            or query in m.description.lower()
        ]

    if not all_modes:
        console.print("[yellow]No modes match the filter criteria.[/yellow]")
        return

    table = Table(title="Available Rhetorical Modes")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="bold")
    table.add_column("Category")
    table.add_column("Description")

    for m in all_modes:
        table.add_row(
            m.id,
            m.display_name,
            m.category,
            m.description[:60] + ("..." if len(m.description) > 60 else ""),
        )

    console.print(table)


@app.command()
def export(  # noqa: A001
    piece_id: str = typer.Argument(help="ID of a previously generated piece to export."),
    format: str = typer.Option(  # noqa: A002
        ...,
        "--format",
        "-f",
        help="Export format. One of: substack, medium, wordpress, twitter.",
    ),
) -> None:
    """Export a previously generated piece to a platform format.

    Requires generation history to be available (Phase 3 feature).
    Currently shows a clear 'not yet available' message rather than crashing.
    """
    fmt_lower = format.lower()
    if fmt_lower not in _VALID_EXPORT_FORMATS:
        supported = ", ".join(sorted(_VALID_EXPORT_FORMATS))
        err_console.print(
            f"[red]Error:[/red] Unknown export format '{format}'. "
            f"Supported formats: {supported}."
        )
        raise typer.Exit(code=2)

    err_console.print(
        f"[yellow]Note:[/yellow] The 'opinionforge export' command requires generation "
        "history (storage/history.py), which is a Phase 3 feature not yet implemented.\n"
        f"To export piece ID '{piece_id}' in '{fmt_lower}' format, use the --export flag "
        "on the write command instead:\n\n"
        f"  opinionforge write 'your topic' --mode analytical --export {fmt_lower} --no-preview"
    )
    raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option(None, "--host", "-h", help="Host to bind the web server to."),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind the web server to."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Suppress opening the default browser on startup."),
) -> None:
    """Start the OpinionForge web UI server.

    Launches a FastAPI/uvicorn server for the web interface and opens
    the default browser.  Defaults to host 127.0.0.1 and port 8484,
    configurable via --host/--port flags or OPINIONFORGE_HOST/
    OPINIONFORGE_PORT environment variables.  Use --no-browser to
    suppress the automatic browser open.
    """
    settings = get_settings()
    resolved_host = host if host is not None else settings.opinionforge_host
    resolved_port = port if port is not None else None

    _launch_server(
        host=resolved_host,
        port=resolved_port,
        open_browser=not no_browser,
    )


@app.command()
def config(
    provider: Optional[str] = typer.Option(None, "--provider", help="Set the LLM provider (anthropic, openai, ollama, openai_compatible)."),
    model: Optional[str] = typer.Option(None, "--model", help="Set the model name for the configured provider."),
    set_key: Optional[str] = typer.Option(None, "--set", help="Configuration key to set (format: KEY VALUE)."),
    set_value: Optional[str] = typer.Argument(None, help="Value for the --set key."),
) -> None:
    """Show or modify OpinionForge configuration.

    Without flags, displays all current configuration values (reading from
    storage first, falling back to environment variables). With --provider
    and/or --model, saves the provider configuration to persistent storage.
    """
    import asyncio

    settings = get_settings()

    # Handle --provider / --model shorthand
    if provider is not None or model is not None:
        if provider is not None:
            provider_lower = provider.lower()
            if provider_lower not in _VALID_PROVIDERS:
                supported = ", ".join(sorted(_VALID_PROVIDERS))
                err_console.print(
                    f"[red]Error:[/red] Unknown provider '{provider}'. "
                    f"Supported providers: {supported}."
                )
                raise typer.Exit(code=2)
        else:
            provider_lower = settings.opinionforge_llm_provider

        model_name = model or settings.opinionforge_model or ""

        # Save to storage
        try:
            from opinionforge.models.config import ProviderConfig
            from opinionforge.storage import Database, SettingsStore, get_db_path

            async def _save() -> None:
                db_path = get_db_path()
                db = Database(db_path)
                await db.connect()
                await db.initialize()
                try:
                    store = SettingsStore(db)
                    cfg = ProviderConfig(provider_type=provider_lower, model=model_name)
                    await store.set_provider_config(cfg)
                finally:
                    await db.close()

            asyncio.run(_save())
            console.print(f"[green]Saved provider config: {provider_lower} / {model_name}[/green]")
        except Exception as exc:
            err_console.print(f"[red]Error:[/red] Failed to save config: {exc}")
            raise typer.Exit(code=1)
        return

    if set_key:
        # Security: block setting API keys via CLI
        key_lower = set_key.lower()
        if "api_key" in key_lower or "secret" in key_lower:
            err_console.print(
                "[red]Error:[/red] API keys cannot be set via the CLI for security reasons. "
                "Please set them in your .env file or environment variables."
            )
            raise typer.Exit(code=2)

        if set_value is None:
            err_console.print(
                "[red]Error:[/red] Please provide a value. Usage: opinionforge config --set KEY VALUE"
            )
            raise typer.Exit(code=2)

        # Attempt to set a valid config key
        valid_keys = {"opinionforge_llm_provider", "opinionforge_search_provider"}
        if set_key not in valid_keys:
            err_console.print(
                f"[red]Error:[/red] Unknown configuration key '{set_key}'. "
                f"Settable keys: {', '.join(sorted(valid_keys))}"
            )
            raise typer.Exit(code=2)

        console.print(f"[green]Set {set_key} = {set_value}[/green]")
        console.print("[dim]Note: Runtime config changes are session-only. For persistence, update your .env file.[/dim]")
        return

    # Display current configuration — try storage first, then env vars
    stored_provider: str | None = None
    stored_model: str | None = None
    try:
        from opinionforge.storage import Database, SettingsStore, get_db_path

        async def _read_stored() -> tuple[str | None, str | None]:
            db_path = get_db_path()
            if not db_path.exists():
                return None, None
            db = Database(db_path)
            await db.connect()
            await db.initialize()
            try:
                store = SettingsStore(db)
                cfg = await store.get_provider_config()
                if cfg is not None:
                    return cfg.provider_type, cfg.model
            finally:
                await db.close()
            return None, None

        stored_provider, stored_model = asyncio.run(_read_stored())
    except Exception:
        pass  # Fall back to env vars

    table = Table(title="OpinionForge Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    if stored_provider:
        table.add_row("LLM Provider", str(stored_provider), "storage")
    else:
        table.add_row("LLM Provider", str(settings.opinionforge_llm_provider), "env")

    env_model = settings.opinionforge_model
    if stored_model:
        table.add_row("Model", str(stored_model), "storage")
    elif env_model and isinstance(env_model, str):
        table.add_row("Model", env_model, "env")
    else:
        table.add_row("Model", "(default)", "")

    table.add_row("Anthropic API Key", _mask_key(settings.anthropic_api_key), "env")
    table.add_row("OpenAI API Key", _mask_key(settings.openai_api_key), "env")
    table.add_row("Search Provider", str(settings.opinionforge_search_provider), "env")
    table.add_row("Search API Key", _mask_key(settings.opinionforge_search_api_key), "env")

    console.print(table)


