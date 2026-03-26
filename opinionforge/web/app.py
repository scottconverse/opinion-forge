"""FastAPI application for OpinionForge web UI.

Provides routes for generation, preview, mode browsing, export, onboarding,
and about.  All generation uses the existing engine — no rewrites.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.core.preview import LLMClient
from opinionforge.models.config import (
    ModeBlendConfig,
    ProviderConfig,
    SearchConfig,
    StanceConfig,
    UserPreferences,
)
from opinionforge.models.topic import TopicContext
from opinionforge.providers.base import ProviderError
from opinionforge.providers.registry import ProviderRegistry
from opinionforge.storage.database import Database
from opinionforge.storage.exports import ExportStore
from opinionforge.storage.pieces import PieceStore
from opinionforge.storage.settings import SettingsStore
from opinionforge.web.sse import generation_event_stream

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

# Category display order for the mode grid
_CATEGORY_ORDER = ["confrontational", "investigative", "deliberative", "literary"]


def _group_modes_by_category(modes: list) -> OrderedDict:
    """Group ModeProfile instances by category in display order.

    Args:
        modes: List of ModeProfile objects.

    Returns:
        An OrderedDict mapping category name to list of ModeProfile objects.
    """
    groups: OrderedDict = OrderedDict()
    for cat in _CATEGORY_ORDER:
        groups[cat] = []
    for m in modes:
        cat = m.category
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(m)
    return groups


def _get_provider_name(app_state: object) -> str | None:
    """Extract the provider name from app state if a provider is configured.

    Args:
        app_state: The FastAPI app.state object.

    Returns:
        A provider/model name string, or None if no provider is set.
    """
    provider = getattr(app_state, "llm_provider", None)
    if provider is not None:
        try:
            return provider.model_name()
        except Exception:
            return "unknown"

    client = getattr(app_state, "llm_client", None)
    if client is not None:
        return None  # Client is set, provider name will be in SSE done event

    return None


def _connection_fix_instruction(provider_type: str, message: str) -> str:
    """Return an actionable fix instruction for a connection error.

    Args:
        provider_type: The provider that failed.
        message: The error message text.

    Returns:
        A user-friendly fix instruction string.
    """
    msg_lower = message.lower()
    if provider_type == "ollama":
        if "not running" in msg_lower or "connect" in msg_lower:
            return "Start Ollama with: ollama serve"
        if "model" in msg_lower:
            return "Pull the model first with: ollama pull <model-name>"
        return "Check that Ollama is running and the model is available."
    if "auth" in msg_lower or "401" in msg_lower or "api key" in msg_lower:
        return "Check that your API key is correct and has not expired."
    if "429" in msg_lower or "rate" in msg_lower:
        return "Rate limit exceeded. Wait a moment and try again."
    if "timeout" in msg_lower:
        return "The provider did not respond in time. Check your internet connection."
    if "connect" in msg_lower or "network" in msg_lower:
        return "Could not reach the provider. Check your internet connection and URL."
    return "Check your credentials and try again."


def _generation_fix_instruction(message: str) -> str:
    """Return an actionable fix instruction for a generation error.

    Args:
        message: The error message text.

    Returns:
        A user-friendly fix instruction string.
    """
    msg_lower = message.lower()
    if "auth" in msg_lower or "api key" in msg_lower:
        return "Go back to Step 2 and verify your API key."
    if "model" in msg_lower and "not found" in msg_lower:
        return "The selected model is not available. Go back to Step 2 and choose a different model."
    if "timeout" in msg_lower:
        return "Generation timed out. Try again or switch to a faster model."
    if "connect" in msg_lower:
        return "Could not reach the provider. Check that the service is running."
    return "Check your provider configuration in Step 2 and try again."


def create_app(
    *,
    client: LLMClient | None = None,
    provider: object | None = None,
    db_path: str | Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        client: Optional LLM client for dependency injection (used in tests).
        provider: Optional LLMProvider instance. When provided, generation
            routes use this provider. When neither client nor provider is
            given, generation routes redirect to /setup.
        db_path: Optional database path. Use ``:memory:`` for testing.
            Defaults to the platform-specific path.

    Returns:
        A fully-configured FastAPI instance.
    """
    app = FastAPI(title="OpinionForge", version="2.0.0")

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Jinja2 templates
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Store injected client and provider on app state
    app.state.llm_client = client
    app.state.llm_provider = provider
    app.state.db_path = db_path

    # If a provider is given but no client, create a sync wrapper
    if provider is not None and client is None:
        from opinionforge.core.preview import create_llm_client_from_provider

        app.state.llm_client = create_llm_client_from_provider(provider)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _base_template(request: Request) -> str:
        """Return the appropriate base template based on HX-Request header.

        HTMX requests receive 'partial_base.html' (no page wrapper).
        Direct browser navigation receives the full 'base.html'.

        Args:
            request: The incoming request.

        Returns:
            The template filename to use as the extends base.
        """
        if request.headers.get("HX-Request"):
            return "partial_base.html"
        return "base.html"

    def _has_provider_configured() -> bool:
        """Check whether a provider or client is configured.

        Returns:
            True if generation is possible, False if /setup is needed.
        """
        return app.state.llm_client is not None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Render the home page with the generation form.

        Fetches recent pieces (last 5) and recent topics (last 10) from
        storage for the sidebar and topic suggestions datalist.
        If no LLM provider is configured, redirects to onboarding.
        """
        from opinionforge.modes import list_modes

        # Redirect to onboarding if onboarding_completed flag is False in settings.
        # When a provider/client is explicitly injected (programmatic/test mode),
        # skip the DB check and treat onboarding as complete.
        if app.state.llm_client is None and app.state.llm_provider is None:
            try:
                db = await _get_db()
                try:
                    ss = SettingsStore(db)
                    prefs = await ss.get_user_preferences()
                    if not prefs.onboarding_completed:
                        return RedirectResponse(url="/setup", status_code=303)
                finally:
                    await db.close()
            except Exception:
                return RedirectResponse(url="/setup", status_code=303)

        all_modes = list_modes()
        categories = _group_modes_by_category(all_modes)
        selected_mode = request.query_params.get("mode", "")

        # Include provider name in template context
        provider_name = _get_provider_name(app.state)

        # Fetch recent pieces and topics from storage (best-effort)
        recent_pieces: list[dict] = []
        recent_topics: list[str] = []
        try:
            db = await _get_db()
            try:
                ps = PieceStore(db)
                recent_pieces = await ps.list_all(limit=5)
                # Extract last 10 unique topics for the datalist
                all_recent = await ps.list_all(limit=10)
                seen: set[str] = set()
                for p in all_recent:
                    topic_text = p.get("topic", "")
                    if topic_text and topic_text not in seen:
                        seen.add(topic_text)
                        recent_topics.append(topic_text)
            finally:
                await db.close()
        except Exception:
            logger.debug("Could not fetch recent pieces for home page")

        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "base_template": _base_template(request),
                "categories": categories,
                "modes": all_modes,
                "selected_mode": selected_mode,
                "stance_default": 0,
                "intensity_default": 0.5,
                "length_default": "standard",
                "provider_name": provider_name,
                "recent_pieces": recent_pieces,
                "recent_topics": recent_topics,
                "current_page": "home",
            },
        )

    @app.get("/about", response_class=HTMLResponse)
    async def about(request: Request) -> HTMLResponse:
        """Render the about page."""
        return templates.TemplateResponse(
            request, "about.html", {"base_template": _base_template(request), "current_page": "about"}
        )

    # ------------------------------------------------------------------
    # Settings store helper
    # ------------------------------------------------------------------

    async def _get_settings_store() -> SettingsStore:
        """Return a connected SettingsStore, reusing the DB connection.

        Caches the database connection on ``app.state`` so that
        in-memory databases persist across requests within the same
        application lifecycle.

        Returns:
            A :class:`SettingsStore` backed by an initialised database.
        """
        existing_db: Database | None = getattr(app.state, "_settings_db", None)
        if existing_db is not None and existing_db._conn is not None:
            return SettingsStore(existing_db)

        db = Database(app.state.db_path)
        await db.connect()
        await db.initialize()
        app.state._settings_db = db
        return SettingsStore(db)

    # ------------------------------------------------------------------
    # Onboarding routes
    # ------------------------------------------------------------------

    @app.get("/setup", response_class=HTMLResponse)
    async def setup(request: Request) -> HTMLResponse:
        """Render the 5-step onboarding wizard.

        Detects Ollama status and available models on load so the
        template can pre-populate the Ollama card.

        Args:
            request: The incoming request.
        """
        registry = ProviderRegistry()
        ollama_models: list[str] = []
        try:
            models = await registry.list_ollama_models()
            ollama_models = models
        except Exception:
            pass

        return templates.TemplateResponse(
            request,
            "setup.html",
            {
                "base_template": _base_template(request),
                "ollama_models": ollama_models,
            },
        )

    @app.post("/setup/test-connection", response_class=HTMLResponse)
    async def setup_test_connection(
        request: Request,
        provider_type: str = Form(""),
        api_key: str = Form(""),
        base_url: str = Form(""),
        model: str = Form(""),
    ) -> HTMLResponse:
        """Test connectivity to a provider and return an HTMX partial.

        Args:
            request: The incoming request.
            provider_type: Provider backend to test.
            api_key: Optional API key.
            base_url: Optional base URL.
            model: Model identifier.
        """
        registry = ProviderRegistry()

        # Special case: Ollama detection without a model selected
        if provider_type == "ollama" and not model:
            try:
                is_running = await registry.detect_ollama(
                    base_url or "http://localhost:11434"
                )
                if is_running:
                    return templates.TemplateResponse(
                        request,
                        "partials/connection_result.html",
                        {"success": True, "model_name": "Ollama server"},
                    )
                else:
                    return templates.TemplateResponse(
                        request,
                        "partials/connection_result.html",
                        {
                            "success": False,
                            "error_message": "Ollama is not running.",
                            "fix_instruction": "Start Ollama with: ollama serve",
                        },
                    )
            except Exception as exc:
                return templates.TemplateResponse(
                    request,
                    "partials/connection_result.html",
                    {
                        "success": False,
                        "error_message": f"Could not reach Ollama: {exc}",
                        "fix_instruction": "Start Ollama with: ollama serve",
                    },
                )

        # Build kwargs for the provider constructor
        kwargs: dict[str, object] = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        try:
            provider_inst = registry.create_provider(provider_type, **kwargs)
            success, message = await registry.test_connection(provider_inst)
        except (ValueError, ProviderError) as exc:
            success = False
            message = str(exc)

        if success:
            return templates.TemplateResponse(
                request,
                "partials/connection_result.html",
                {"success": True, "model_name": model or provider_type},
            )

        # Map common errors to actionable fix instructions
        fix = _connection_fix_instruction(provider_type, message)
        return templates.TemplateResponse(
            request,
            "partials/connection_result.html",
            {
                "success": False,
                "error_message": message,
                "fix_instruction": fix,
            },
        )

    @app.post("/setup/save-provider", response_class=HTMLResponse)
    async def setup_save_provider(
        request: Request,
        provider_type: str = Form(""),
        api_key: str = Form(""),
        base_url: str = Form(""),
        model: str = Form(""),
    ) -> HTMLResponse:
        """Save the selected LLM provider configuration to the settings store.

        Args:
            request: The incoming request.
            provider_type: Provider backend.
            api_key: Optional API key.
            base_url: Optional base URL.
            model: Model identifier.
        """
        store = await _get_settings_store()
        config = ProviderConfig(
            provider_type=provider_type,
            model=model,
            api_key=api_key or None,
            base_url=base_url or None,
        )
        await store.set_provider_config(config)
        return HTMLResponse(content="ok", status_code=200)

    @app.post("/setup/save-search", response_class=HTMLResponse)
    async def setup_save_search(
        request: Request,
        provider: str = Form(""),
        api_key: str = Form(""),
    ) -> HTMLResponse:
        """Save search provider configuration.

        When ``provider`` is ``'none'``, no search provider is stored
        (the user chose to skip).

        Args:
            request: The incoming request.
            provider: Search backend identifier.
            api_key: Optional API key.
        """
        store = await _get_settings_store()
        config = SearchConfig(
            provider=provider,
            api_key=api_key or None,
        )
        import json as _json

        await store.set("search_config", _json.dumps(config.model_dump()))
        return HTMLResponse(content="ok", status_code=200)

    @app.post("/setup/test-generate", response_class=HTMLResponse)
    async def setup_test_generate(request: Request) -> HTMLResponse:
        """Run a test generation using the configured provider.

        Uses the fixed sample topic 'The Future of Remote Work' with
        analytical mode, stance 0, and intensity 0.5.

        Args:
            request: The incoming request.
        """
        try:
            # Load provider config from settings
            store = await _get_settings_store()
            config = await store.get_provider_config()
            if config is None:
                return templates.TemplateResponse(
                    request,
                    "partials/test_generate_result.html",
                    {
                        "success": False,
                        "error_message": "No provider configured.",
                        "fix_instruction": (
                            "Go back to Step 2 and configure a provider."
                        ),
                    },
                )

            # Create a provider and run a short generation
            registry = ProviderRegistry()
            gen_kwargs: dict[str, object] = {"model": config.model}
            if config.api_key:
                gen_kwargs["api_key"] = config.api_key
            if config.base_url:
                gen_kwargs["base_url"] = config.base_url
            provider_inst = registry.create_provider(
                config.provider_type, **gen_kwargs
            )

            result = await provider_inst.generate(
                system_prompt=(
                    "You are a skilled opinion writer. Write in an analytical "
                    "rhetorical mode: prioritize data, evidence, and structured "
                    "argumentation. Keep it under 200 words."
                ),
                user_prompt=(
                    "Write a short opinion piece about: The Future of Remote Work"
                ),
                max_tokens=512,
            )

            # Parse title from first markdown heading, or use default
            title = "The Future of Remote Work"
            lines = result.strip().split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    title = stripped.lstrip("#").strip()
                    break

            preview = result[:200]

            return templates.TemplateResponse(
                request,
                "partials/test_generate_result.html",
                {
                    "success": True,
                    "title": title,
                    "preview_text": preview,
                },
            )

        except (ProviderError, Exception) as exc:
            logger.warning("Test generation failed: %s", exc)
            fix = _generation_fix_instruction(str(exc))
            return templates.TemplateResponse(
                request,
                "partials/test_generate_result.html",
                {
                    "success": False,
                    "error_message": str(exc),
                    "fix_instruction": fix,
                },
            )

    @app.post("/setup/complete")
    async def setup_complete(request: Request) -> RedirectResponse:
        """Mark onboarding as completed and redirect to the home page.

        Saves ``onboarding_completed=True`` in user preferences.

        Args:
            request: The incoming request.
        """
        store = await _get_settings_store()
        prefs = await store.get_user_preferences()
        prefs.onboarding_completed = True
        await store.set_user_preferences(prefs)

        # If this is an HTMX request, return a redirect header
        if request.headers.get("HX-Request"):
            response = HTMLResponse(content="", status_code=200)
            response.headers["HX-Redirect"] = "/"
            return response  # type: ignore[return-value]
        return RedirectResponse(url="/", status_code=303)

    @app.get("/modes", response_class=HTMLResponse)
    async def modes_list(request: Request) -> HTMLResponse:
        """List all available rhetorical modes."""
        from opinionforge.modes import list_modes

        all_modes = list_modes()
        categories = _group_modes_by_category(all_modes)
        return templates.TemplateResponse(
            request,
            "modes.html",
            {
                "base_template": _base_template(request),
                "modes": all_modes,
                "categories": categories,
                "current_page": "modes",
            },
        )

    @app.get("/modes/{mode_id}", response_class=HTMLResponse)
    async def mode_detail(request: Request, mode_id: str) -> HTMLResponse:
        """Show details for a single rhetorical mode.

        Args:
            request: The incoming request.
            mode_id: The slug identifier for the mode.

        Raises:
            HTTPException: 404 if the mode is not found.
        """
        from opinionforge.modes import load_mode

        try:
            profile = load_mode(mode_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Mode '{mode_id}' not found.")
        return templates.TemplateResponse(
            request,
            "mode_detail.html",
            {"base_template": _base_template(request), "mode": profile},
        )

    def _parse_generation_params(
        topic: str,
        mode: str,
        stance: int,
        intensity: float,
        length: str,
    ) -> tuple[TopicContext, ModeBlendConfig, StanceConfig, str]:
        """Parse and validate generation parameters, shared by POST and GET routes.

        Args:
            topic: The topic text.
            mode: Rhetorical mode or blend specification.
            stance: Argumentative emphasis direction (-100 to +100).
            intensity: Rhetorical heat (0.0 to 1.0).
            length: Length preset or custom word count.

        Returns:
            A tuple of (TopicContext, ModeBlendConfig, StanceConfig, length).

        Raises:
            HTTPException: 422 if topic is empty or blend syntax is invalid.
        """
        if not topic.strip():
            raise HTTPException(status_code=422, detail="Topic is required.")

        mode_str = mode.strip()
        if ":" not in mode_str:
            mode_config = ModeBlendConfig(modes=[(mode_str, 100.0)])
        else:
            parts = [p.strip() for p in mode_str.split(",") if p.strip()]
            modes_list: list[tuple[str, float]] = []
            for part in parts:
                if ":" not in part:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid blend syntax '{part}'.",
                    )
                name, weight_str = part.rsplit(":", 1)
                try:
                    weight = float(weight_str)
                except ValueError:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid weight '{weight_str}' for mode '{name}'.",
                    )
                modes_list.append((name.strip(), weight))
            mode_config = ModeBlendConfig(modes=modes_list)

        stance_cfg = StanceConfig(position=stance, intensity=intensity)
        topic_ctx = TopicContext(
            title=topic.strip(),
            summary=topic.strip(),
            raw_input=topic.strip(),
            input_type="text",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
        )
        return topic_ctx, mode_config, stance_cfg, length

    def _make_event_source(
        topic_ctx: TopicContext,
        mode_config: ModeBlendConfig,
        stance_cfg: StanceConfig,
        length: str,
    ) -> EventSourceResponse:
        """Build an EventSourceResponse for the generation pipeline.

        Args:
            topic_ctx: The normalised topic context.
            mode_config: Rhetorical mode blend configuration.
            stance_cfg: Stance and intensity controls.
            length: Target word count or preset name.

        Returns:
            An EventSourceResponse streaming generation events.
        """
        injected_client = app.state.llm_client

        async def event_generator():  # type: ignore[return]
            async for event in generation_event_stream(
                topic=topic_ctx,
                mode_config=mode_config,
                stance=stance_cfg,
                target_length=length,
                client=injected_client,
            ):
                yield event

        return EventSourceResponse(event_generator())

    @app.post("/generate")
    async def generate(
        topic: str = Form(""),
        mode: str = Form("analytical"),
        stance: int = Form(0),
        intensity: float = Form(0.5),
        length: str = Form("standard"),
        export_format: Optional[str] = Form(None),
        image_prompt: bool = Form(False),
        image_style: str = Form("editorial"),
        image_platform: str = Form("substack"),
    ):  # type: ignore[return]
        """Generate an opinion piece via SSE streaming (POST, form data).

        If no provider is configured, redirects to /setup.

        Args:
            topic: The topic text.
            mode: Rhetorical mode or blend specification.
            stance: Argumentative emphasis direction (-100 to +100).
            intensity: Rhetorical heat (0.0 to 1.0).
            length: Length preset or custom word count.
            export_format: Optional export format.
            image_prompt: Whether to generate an image prompt.
            image_style: Visual style for the image prompt.
            image_platform: Target platform for image dimensions.

        Raises:
            HTTPException: 422 if topic is empty.
        """
        if not _has_provider_configured():
            return RedirectResponse(url="/setup", status_code=303)

        topic_ctx, mode_config, stance_cfg, length = _parse_generation_params(
            topic, mode, stance, intensity, length,
        )
        return _make_event_source(topic_ctx, mode_config, stance_cfg, length)

    @app.get("/generate/stream")
    async def generate_stream(
        topic: str = Query(""),
        mode: str = Query("analytical"),
        stance: int = Query(0),
        intensity: float = Query(0.5),
        length: str = Query("standard"),
    ):  # type: ignore[return]
        """Generate an opinion piece via SSE streaming (GET, query params).

        This endpoint is used by the HTMX SSE extension (EventSource requires GET).
        If no provider is configured, redirects to /setup.

        Args:
            topic: The topic text.
            mode: Rhetorical mode or blend specification.
            stance: Argumentative emphasis direction (-100 to +100).
            intensity: Rhetorical heat (0.0 to 1.0).
            length: Length preset or custom word count.

        Raises:
            HTTPException: 422 if topic is empty.
        """
        if not _has_provider_configured():
            return RedirectResponse(url="/setup", status_code=303)

        topic_ctx, mode_config, stance_cfg, length = _parse_generation_params(
            topic, mode, stance, intensity, length,
        )
        return _make_event_source(topic_ctx, mode_config, stance_cfg, length)

    @app.get("/preview", response_class=HTMLResponse)
    async def preview_route(
        request: Request,
        topic: str = Query(""),
        mode: str = Query("analytical"),
        stance: int = Query(0),
        intensity: float = Query(0.5),
    ) -> HTMLResponse:
        """Generate a tone preview (HTMX-compatible HTML partial).

        Args:
            request: The incoming request.
            topic: The topic text.
            mode: Rhetorical mode or blend specification.
            stance: Argumentative emphasis direction.
            intensity: Rhetorical heat.

        Raises:
            HTTPException: 422 if topic is empty.
        """
        if not topic.strip():
            raise HTTPException(status_code=422, detail="Topic is required.")

        from opinionforge.core.mode_engine import blend_modes
        from opinionforge.core.preview import generate_preview
        from opinionforge.core.stance import apply_stance

        # Parse mode
        mode_str = mode.strip()
        if ":" not in mode_str:
            mode_config = ModeBlendConfig(modes=[(mode_str, 100.0)])
        else:
            parts = [p.strip() for p in mode_str.split(",") if p.strip()]
            modes_list: list[tuple[str, float]] = []
            for part in parts:
                name, weight_str = part.rsplit(":", 1)
                modes_list.append((name.strip(), float(weight_str)))
            mode_config = ModeBlendConfig(modes=modes_list)

        stance_cfg = StanceConfig(position=stance, intensity=intensity)

        topic_ctx = TopicContext(
            title=topic.strip(),
            summary=topic.strip(),
            raw_input=topic.strip(),
            input_type="text",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
        )

        mode_prompt = blend_modes(mode_config)
        modified_prompt = apply_stance(mode_prompt, stance_cfg)

        injected_client = app.state.llm_client
        try:
            preview_text = generate_preview(
                topic_ctx, modified_prompt, stance_cfg, client=injected_client
            )
        except RuntimeError as exc:
            return templates.TemplateResponse(
                request,
                "partials/error.html",
                {"error": str(exc)},
                status_code=500,
            )

        # Include provider name in response
        provider_name = _get_provider_name(app.state)

        return templates.TemplateResponse(
            request,
            "partials/preview_result.html",
            {"preview_text": preview_text, "provider_name": provider_name},
        )

    @app.post("/export", response_class=HTMLResponse)
    async def export_route(
        request: Request,
        content: str = Form(""),
        title: str = Form("Untitled"),
        format: str = Form("substack"),
    ) -> HTMLResponse:
        """Export piece content to a platform format (HTML partial).

        Builds a minimal GeneratedPiece from the submitted content and
        dispatches it through the existing export() function.

        Args:
            request: The incoming request.
            content: The piece body text.
            title: The piece title.
            format: The target export format.

        Raises:
            HTTPException: 422 if content is empty or format is invalid.
        """
        if not content.strip():
            raise HTTPException(status_code=422, detail="Content is required.")

        from datetime import datetime, timezone

        from opinionforge.exporters import export as do_export
        from opinionforge.models.piece import GeneratedPiece

        # Build a minimal piece for the exporter
        piece = GeneratedPiece(
            id="export-preview",
            created_at=datetime.now(timezone.utc),
            topic=TopicContext(
                title=title,
                summary=title,
                raw_input=title,
                input_type="text",
                key_claims=[],
                key_entities=[],
                subject_domain="general",
            ),
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0, intensity=0.5),
            target_length=800,
            actual_length=len(content.split()),
            title=title,
            body=content,
            preview_text=content[:200],
            sources=[],
            research_queries=[],
            disclaimer=MANDATORY_DISCLAIMER,
        )

        try:
            exported = do_export(piece, format)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return templates.TemplateResponse(
            request,
            "partials/export_result.html",
            {"exported_content": exported, "format": format},
        )

    # ------------------------------------------------------------------
    # History Routes
    # ------------------------------------------------------------------

    _HISTORY_PAGE_SIZE = 20

    async def _get_db() -> Database:
        """Open a connected, initialised Database from app state path."""
        db = Database(app.state.db_path)
        await db.connect()
        await db.initialize()
        return db

    @app.get("/history", response_class=HTMLResponse)
    async def history_list(
        request: Request, page: int = Query(1, ge=1),
    ) -> HTMLResponse:
        """Render the history page with paginated piece cards."""
        from opinionforge.modes import list_modes as _list_modes

        db = await _get_db()
        try:
            store = PieceStore(db)
            total = await store.count()
            total_pages = max(1, math.ceil(total / _HISTORY_PAGE_SIZE))
            offset = (page - 1) * _HISTORY_PAGE_SIZE
            pieces = await store.list_all(
                limit=_HISTORY_PAGE_SIZE, offset=offset,
            )
        finally:
            await db.close()

        return templates.TemplateResponse(
            request, "history.html", {
                "base_template": _base_template(request),
                "pieces": pieces,
                "modes": _list_modes(),
                "page": page,
                "total_pages": total_pages,
                "current_page": "history",
            },
        )

    @app.get("/history/{piece_id}", response_class=HTMLResponse)
    async def history_detail(
        request: Request, piece_id: str,
    ) -> HTMLResponse:
        """Render the detail view for a single piece."""
        db = await _get_db()
        try:
            ps = PieceStore(db)
            piece = await ps.get(piece_id)
            if piece is None:
                raise HTTPException(status_code=404, detail="Piece not found.")
            es = ExportStore(db)
            exports = await es.get_by_piece(piece_id)
        finally:
            await db.close()

        return templates.TemplateResponse(
            request, "history_detail.html", {
                "base_template": _base_template(request),
                "piece": piece,
                "exports": exports,
                "current_page": "history",
            },
        )

    @app.post("/history/search", response_class=HTMLResponse)
    async def history_search(
        request: Request,
        query: str = Form(""),
        mode: str = Form(""),
        stance_min: Optional[str] = Form(None),
        stance_max: Optional[str] = Form(None),
        date_from: str = Form(""),
        date_to: str = Form(""),
        sort_by: str = Form("newest"),
    ) -> HTMLResponse:
        """Search/filter pieces, return HTMX partial with piece cards."""
        db = await _get_db()
        try:
            store = PieceStore(db)
            f_mode = mode or None
            try:
                f_smin = int(stance_min) if stance_min else None
            except ValueError:
                f_smin = None
            try:
                f_smax = int(stance_max) if stance_max else None
            except ValueError:
                f_smax = None
            f_dfrom = date_from or None
            f_dto = date_to or None

            if query.strip():
                pieces = await store.search(query.strip())
                if f_mode:
                    pieces = [p for p in pieces if p.get("mode") == f_mode]
                if f_smin is not None:
                    pieces = [p for p in pieces if (p.get("stance_position") or 0) >= f_smin]
                if f_smax is not None:
                    pieces = [p for p in pieces if (p.get("stance_position") or 0) <= f_smax]
                if f_dfrom:
                    pieces = [p for p in pieces if (p.get("created_at") or "") >= f_dfrom]
                if f_dto:
                    pieces = [p for p in pieces if (p.get("created_at") or "") <= f_dto]
            else:
                pieces = await store.filter_by(
                    mode=f_mode, stance_min=f_smin, stance_max=f_smax,
                    date_from=f_dfrom, date_to=f_dto,
                )
        finally:
            await db.close()

        if sort_by == "oldest":
            pieces.sort(key=lambda p: p.get("created_at") or "")
        elif sort_by == "newest":
            pieces.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        elif sort_by == "words_desc":
            pieces.sort(key=lambda p: p.get("actual_length") or 0, reverse=True)
        elif sort_by == "words_asc":
            pieces.sort(key=lambda p: p.get("actual_length") or 0)
        elif sort_by == "mode_az":
            pieces.sort(key=lambda p: p.get("mode") or "")

        if pieces:
            parts = []
            for piece in pieces:
                parts.append(
                    templates.get_template("partials/piece_card.html").render(piece=piece)
                )
            return HTMLResponse("".join(parts))
        return HTMLResponse(
            '<div class="empty-state"><p>No pieces match your search criteria.</p></div>'
        )

    @app.post("/history/{piece_id}/delete")
    async def history_delete(piece_id: str) -> RedirectResponse:
        """Delete a single piece and redirect to /history."""
        db = await _get_db()
        try:
            await PieceStore(db).delete(piece_id)
        finally:
            await db.close()
        return RedirectResponse(url="/history", status_code=303)

    @app.post("/history/bulk-delete", response_class=HTMLResponse)
    async def history_bulk_delete(
        request: Request, piece_ids: list[str] = Form([]),
    ) -> HTMLResponse:
        """Delete multiple pieces and return updated piece grid."""
        db = await _get_db()
        try:
            store = PieceStore(db)
            if piece_ids:
                await store.bulk_delete(piece_ids)
            pieces = await store.list_all(limit=_HISTORY_PAGE_SIZE, offset=0)
        finally:
            await db.close()

        if pieces:
            parts = []
            for piece in pieces:
                parts.append(
                    templates.get_template("partials/piece_card.html").render(piece=piece)
                )
            return HTMLResponse("".join(parts))
        return HTMLResponse(
            '<div class="empty-state"><p>No pieces yet. Generate your first piece '
            'from the <a href="/">home page</a>.</p></div>'
        )

    @app.post("/history/{piece_id}/export", response_class=HTMLResponse)
    async def history_export(
        request: Request, piece_id: str, format: str = Form("substack"),
    ) -> HTMLResponse:
        """Re-export a stored piece to a platform format."""
        from datetime import datetime, timezone
        from opinionforge.exporters import export as do_export
        from opinionforge.models.piece import GeneratedPiece

        db = await _get_db()
        try:
            ps = PieceStore(db)
            piece_data = await ps.get(piece_id)
            if piece_data is None:
                raise HTTPException(status_code=404, detail="Piece not found.")

            mc_raw = piece_data.get("mode_config")
            if isinstance(mc_raw, dict) and "modes" in mc_raw:
                mc = ModeBlendConfig(modes=[(m[0], m[1]) for m in mc_raw["modes"]])
            else:
                mc = ModeBlendConfig(modes=[(piece_data.get("mode") or "analytical", 100.0)])

            piece_obj = GeneratedPiece(
                id=piece_data["id"],
                created_at=(
                    datetime.fromisoformat(piece_data["created_at"])
                    if piece_data.get("created_at")
                    else datetime.now(timezone.utc)
                ),
                topic=TopicContext(
                    title=piece_data.get("title") or piece_data.get("topic", ""),
                    summary=piece_data.get("topic") or "",
                    raw_input=piece_data.get("topic") or "",
                    input_type="text", key_claims=[], key_entities=[],
                    subject_domain="general",
                ),
                mode_config=mc,
                stance=StanceConfig(
                    position=piece_data.get("stance_position") or 0,
                    intensity=piece_data.get("stance_intensity") or 0.5,
                ),
                target_length=piece_data.get("target_length") or 800,
                actual_length=piece_data.get("actual_length") or 0,
                title=piece_data.get("title") or "Untitled",
                subtitle=piece_data.get("subtitle"),
                body=piece_data.get("body") or "",
                preview_text=piece_data.get("preview_text") or "",
                sources=[], research_queries=[],
                disclaimer=piece_data.get("disclaimer") or MANDATORY_DISCLAIMER,
            )

            try:
                exported = do_export(piece_obj, format)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

            await ExportStore(db).save(piece_id, format, exported)
        finally:
            await db.close()

        return templates.TemplateResponse(
            request, "partials/export_result.html",
            {"exported_content": exported, "format": format},
        )

    # ------------------------------------------------------------------
    # Settings routes
    # ------------------------------------------------------------------

    def _mask_api_key(key: str | None) -> str:
        """Mask an API key, showing only the last 4 characters.

        Args:
            key: The plaintext API key, or None.

        Returns:
            A masked string like ``'****abcd'``, or empty string.
        """
        if not key:
            return ""
        if len(key) <= 4:
            return "*" * len(key)
        return "*" * (len(key) - 4) + key[-4:]

    def _get_models_for_provider(provider_type: str) -> list[str]:
        """Return suggested model identifiers for a provider type.

        Args:
            provider_type: One of the supported provider type strings.

        Returns:
            A list of model name strings.
        """
        model_map: dict[str, list[str]] = {
            "anthropic": [
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-haiku-20240307",
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
            "openai_compatible": [
                "gpt-4o",
                "custom-model",
            ],
            "ollama": [
                "llama3",
                "mistral",
                "codellama",
                "phi3",
            ],
        }
        return model_map.get(provider_type, [])

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        """Render the settings page with current values from storage.

        Args:
            request: The incoming request.
        """
        import json as _json

        from opinionforge.modes import list_modes
        from opinionforge.storage.database import get_db_path
        from opinionforge.storage.encryption import decrypt_key

        async with Database(app.state.db_path) as db:
            settings_store = SettingsStore(db)
            piece_store = PieceStore(db)

            provider_config = await settings_store.get_provider_config()
            preferences = await settings_store.get_user_preferences()
            piece_count = await piece_store.count()

            raw_search = await settings_store.get("search_config")
            search_config = None
            if raw_search:
                search_config = SearchConfig(**_json.loads(raw_search))

        masked_api_key = ""
        if provider_config and provider_config.api_key:
            try:
                plaintext = decrypt_key(provider_config.api_key)
                masked_api_key = _mask_api_key(plaintext)
            except (ValueError, Exception):
                masked_api_key = _mask_api_key(provider_config.api_key)

        masked_search_key = ""
        if search_config and search_config.api_key:
            try:
                plaintext = decrypt_key(search_config.api_key)
                masked_search_key = _mask_api_key(plaintext)
            except (ValueError, Exception):
                masked_search_key = _mask_api_key(search_config.api_key)

        ptype = provider_config.provider_type if provider_config else "anthropic"
        suggested_models = _get_models_for_provider(ptype)
        all_modes = list_modes()
        db_path_display = str(get_db_path())

        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "base_template": _base_template(request),
                "provider_config": provider_config,
                "search_config": search_config,
                "preferences": preferences,
                "masked_api_key": masked_api_key,
                "masked_search_key": masked_search_key,
                "models": suggested_models,
                "all_modes": all_modes,
                "db_path": db_path_display,
                "piece_count": piece_count,
                "current_page": "settings",
            },
        )

    @app.post("/settings/provider", response_class=HTMLResponse)
    async def save_provider(
        request: Request,
        provider_type: str = Form("anthropic"),
        model: str = Form(""),
        api_key: str = Form(""),
        base_url: str = Form(""),
        test_only: int = Query(0),
    ) -> HTMLResponse:
        """Save LLM provider config after testing the connection.

        Args:
            request: The incoming request.
            provider_type: Provider backend identifier.
            model: Model name.
            api_key: API key (plaintext from form).
            base_url: Optional base URL.
            test_only: If 1, only test without saving.
        """
        from opinionforge.storage.encryption import decrypt_key, encrypt_key

        actual_api_key = api_key
        if api_key.startswith("*") or not api_key:
            async with Database(app.state.db_path) as db:
                store = SettingsStore(db)
                existing = await store.get_provider_config()
                if existing and existing.api_key:
                    try:
                        actual_api_key = decrypt_key(existing.api_key)
                    except (ValueError, Exception):
                        actual_api_key = ""

        kwargs: dict[str, object] = {"model": model}
        if actual_api_key:
            kwargs["api_key"] = actual_api_key
        if base_url:
            kwargs["base_url"] = base_url

        registry = ProviderRegistry()
        try:
            test_provider = registry.create_provider(provider_type, **kwargs)
            success, message = await registry.test_connection(test_provider)
        except (ProviderError, ValueError, Exception) as exc:
            success = False
            message = str(exc)

        if not success:
            return templates.TemplateResponse(
                request,
                "partials/settings_feedback.html",
                {"success": False, "error": message, "section": "provider"},
            )

        if test_only:
            return templates.TemplateResponse(
                request,
                "partials/settings_feedback.html",
                {"success": True, "section": "provider"},
            )

        encrypted_key = encrypt_key(actual_api_key) if actual_api_key else None
        config = ProviderConfig(
            provider_type=provider_type,
            model=model,
            api_key=encrypted_key,
            base_url=base_url or None,
        )

        async with Database(app.state.db_path) as db:
            store = SettingsStore(db)
            await store.set_provider_config(config)

        return templates.TemplateResponse(
            request,
            "partials/settings_feedback.html",
            {"success": True, "section": "provider"},
        )

    @app.post("/settings/search", response_class=HTMLResponse)
    async def save_search(
        request: Request,
        search_provider: str = Form("none"),
        search_api_key: str = Form(""),
    ) -> HTMLResponse:
        """Save search provider configuration.

        Args:
            request: The incoming request.
            search_provider: Search backend identifier.
            search_api_key: API key for the search provider.
        """
        import json as _json

        from opinionforge.storage.encryption import encrypt_key

        encrypted_key = None
        if search_api_key and not search_api_key.startswith("*"):
            encrypted_key = encrypt_key(search_api_key)
        elif search_api_key.startswith("*"):
            async with Database(app.state.db_path) as db:
                store = SettingsStore(db)
                raw = await store.get("search_config")
                if raw:
                    existing = SearchConfig(**_json.loads(raw))
                    encrypted_key = existing.api_key

        config = SearchConfig(provider=search_provider, api_key=encrypted_key)
        async with Database(app.state.db_path) as db:
            store = SettingsStore(db)
            await store.set("search_config", _json.dumps(config.model_dump()))

        return templates.TemplateResponse(
            request,
            "partials/settings_feedback.html",
            {"success": True, "section": "search"},
        )

    @app.post("/settings/test-search", response_class=HTMLResponse)
    async def test_search_connection(
        request: Request,
        search_provider: str = Form("none"),
        search_api_key: str = Form(""),
    ) -> HTMLResponse:
        """Test the search provider connection without saving.

        Args:
            request: The incoming request.
            search_provider: Search backend identifier.
            search_api_key: API key for the search provider (plaintext from form).
        """
        import json as _json

        from opinionforge.storage.encryption import decrypt_key

        if search_provider == "none":
            return templates.TemplateResponse(
                request,
                "partials/settings_feedback.html",
                {"success": False, "error": "No search provider selected.", "section": "search"},
            )

        actual_api_key = search_api_key
        if search_api_key.startswith("*") or not search_api_key:
            async with Database(app.state.db_path) as db:
                store = SettingsStore(db)
                raw = await store.get("search_config")
                if raw:
                    existing = SearchConfig(**_json.loads(raw))
                    if existing.api_key:
                        try:
                            actual_api_key = decrypt_key(existing.api_key)
                        except (ValueError, Exception):
                            actual_api_key = ""

        if not actual_api_key:
            return templates.TemplateResponse(
                request,
                "partials/settings_feedback.html",
                {"success": False, "error": "No API key provided.", "section": "search"},
            )

        try:
            from opinionforge.utils.search import TavilySearchClient

            search_client = TavilySearchClient(api_key=actual_api_key)
            results = search_client.search("test", max_results=1)
            success = True
            message = None
        except SystemExit as exc:
            success = False
            message = "Invalid or unauthorized API key." if exc.code == 5 else "Rate limit exceeded."
        except Exception as exc:
            success = False
            message = str(exc)

        return templates.TemplateResponse(
            request,
            "partials/settings_feedback.html",
            {"success": success, "error": message if not success else None, "section": "search"},
        )

    @app.post("/settings/preferences", response_class=HTMLResponse)
    async def save_preferences(
        request: Request,
        default_mode: str = Form("analytical"),
        default_stance: int = Form(0),
        default_intensity: float = Form(0.5),
        default_length: str = Form("standard"),
        theme: str = Form("light"),
    ) -> HTMLResponse:
        """Save user preferences.

        Args:
            request: The incoming request.
            default_mode: Default rhetorical mode.
            default_stance: Default stance position.
            default_intensity: Default intensity.
            default_length: Default length preset.
            theme: UI theme (light/dark).
        """
        prefs = UserPreferences(
            default_mode=default_mode,
            default_stance=default_stance,
            default_intensity=default_intensity,
            default_length=default_length,
            theme=theme,
        )
        async with Database(app.state.db_path) as db:
            store = SettingsStore(db)
            await store.set_user_preferences(prefs)

        return templates.TemplateResponse(
            request,
            "partials/settings_feedback.html",
            {"success": True, "section": "preferences"},
        )

    @app.post("/settings/export-data")
    async def export_data(request: Request) -> Response:
        """Export all pieces as a JSON file download.

        Args:
            request: The incoming request.
        """
        import json as _json

        async with Database(app.state.db_path) as db:
            piece_store = PieceStore(db)
            pieces = await piece_store.list_all(limit=10000)

        json_string = _json.dumps(pieces, indent=2, default=str)
        return Response(
            content=json_string,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    "attachment; filename=opinionforge_export.json"
                ),
            },
        )

    @app.post("/settings/clear-history", response_class=HTMLResponse)
    async def clear_history_settings(request: Request) -> HTMLResponse:
        """Delete all pieces and exports, returning a feedback partial.

        Args:
            request: The incoming request.
        """
        async with Database(app.state.db_path) as db:
            await db.execute("DELETE FROM exports")
            await db.execute("DELETE FROM pieces")
            await db.commit()

        return templates.TemplateResponse(
            request,
            "partials/settings_feedback.html",
            {"success": True, "section": "data"},
        )

    return app
