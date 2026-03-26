"""FastAPI application for OpinionForge web UI.

Provides routes for generation, preview, mode browsing, export, and about.
All generation uses the existing engine — no rewrites.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.core.preview import LLMClient
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.topic import TopicContext
from opinionforge.web.sse import generation_event_stream

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


def create_app(*, client: LLMClient | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        client: Optional LLM client for dependency injection (used in tests).

    Returns:
        A fully-configured FastAPI instance.
    """
    app = FastAPI(title="OpinionForge", version="1.0.0")

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Jinja2 templates
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Store injected client on app state for use in route handlers
    app.state.llm_client = client

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

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        """Render the home page with the generation form."""
        from opinionforge.modes import list_modes

        all_modes = list_modes()
        categories = _group_modes_by_category(all_modes)
        selected_mode = request.query_params.get("mode", "")
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
            },
        )

    @app.get("/about", response_class=HTMLResponse)
    async def about(request: Request) -> HTMLResponse:
        """Render the about page."""
        return templates.TemplateResponse(
            request, "about.html", {"base_template": _base_template(request)}
        )

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
    ) -> EventSourceResponse:
        """Generate an opinion piece via SSE streaming (POST, form data).

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
    ) -> EventSourceResponse:
        """Generate an opinion piece via SSE streaming (GET, query params).

        This endpoint is used by the HTMX SSE extension (EventSource requires GET).

        Args:
            topic: The topic text.
            mode: Rhetorical mode or blend specification.
            stance: Argumentative emphasis direction (-100 to +100).
            intensity: Rhetorical heat (0.0 to 1.0).
            length: Length preset or custom word count.

        Raises:
            HTTPException: 422 if topic is empty.
        """
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

        return templates.TemplateResponse(
            request,
            "partials/preview_result.html",
            {"preview_text": preview_text},
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

    return app
