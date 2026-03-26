"""SSE streaming helper for OpinionForge generation pipeline.

Wraps the synchronous generate_piece() function, yielding Server-Sent Events
for each stage of the generation process: researching, generating, screening,
done (or error).
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from opinionforge.core.generator import MANDATORY_DISCLAIMER, generate_piece
from opinionforge.core.preview import LLMClient
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece
from opinionforge.models.topic import TopicContext

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)


def _format_sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """Format a single SSE event payload.

    Args:
        event: The SSE event name (e.g. 'progress', 'done', 'error').
        data: The JSON-serialisable data payload.

    Returns:
        A dict with 'event' and 'data' keys for sse-starlette.
    """
    return {"event": event, "data": json.dumps(data)}


def _run_generation(
    topic: TopicContext,
    mode_config: ModeBlendConfig,
    stance: StanceConfig,
    target_length: int | str,
    research_context: str | None,
    client: LLMClient | None,
) -> GeneratedPiece:
    """Run generate_piece synchronously (called via run_in_executor).

    Args:
        topic: The normalised topic context.
        mode_config: Rhetorical mode blend configuration.
        stance: Stance and intensity controls.
        target_length: Target word count.
        research_context: Optional research context string.
        client: Optional injected LLM client.

    Returns:
        The generated piece.
    """
    return generate_piece(
        topic=topic,
        mode_config=mode_config,
        stance=stance,
        target_length=target_length,
        research_context=research_context,
        client=client,
    )


async def generation_event_stream(
    topic: TopicContext,
    mode_config: ModeBlendConfig,
    stance: StanceConfig,
    target_length: int | str = "standard",
    research_context: str | None = None,
    client: LLMClient | None = None,
) -> AsyncGenerator[dict[str, str], None]:
    """Yield SSE events wrapping the generation pipeline.

    Stages emitted:
        * ``researching`` — before generation begins (placeholder for future
          research integration).
        * ``generating`` — generation has started.
        * ``screening`` — similarity screening in progress.
        * ``done`` — generation complete; payload includes rendered HTML.
        * ``error`` — an error occurred; payload includes error message.

    The generator runs in a thread-pool executor so the async event loop is
    never blocked by the synchronous ``generate_piece()`` call.

    Args:
        topic: The normalised topic context.
        mode_config: Rhetorical mode blend configuration.
        stance: Stance and intensity controls.
        target_length: Target word count or preset name.
        research_context: Optional research context string.
        client: Optional injected LLM client (for tests).

    Yields:
        SSE-formatted dicts with 'event' and 'data' keys.
    """
    # Stage 1: researching
    yield _format_sse("progress", {"stage": "researching", "message": "Preparing research context..."})

    # Stage 2: generating
    yield _format_sse("progress", {"stage": "generating", "message": "Generating opinion piece..."})

    # Run the synchronous generation in a thread so we don't block the loop.
    loop = asyncio.get_running_loop()
    try:
        piece: GeneratedPiece = await loop.run_in_executor(
            None,
            partial(
                _run_generation,
                topic=topic,
                mode_config=mode_config,
                stance=stance,
                target_length=target_length,
                research_context=research_context,
                client=client,
            ),
        )
    except RuntimeError as exc:
        exc_msg = str(exc)
        if "similarity screening" in exc_msg.lower():
            yield _format_sse("progress", {"stage": "screening", "message": "Screening output..."})
            yield _format_sse("error", {"stage": "screening", "message": exc_msg})
        else:
            yield _format_sse("error", {"stage": "generating", "message": exc_msg})
        return
    except Exception as exc:
        yield _format_sse("error", {"stage": "generating", "message": str(exc)})
        return

    # Stage 3: screening passed (if we got here, it passed)
    yield _format_sse("progress", {"stage": "screening", "message": "Screening passed."})

    # Stage 4: done — render the full piece_result.html partial
    template = _jinja_env.get_template("partials/piece_result.html")
    html = template.render(
        title=piece.title,
        subtitle=getattr(piece, "subtitle", None),
        body=piece.body,
        sources=piece.sources or [],
        image_prompt=getattr(piece, "image_prompt", None),
        disclaimer=piece.disclaimer,
    )
    yield _format_sse(
        "done",
        {
            "stage": "done",
            "title": piece.title,
            "body": piece.body,
            "disclaimer": piece.disclaimer,
            "html": html,
        },
    )
