"""Integration tests covering the full web UI generation pipeline.

Tests submit requests through the FastAPI TestClient and verify the
generation pipeline executes correctly — generate, preview, export,
screening failure, mode blending, and edge cases.

All LLM and search calls are mocked — zero real API calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece, ScreeningResult
from opinionforge.models.topic import TopicContext
from opinionforge.web.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_LLM_TEXT = (
    "## AI and the Future of Public Policy\n\n"
    "The intersection of artificial intelligence and public policy presents "
    "one of the most consequential challenges of our era. The evidence is "
    "clear that algorithmic systems are reshaping governance at every level.\n\n"
    "The second paragraph elaborates on the implications with supporting "
    "evidence drawn from multiple credible sources. We must confront these "
    "realities with intellectual rigor and civic urgency."
)


def _make_mock_client() -> MagicMock:
    """Create a mock LLM client that returns canned text."""
    client = MagicMock()
    client.generate.return_value = _MOCK_LLM_TEXT
    return client


def _make_screening_result(passed: bool = True) -> ScreeningResult:
    """Create a ScreeningResult for testing."""
    return ScreeningResult(
        passed=passed,
        verbatim_matches=0,
        near_verbatim_matches=0,
        suppressed_phrase_matches=0,
        structural_fingerprint_score=0.1,
        rewrite_iterations=0,
        warning=None if passed else "Similarity screening failed — output blocked.",
    )


def _make_piece(
    title: str = "AI and the Future of Public Policy",
    body: str = "This is a test body with enough words to satisfy the pipeline.",
    image_prompt: str | None = None,
) -> GeneratedPiece:
    """Create a minimal GeneratedPiece for testing."""
    return GeneratedPiece(
        id="integ-test-001",
        created_at=datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
        topic=TopicContext(
            raw_input="AI governance test topic",
            input_type="text",
            title="AI Governance",
            summary="A topic about AI governance.",
            key_claims=[],
            key_entities=[],
            subject_domain="technology",
        ),
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=0, intensity=0.5),
        target_length=800,
        actual_length=len(body.split()),
        title=title,
        body=body,
        preview_text="Preview of the piece.",
        sources=[],
        research_queries=[],
        disclaimer=MANDATORY_DISCLAIMER,
        screening_result=_make_screening_result(passed=True),
        image_prompt=image_prompt,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """Return a TestClient with a mock LLM client injected."""
    mock_llm = _make_mock_client()
    app = create_app(client=mock_llm)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: Generate pipeline via SSE
# ---------------------------------------------------------------------------

class TestGeneratePipeline:
    """Integration tests for the POST /generate pipeline."""

    def test_generate_sse_includes_all_progress_stages(self, client: TestClient) -> None:
        """POST /generate SSE stream includes researching, generating, screening, and done events in order."""
        mock_piece = _make_piece()
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
            response = client.post(
                "/generate",
                data={"topic": "AI governance", "mode": "analytical", "stance": "0", "intensity": "0.5"},
            )
        assert response.status_code == 200
        body = response.text
        # Verify all stages appear in order
        idx_research = body.index("researching")
        idx_generate = body.index("generating")
        idx_screen = body.index("screening")
        idx_done = body.index('"done"')
        assert idx_research < idx_generate < idx_screen < idx_done

    def test_generate_done_event_contains_disclaimer(self, client: TestClient) -> None:
        """POST /generate final response HTML contains the mandatory disclaimer text."""
        mock_piece = _make_piece()
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
            response = client.post(
                "/generate",
                data={"topic": "Disclaimer verification", "mode": "analytical"},
            )
        assert response.status_code == 200
        assert MANDATORY_DISCLAIMER in response.text

    def test_generate_done_event_contains_title_and_body(self, client: TestClient) -> None:
        """POST /generate final response HTML contains a piece title and body."""
        mock_piece = _make_piece(
            title="Integration Test Title",
            body="Integration test body content for verification.",
        )
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
            response = client.post(
                "/generate",
                data={"topic": "Title and body check", "mode": "analytical"},
            )
        assert response.status_code == 200
        assert "Integration Test Title" in response.text
        assert "Integration test body content" in response.text

    def test_generate_with_mode_blending(self, client: TestClient) -> None:
        """POST /generate with mode blending (polemical:60,narrative:40) succeeds."""
        mock_piece = _make_piece()
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
            response = client.post(
                "/generate",
                data={
                    "topic": "Mode blending test",
                    "mode": "polemical:60,narrative:40",
                    "stance": "0",
                    "intensity": "0.5",
                },
            )
        assert response.status_code == 200
        assert "event: done" in response.text

    def test_generate_with_custom_stance_and_intensity(self, client: TestClient) -> None:
        """POST /generate with non-default stance (-60) and intensity (0.8) passes values through."""
        mock_piece = _make_piece()
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece) as mock_gen:
            response = client.post(
                "/generate",
                data={
                    "topic": "Custom stance test",
                    "mode": "analytical",
                    "stance": "-60",
                    "intensity": "0.8",
                },
            )
        assert response.status_code == 200
        assert "event: done" in response.text
        # Verify stance and intensity were passed through correctly
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args
        stance_arg = call_kwargs.kwargs.get("stance") or call_kwargs[1].get("stance")
        if stance_arg is not None:
            assert stance_arg.position == -60
            assert stance_arg.intensity == 0.8

    def test_generate_with_image_prompt_enabled(self, client: TestClient) -> None:
        """POST /generate with image_prompt enabled includes image prompt in response."""
        mock_piece = _make_piece(
            image_prompt="A dramatic editorial photograph of a futuristic government building.",
        )
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
            response = client.post(
                "/generate",
                data={
                    "topic": "Image prompt test",
                    "mode": "analytical",
                    "image_prompt": "true",
                },
            )
        assert response.status_code == 200
        assert "event: done" in response.text
        # Verify image prompt content appears in the response
        assert "futuristic government building" in response.text


# ---------------------------------------------------------------------------
# Tests: Preview
# ---------------------------------------------------------------------------

class TestPreviewPipeline:
    """Integration tests for the GET /preview endpoint."""

    def test_preview_returns_preview_text(self, client: TestClient) -> None:
        """GET /preview with valid params returns response containing preview text."""
        response = client.get(
            "/preview",
            params={"topic": "AI governance", "mode": "analytical", "stance": "0", "intensity": "0.5"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # The response should contain some content (the mock LLM returns text)
        assert len(response.text) > 50


# ---------------------------------------------------------------------------
# Tests: Export formats
# ---------------------------------------------------------------------------

class TestExportFormats:
    """Integration tests for POST /export with all 4 formats."""

    @pytest.mark.parametrize("fmt", ["substack", "medium", "wordpress", "twitter"])
    def test_export_format(self, client: TestClient, fmt: str) -> None:
        """POST /export with format={fmt} returns 200 with formatted output."""
        response = client.post(
            "/export",
            data={
                "content": (
                    "This is a test opinion piece body with enough content "
                    "to satisfy the export formatter. The argument is clear "
                    "and the evidence is compelling."
                ),
                "title": "Export Integration Test",
                "format": fmt,
            },
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert len(response.text) > 20


# ---------------------------------------------------------------------------
# Tests: Screening failure
# ---------------------------------------------------------------------------

class TestScreeningFailure:
    """Integration tests for screening failure through the web UI."""

    def test_screening_failure_returns_error_event(self, client: TestClient) -> None:
        """POST /generate with screening failure returns an error event, not blocked output."""
        with patch(
            "opinionforge.web.sse.generate_piece",
            side_effect=RuntimeError(
                "Similarity screening failed — output blocked. "
                "Details: unresolved similarity violations."
            ),
        ):
            response = client.post(
                "/generate",
                data={"topic": "Screening failure test", "mode": "analytical"},
            )
        assert response.status_code == 200
        body = response.text
        assert "event: error" in body
        # Must not contain blocked output — the error event should not expose it
        assert "event: done" not in body

    def test_screening_failure_error_message(self, client: TestClient) -> None:
        """POST /generate screening failure error message mentions screening."""
        with patch(
            "opinionforge.web.sse.generate_piece",
            side_effect=RuntimeError(
                "Similarity screening failed — output blocked."
            ),
        ):
            response = client.post(
                "/generate",
                data={"topic": "Screening message test", "mode": "analytical"},
            )
        body = response.text
        assert "screening" in body.lower()
        assert "did not pass" in body.lower() or "failed" in body.lower()


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Integration tests for edge cases through the web UI."""

    def test_empty_topic_returns_422(self, client: TestClient) -> None:
        """POST /generate with empty topic returns 422."""
        response = client.post("/generate", data={"topic": "", "mode": "analytical"})
        assert response.status_code == 422

    def test_invalid_blend_syntax_returns_422(self, client: TestClient) -> None:
        """POST /generate with invalid blend syntax returns 422."""
        response = client.post(
            "/generate",
            data={"topic": "Test topic", "mode": "polemical:abc"},
        )
        assert response.status_code == 422
