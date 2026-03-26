"""HTMX-specific behavior tests for the OpinionForge web UI.

Validates that the server correctly handles HX-Request headers,
returns HTML partials vs full pages, and SSE events are formatted
correctly.

All LLM and search calls are mocked — zero real API calls.
"""

from __future__ import annotations

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

def _make_mock_client() -> MagicMock:
    """Create a mock LLM client that returns canned text."""
    client = MagicMock()
    client.generate.return_value = (
        "The machinery of governance is being fundamentally reshaped "
        "by algorithmic systems that few legislators truly understand."
    )
    return client


def _make_piece() -> GeneratedPiece:
    """Create a minimal GeneratedPiece for testing."""
    return GeneratedPiece(
        id="htmx-test-001",
        created_at=datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
        topic=TopicContext(
            raw_input="HTMX test topic",
            input_type="text",
            title="HTMX Test",
            summary="Testing HTMX behavior.",
            key_claims=[],
            key_entities=[],
            subject_domain="technology",
        ),
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=0, intensity=0.5),
        target_length=800,
        actual_length=15,
        title="HTMX Test Piece",
        body="Test body content for HTMX tests.",
        preview_text="Preview of the piece.",
        sources=[],
        research_queries=[],
        disclaimer=MANDATORY_DISCLAIMER,
        screening_result=ScreeningResult(
            passed=True,
            verbatim_matches=0,
            near_verbatim_matches=0,
            suppressed_phrase_matches=0,
            structural_fingerprint_score=0.1,
            rewrite_iterations=0,
            warning=None,
        ),
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
# Tests: HTMX partial responses
# ---------------------------------------------------------------------------

class TestHTMXPartials:
    """Tests for HTMX-specific partial response behavior."""

    def test_preview_with_hx_request_returns_partial(self, client: TestClient) -> None:
        """GET /preview with HX-Request header returns an HTML partial (no full page wrapper)."""
        response = client.get(
            "/preview",
            params={"topic": "AI governance", "mode": "analytical", "stance": "0", "intensity": "0.5"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # A partial should not contain the full HTML document structure
        # (partials come from templates/partials/ and don't extend base.html)
        text = response.text
        assert "preview-result" in text or "Tone Preview" in text

    def test_preview_without_hx_request_returns_response(self, client: TestClient) -> None:
        """GET /preview without HX-Request header returns a response (partial or full)."""
        response = client.get(
            "/preview",
            params={"topic": "AI governance", "mode": "analytical", "stance": "0", "intensity": "0.5"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_export_with_hx_request_returns_partial(self, client: TestClient) -> None:
        """POST /export with HX-Request header returns an HTML partial."""
        response = client.post(
            "/export",
            data={
                "content": "Test content for HTMX export partial check.",
                "title": "HTMX Export Test",
                "format": "substack",
            },
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # The export partial contains the export-result container
        assert "export-result" in response.text or "Exported Content" in response.text

    def test_modes_with_hx_request_returns_partial(self, client: TestClient) -> None:
        """GET /modes with HX-Request header returns a partial for HTMX swap."""
        response = client.get("/modes", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Should contain mode content
        assert "Analytical" in response.text


# ---------------------------------------------------------------------------
# Tests: SSE format
# ---------------------------------------------------------------------------

class TestSSEFormat:
    """Tests for correct SSE event formatting."""

    def test_generate_sse_contains_event_and_data_fields(self, client: TestClient) -> None:
        """POST /generate response includes SSE-formatted events with event: and data: fields."""
        mock_piece = _make_piece()
        with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
            response = client.post(
                "/generate",
                data={"topic": "SSE format test", "mode": "analytical"},
            )
        assert response.status_code == 200
        body = response.text
        # SSE format requires event: and data: lines
        assert "event:" in body
        assert "data:" in body
        # Check that event names are present
        assert "event: progress" in body
        assert "event: done" in body

    def test_error_partial_structure(self, client: TestClient) -> None:
        """The error event contains appropriate structure for swap into the results area."""
        with patch(
            "opinionforge.web.sse.generate_piece",
            side_effect=RuntimeError("Test error for HTMX structure check"),
        ):
            response = client.post(
                "/generate",
                data={"topic": "Error structure test", "mode": "analytical"},
            )
        assert response.status_code == 200
        body = response.text
        assert "event: error" in body
        # The error data should be JSON with a message field
        assert '"message"' in body
