"""FastAPI route tests for the OpinionForge web UI.

All LLM and search calls are mocked — zero real API calls.
Uses FastAPI TestClient to exercise every route defined in app.py.
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
    "## Test Opinion Piece\n\n"
    "This is a test opinion piece body about an important topic. "
    "The evidence is compelling and the argument is clear. "
    "We must consider all perspectives carefully.\n\n"
    "The second paragraph elaborates on the key points with supporting "
    "evidence drawn from multiple credible sources."
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


def _make_piece(title: str = "Test Opinion Piece", body: str = "Body text here.") -> GeneratedPiece:
    """Create a minimal GeneratedPiece for testing."""
    return GeneratedPiece(
        id="test-id-001",
        created_at=datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
        topic=TopicContext(
            raw_input="test topic",
            input_type="text",
            title="Test Topic",
            summary="A test topic summary.",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
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
# Tests: basic pages
# ---------------------------------------------------------------------------

def test_home_returns_200(client: TestClient) -> None:
    """GET / returns 200 with text/html content type."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_home_contains_opinionforge(client: TestClient) -> None:
    """GET / response body mentions OpinionForge."""
    response = client.get("/")
    assert "OpinionForge" in response.text


def test_about_returns_200(client: TestClient) -> None:
    """GET /about returns 200 with text/html content type."""
    response = client.get("/about")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_about_contains_version(client: TestClient) -> None:
    """GET /about mentions version info."""
    response = client.get("/about")
    assert "1.0.0" in response.text


# ---------------------------------------------------------------------------
# Tests: modes listing
# ---------------------------------------------------------------------------

def test_modes_returns_200(client: TestClient) -> None:
    """GET /modes returns 200 and response contains all 12 mode IDs."""
    response = client.get("/modes")
    assert response.status_code == 200
    expected_ids = [
        "analytical", "aphoristic", "data_driven", "dialectical",
        "forensic", "measured", "narrative", "oratorical",
        "polemical", "populist", "provocative", "satirical",
    ]
    for mode_id in expected_ids:
        assert mode_id in response.text, f"Mode '{mode_id}' not found in /modes response"


def test_mode_detail_polemical(client: TestClient) -> None:
    """GET /modes/polemical returns 200 with mode details."""
    response = client.get("/modes/polemical")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Polemical" in response.text


def test_mode_detail_not_found(client: TestClient) -> None:
    """GET /modes/nonexistent returns 404."""
    response = client.get("/modes/nonexistent")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests: preview
# ---------------------------------------------------------------------------

def test_preview_valid_params(client: TestClient) -> None:
    """GET /preview with valid params returns 200 with HTML partial."""
    response = client.get(
        "/preview",
        params={"topic": "AI governance", "mode": "analytical", "stance": 0, "intensity": 0.5},
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_preview_missing_topic(client: TestClient) -> None:
    """GET /preview with empty topic returns 422."""
    response = client.get("/preview", params={"topic": "", "mode": "analytical"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: generate (SSE)
# ---------------------------------------------------------------------------

def test_generate_valid_returns_sse(client: TestClient) -> None:
    """POST /generate with valid form data returns 200 with text/event-stream."""
    mock_piece = _make_piece(
        title="Test Opinion Piece",
        body="This is a test body with enough words to pass.",
    )
    with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
        response = client.post(
            "/generate",
            data={"topic": "AI governance", "mode": "analytical", "stance": "0", "intensity": "0.5"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


def test_generate_missing_topic(client: TestClient) -> None:
    """POST /generate with empty topic returns 422."""
    response = client.post("/generate", data={"topic": "", "mode": "analytical"})
    assert response.status_code == 422


def test_generate_sse_contains_done_event(client: TestClient) -> None:
    """POST /generate SSE stream includes a 'done' event with piece content."""
    mock_piece = _make_piece(
        title="SSE Test Piece",
        body="This is the SSE test body with appropriate content.",
    )
    with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
        response = client.post(
            "/generate",
            data={"topic": "Testing SSE", "mode": "analytical"},
        )
        assert response.status_code == 200
        body_text = response.text
        assert "event: done" in body_text


def test_generate_sse_includes_disclaimer(client: TestClient) -> None:
    """POST /generate done event includes the mandatory disclaimer text."""
    mock_piece = _make_piece(
        title="Disclaimer Test Piece",
        body="This piece tests the disclaimer requirement.",
    )
    with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
        response = client.post(
            "/generate",
            data={"topic": "Disclaimer check", "mode": "analytical"},
        )
        assert response.status_code == 200
        assert MANDATORY_DISCLAIMER in response.text


def test_generate_screening_failure_yields_error(client: TestClient) -> None:
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
            data={"topic": "Screening test", "mode": "analytical"},
        )
        assert response.status_code == 200
        body_text = response.text
        assert "event: error" in body_text
        assert "screening" in body_text.lower()


def test_generate_generic_error_yields_error_event(client: TestClient) -> None:
    """POST /generate with a generic generation error yields an error event."""
    with patch(
        "opinionforge.web.sse.generate_piece",
        side_effect=RuntimeError("LLM provider returned an unexpected error"),
    ):
        response = client.post(
            "/generate",
            data={"topic": "Error test", "mode": "analytical"},
        )
        assert response.status_code == 200
        body_text = response.text
        assert "event: error" in body_text


def test_generate_progress_events(client: TestClient) -> None:
    """POST /generate SSE stream includes progress events for all stages."""
    mock_piece = _make_piece(
        title="Progress Test Piece",
        body="Testing that progress events are emitted for each stage.",
    )
    with patch("opinionforge.web.sse.generate_piece", return_value=mock_piece):
        response = client.post(
            "/generate",
            data={"topic": "Progress events", "mode": "analytical"},
        )
        body_text = response.text
        assert "researching" in body_text
        assert "generating" in body_text
        assert "screening" in body_text


# ---------------------------------------------------------------------------
# Tests: export
# ---------------------------------------------------------------------------

def test_export_valid_returns_200(client: TestClient) -> None:
    """POST /export with valid data returns 200."""
    response = client.post(
        "/export",
        data={
            "content": "This is a test opinion piece body.",
            "title": "Test Title",
            "format": "substack",
        },
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_export_empty_content_returns_422(client: TestClient) -> None:
    """POST /export with empty content returns 422."""
    response = client.post(
        "/export",
        data={"content": "", "title": "Test", "format": "substack"},
    )
    assert response.status_code == 422


def test_export_invalid_format_returns_422(client: TestClient) -> None:
    """POST /export with unknown format returns 422."""
    response = client.post(
        "/export",
        data={"content": "Some body text.", "title": "Test", "format": "unknown_fmt"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: static files
# ---------------------------------------------------------------------------

def test_static_css_accessible(client: TestClient) -> None:
    """GET /static/style.css returns 200."""
    response = client.get("/static/style.css")
    assert response.status_code == 200
