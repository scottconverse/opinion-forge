"""Tests for --provider and --model CLI flags on write and preview commands.

All LLM calls are mocked -- zero real API calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from opinionforge.cli import app
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece, ScreeningResult
from opinionforge.models.topic import TopicContext

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_MOCK_LLM_OUTPUT = (
    "## Test Piece Title\n\n"
    "This is a test opinion piece body about an important topic. "
    "The evidence is compelling and the argument is clear. "
    "We must consider all perspectives carefully.\n\n"
    "The second paragraph elaborates on the key points."
)

_MOCK_MODE_PROMPT = "Write analytically with measured rhetorical force."

_MOCK_SCREENING_RESULT = ScreeningResult(
    passed=True,
    verbatim_matches=0,
    near_verbatim_matches=0,
    suppressed_phrase_matches=0,
    structural_fingerprint_score=0.0,
    rewrite_iterations=0,
)

_MOCK_PREVIEW_TEXT = "This is a preview of the opinion piece."


def _make_mock_provider(model_name_str: str = "ollama/llama3") -> MagicMock:
    """Create a mock LLMProvider."""
    from unittest.mock import AsyncMock

    provider = MagicMock()
    provider.generate = AsyncMock(return_value=_MOCK_LLM_OUTPUT)
    provider.model_name.return_value = model_name_str
    return provider


@contextmanager
def _patch_write_pipeline() -> Generator[MagicMock, None, None]:
    """Patch the entire write pipeline for CLI tests.

    Patches topic ingestion, mode loading, stance, generation, screening,
    and storage to allow testing CLI flags in isolation.

    Yields:
        The mock generate_piece function.
    """
    from datetime import datetime, timezone

    mock_piece = GeneratedPiece(
        id="test-id",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        topic=TopicContext(
            raw_input="test", input_type="text", title="Test",
            summary="Test", key_claims=[], key_entities=[],
            subject_domain="general",
        ),
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=0, intensity=0.5),
        target_length=800,
        actual_length=50,
        title="Test Piece Title",
        body="Test body.",
        preview_text="Test preview.",
        sources=[],
        research_queries=[],
        disclaimer="AI-generated.",
        screening_result=_MOCK_SCREENING_RESULT,
    )

    with patch(
        "opinionforge.cli._ingest_topic",
        return_value=TopicContext(
            raw_input="test", input_type="text", title="Test",
            summary="Test", key_claims=[], key_entities=[],
            subject_domain="general",
        ),
    ), patch(
        "opinionforge.cli._load_modes",
        return_value=_MOCK_MODE_PROMPT,
    ), patch(
        "opinionforge.core.stance.apply_stance",
        return_value=_MOCK_MODE_PROMPT,
    ), patch(
        "opinionforge.core.generator.generate_piece",
        return_value=mock_piece,
    ) as mock_gen, patch(
        "opinionforge.core.preview.generate_preview",
        return_value=_MOCK_PREVIEW_TEXT,
    ), patch(
        "opinionforge.core.preview.create_llm_client",
        return_value=MagicMock(),
    ), patch(
        "opinionforge.core.preview.create_llm_client_from_provider",
        return_value=MagicMock(),
    ), patch(
        "opinionforge.providers.registry.ProviderRegistry.create_provider",
        return_value=_make_mock_provider(),
    ):
        yield mock_gen


# ---------------------------------------------------------------------------
# Tests: --provider flag on write command
# ---------------------------------------------------------------------------


class TestWriteProviderFlag:
    """Tests for --provider flag on the write command."""

    def test_provider_ollama(self) -> None:
        """--provider ollama is accepted and passes provider_type to generate_piece."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app, ["write", "Test topic", "--provider", "ollama", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 0, result.output
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs.get("provider_type") == "ollama"

    def test_provider_anthropic(self) -> None:
        """--provider anthropic is accepted."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app, ["write", "Test topic", "--provider", "anthropic", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 0, result.output
        assert mock_gen.call_args.kwargs.get("provider_type") == "anthropic"

    def test_provider_openai(self) -> None:
        """--provider openai is accepted."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app, ["write", "Test topic", "--provider", "openai", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 0, result.output
        assert mock_gen.call_args.kwargs.get("provider_type") == "openai"

    def test_provider_openai_compatible(self) -> None:
        """--provider openai_compatible is accepted."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app, ["write", "Test topic", "--provider", "openai_compatible", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 0, result.output
        assert mock_gen.call_args.kwargs.get("provider_type") == "openai_compatible"

    def test_invalid_provider_exits_code_2(self) -> None:
        """An invalid --provider value exits with code 2."""
        with _patch_write_pipeline():
            result = runner.invoke(
                app, ["write", "Test topic", "--provider", "fakeprovider", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 2

    def test_model_override(self) -> None:
        """--model flag overrides the model name."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app, ["write", "Test topic", "--model", "custom-model-v2", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 0, result.output
        assert mock_gen.call_args.kwargs.get("model") == "custom-model-v2"

    def test_provider_without_model_uses_default(self) -> None:
        """--provider without --model uses the default model for that provider."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app, ["write", "Test topic", "--provider", "anthropic", "--no-preview", "--no-research"],
            )
        assert result.exit_code == 0, result.output
        # model should be None (default)
        assert mock_gen.call_args.kwargs.get("model") is None

    def test_both_flags_together(self) -> None:
        """--provider and --model together are both passed to generate_piece."""
        with _patch_write_pipeline() as mock_gen:
            result = runner.invoke(
                app,
                [
                    "write", "Test topic",
                    "--provider", "ollama",
                    "--model", "llama3:70b",
                    "--no-preview", "--no-research",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_gen.call_args.kwargs.get("provider_type") == "ollama"
        assert mock_gen.call_args.kwargs.get("model") == "llama3:70b"


# ---------------------------------------------------------------------------
# Tests: --provider flag on preview command
# ---------------------------------------------------------------------------


class TestPreviewProviderFlag:
    """Tests for --provider flag on the preview command."""

    def test_preview_provider_ollama(self) -> None:
        """preview --provider ollama is accepted."""
        with _patch_write_pipeline():
            result = runner.invoke(
                app, ["preview", "Test topic", "--provider", "ollama"],
            )
        assert result.exit_code == 0, result.output

    def test_preview_invalid_provider_exits_code_2(self) -> None:
        """preview --provider with invalid value exits code 2."""
        with _patch_write_pipeline():
            result = runner.invoke(
                app, ["preview", "Test topic", "--provider", "badprovider"],
            )
        assert result.exit_code == 2

    def test_preview_model_override(self) -> None:
        """preview --model flag is accepted."""
        with _patch_write_pipeline():
            result = runner.invoke(
                app, ["preview", "Test topic", "--model", "custom-model"],
            )
        assert result.exit_code == 0, result.output
