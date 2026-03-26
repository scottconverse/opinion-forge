"""End-to-end integration tests exercising the full OpinionForge pipeline.

These tests run the complete flow from CLI input to generated output,
with all external calls (LLM API, search API, URL fetching) mocked.
No real network requests are made. All tests use typer.testing.CliRunner.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from opinionforge.cli import app
from opinionforge.utils.fetcher import FetchResult
from opinionforge.utils.search import SearchResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_GENERATED = (
    "## The Algorithmic Threat to Democracy\n\n"
    "It would be comforting to believe that the machinery of self-governance "
    "is immune to the disruptions of artificial intelligence. Comforting, "
    "but dangerously naive. The evidence before us suggests that democratic "
    "institutions face a challenge unlike any they have encountered since "
    "the invention of the printing press.\n\n"
    "The first and most obvious danger lies in the capacity of AI systems "
    "to generate misinformation at industrial scale. Where once a propagandist "
    "required an army of scribes, a single algorithm can now produce a torrent "
    "of plausible falsehoods faster than any human fact-checker can debunk them.\n\n"
    "We must not succumb to technophobic paralysis. The same tools that "
    "threaten democratic deliberation can also enhance it."
)

_MOCK_PREVIEW = (
    "The machinery of democracy is about to be tested by an adversary "
    "it never anticipated: the algorithm."
)


def _mock_search_results() -> list[SearchResult]:
    """Return a realistic list of mock search results."""
    return [
        SearchResult(
            url="https://www.nytimes.com/2025/01/15/technology/ai-democracy.html",
            title="AI and the Future of Democracy",
            snippet="According to a study, 73 percent of agencies use AI.",
            raw_content="Article about AI in government agencies.",
        ),
        SearchResult(
            url="https://www.washingtonpost.com/tech/2025/ai-elections/",
            title="How AI Could Shape Elections",
            snippet="AI-generated misinformation has surged.",
            raw_content=None,
        ),
    ]


def _mock_fetch_result(url: str = "https://example.com/article") -> FetchResult:
    """Return a successful mock FetchResult."""
    return FetchResult(
        url=url,
        title="Test Article",
        text="According to a study, 73 percent of agencies use AI tools.",
        fetched_at=datetime.now(timezone.utc),
        success=True,
    )


@contextlib.contextmanager
def _full_pipeline_mocks():
    """Context manager that mocks all external calls for E2E tests.

    Patches create_llm_client in both the preview and generator modules
    (each imports it independently), plus search and URL fetching.

    Yields (mock_llm, mock_search).
    """
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _MOCK_GENERATED

    mock_search = MagicMock()
    mock_search.search.return_value = _mock_search_results()

    with patch("opinionforge.core.preview.create_llm_client", return_value=mock_llm), \
         patch("opinionforge.core.generator.create_llm_client", return_value=mock_llm), \
         patch("opinionforge.utils.search.get_search_client", return_value=mock_search), \
         patch("opinionforge.core.research.get_search_client", return_value=mock_search), \
         patch("opinionforge.utils.fetcher.fetch_url", return_value=_mock_fetch_result()), \
         patch("opinionforge.core.research.fetch_url", return_value=_mock_fetch_result()):
        yield mock_llm, mock_search


# ---------------------------------------------------------------------------
# Write command: basic mode/stance/intensity
# ---------------------------------------------------------------------------


class TestWriteCommandBasic:
    """Tests for the write command with the --mode flag."""

    def test_write_polemical_mode_succeeds(self) -> None:
        """'opinionforge write topic --mode polemical' succeeds with exit 0."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "polemical",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert mock_llm.generate.called

    def test_write_with_stance_and_intensity(self) -> None:
        """'opinionforge write topic --mode polemical --stance -60 --intensity 0.8' succeeds."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "polemical",
                "--stance", "-60",
                "--intensity", "0.8",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert mock_llm.generate.called

    def test_write_analytical_mode_succeeds(self) -> None:
        """'opinionforge write topic --mode analytical' succeeds with exit 0."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"

    def test_write_satirical_mode_succeeds(self) -> None:
        """'opinionforge write topic --mode satirical' succeeds with exit 0."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "satirical",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


# ---------------------------------------------------------------------------
# Modes command tests
# ---------------------------------------------------------------------------


class TestModesCommand:
    """Tests for the 'opinionforge modes' command."""

    def test_modes_lists_12_modes(self) -> None:
        """'opinionforge modes' lists 12 modes."""
        result = runner.invoke(app, ["modes"])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        # Check that all 12 mode IDs appear in the output
        expected_modes = [
            "polemical", "analytical", "populist", "satirical",
            "forensic", "oratorical", "narrative", "data_driven",
            "aphoristic", "dialectical", "provocative", "measured",
        ]
        for mode_id in expected_modes:
            assert mode_id in result.stdout, (
                f"Mode '{mode_id}' not listed in modes output"
            )

    def test_modes_category_confrontational(self) -> None:
        """'opinionforge modes --category confrontational' returns only confrontational modes."""
        result = runner.invoke(app, ["modes", "--category", "confrontational"])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        # Should contain confrontational modes
        assert "polemical" in result.stdout or "populist" in result.stdout

    def test_modes_detail_polemical(self) -> None:
        """'opinionforge modes --detail polemical' shows mode details without writer names."""
        result = runner.invoke(app, ["modes", "--detail", "polemical"])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert "polemical" in result.stdout.lower()
        # Must not contain any writer names
        writer_names = ["hitchens", "ivins", "buckley", "christopher", "molly"]
        for name in writer_names:
            assert name not in result.stdout.lower(), (
                f"Writer name '{name}' found in modes --detail output"
            )

    def test_modes_detail_nonexistent_exits_4(self) -> None:
        """'opinionforge modes --detail nonexistent' exits with code 4."""
        result = runner.invoke(app, ["modes", "--detail", "nonexistent_mode_xyz"])
        assert result.exit_code == 4

    def test_modes_exits_zero(self) -> None:
        """'opinionforge modes' exits with code 0."""
        result = runner.invoke(app, ["modes"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Deprecated flag error tests (sprint contract required)
# ---------------------------------------------------------------------------


class TestDeprecatedFlagErrors:
    """Tests for deprecated v0.2.0 flags returning exit code 2."""

    def test_voice_flag_exits_2(self) -> None:
        """'opinionforge write topic --voice hitchens' exits with code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--voice", "hitchens",
                "--no-preview",
            ])
        assert result.exit_code == 2, (
            f"Expected exit code 2 for deprecated --voice flag, got {result.exit_code}"
        )

    def test_spectrum_flag_exits_2(self) -> None:
        """'opinionforge write topic --spectrum 30' exits with code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--spectrum", "30",
                "--no-preview",
            ])
        assert result.exit_code == 2, (
            f"Expected exit code 2 for deprecated --spectrum flag, got {result.exit_code}"
        )

    def test_no_disclaimer_flag_exits_2(self) -> None:
        """'opinionforge write topic --no-disclaimer' exits with code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--no-disclaimer",
                "--no-preview",
            ])
        assert result.exit_code == 2, (
            f"Expected exit code 2 for removed --no-disclaimer flag, got {result.exit_code}"
        )


# ---------------------------------------------------------------------------
# Stance validation tests (sprint contract required)
# ---------------------------------------------------------------------------


class TestStanceValidation:
    """Tests for stance validation through the CLI."""

    def test_stance_101_exits_2(self) -> None:
        """'opinionforge write topic --stance 101' exits with code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--stance", "101",
                "--no-preview",
            ])
        assert result.exit_code == 2, (
            f"Expected exit code 2 for --stance 101, got {result.exit_code}"
        )

    def test_stance_minus_101_exits_2(self) -> None:
        """'opinionforge write topic --stance -101' exits with code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--stance", "-101",
                "--no-preview",
            ])
        assert result.exit_code == 2

    def test_nonexistent_mode_exits_4(self) -> None:
        """'opinionforge write topic --mode nonexistent' exits with code 4."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "nonexistent_mode_xyz",
                "--no-preview",
            ])
        assert result.exit_code == 4, (
            f"Expected exit code 4 for unknown mode, got {result.exit_code}"
        )


# ---------------------------------------------------------------------------
# Export format tests (sprint contract required)
# ---------------------------------------------------------------------------


class TestExportFormats:
    """Tests for export format flags through the CLI."""

    def test_substack_export_contains_disclaimer(self) -> None:
        """'opinionforge write topic --export substack' output contains the mandatory disclaimer."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "polemical",
                "--export", "substack",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert "AI-assisted rhetorical controls" in result.stdout or \
               "original content" in result.stdout, (
            "Disclaimer not found in substack export output"
        )

    def test_medium_export_succeeds(self) -> None:
        """'opinionforge write topic --export medium' succeeds with exit 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--export", "medium",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"

    def test_wordpress_export_succeeds(self) -> None:
        """'opinionforge write topic --export wordpress' succeeds with exit 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--export", "wordpress",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"

    def test_twitter_export_succeeds(self) -> None:
        """'opinionforge write topic --export twitter' succeeds with exit 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--export", "twitter",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


# ---------------------------------------------------------------------------
# Image prompt tests (sprint contract required)
# ---------------------------------------------------------------------------


class TestImagePromptFlag:
    """Tests for the --image-prompt flag."""

    def test_image_prompt_succeeds(self) -> None:
        """'opinionforge write topic --image-prompt' succeeds with mock LLM."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_GENERATED

        with patch("opinionforge.core.preview.create_llm_client", return_value=mock_llm), \
             patch("opinionforge.core.generator.create_llm_client", return_value=mock_llm), \
             patch("opinionforge.core.image_prompt.create_llm_client", return_value=mock_llm), \
             patch("opinionforge.core.research.get_search_client", return_value=MagicMock(search=lambda *a, **kw: [])):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--image-prompt",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


# ---------------------------------------------------------------------------
# Full pipeline text topic
# ---------------------------------------------------------------------------


class TestFullWritePipelineText:
    """Tests for the full write pipeline with text topic input."""

    def test_text_topic_through_full_pipeline(self) -> None:
        """Full write pipeline: text topic -> mode -> stance -> research -> generation -> output."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--stance", "0",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert "Algorithmic Threat" in result.stdout or "Democracy" in result.stdout
        assert mock_llm.generate.called

    def test_pipeline_with_no_research(self) -> None:
        """--no-research flag skips research and generates without sources."""
        with _full_pipeline_mocks() as (mock_llm, mock_search):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert mock_llm.generate.called
        assert not mock_search.search.called


class TestFullWritePipelineURL:
    """Tests for the full write pipeline with URL topic input."""

    def test_url_topic_through_pipeline(self) -> None:
        """Full pipeline with URL topic input (mocked httpx for URL fetch)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>AI is transforming democracy. This is a detailed article.</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        with _full_pipeline_mocks() as (mock_llm, _), \
             patch("opinionforge.core.topic.httpx.get", return_value=mock_response), \
             patch("trafilatura.extract", return_value="AI is transforming democracy. This is a detailed article."):
            result = runner.invoke(app, [
                "write",
                "--url", "https://example-news.com/ai-democracy",
                "--mode", "analytical",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


class TestFullWritePipelineFile:
    """Tests for the full write pipeline with file topic input."""

    def test_file_topic_through_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline with file topic input (temp file)."""
        topic_file = tmp_path / "topic.txt"
        topic_file.write_text(
            "The impact of artificial intelligence on democratic institutions "
            "is one of the most pressing issues of our time.",
            encoding="utf-8",
        )
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write",
                "--file", str(topic_file),
                "--mode", "analytical",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


# ---------------------------------------------------------------------------
# Preview command
# ---------------------------------------------------------------------------


class TestTonePreviewFlow:
    """Tests for the tone preview command."""

    def test_preview_mode_succeeds(self) -> None:
        """Tone preview with --mode and --stance parameters succeeds."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_PREVIEW

        with patch("opinionforge.core.preview.create_llm_client", return_value=mock_llm):
            result = runner.invoke(app, [
                "preview", "AI and democracy",
                "--mode", "analytical",
                "--stance", "-20",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert "Tone Preview" in result.stdout


# ---------------------------------------------------------------------------
# Blend syntax
# ---------------------------------------------------------------------------


class TestVoiceBlendingE2E:
    """Tests for mode blending through the full pipeline."""

    def test_blend_syntax_e2e(self) -> None:
        """Blend syntax parsed -> modes blended -> generation succeeds."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical:60,polemical:40",
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert mock_llm.generate.called


# ---------------------------------------------------------------------------
# Stance edge values
# ---------------------------------------------------------------------------


class TestStanceExtremes:
    """Tests for extreme stance values through the full pipeline."""

    @pytest.mark.parametrize("position", ["-100", "+100"])
    def test_extreme_stance(self, position: str) -> None:
        """Stance at extreme values (-100, +100) through full pipeline without errors."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--stance", position,
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


# ---------------------------------------------------------------------------
# Length presets
# ---------------------------------------------------------------------------


class TestLengthPresets:
    """Tests for length presets through the full pipeline."""

    @pytest.mark.parametrize("preset", ["short", "feature"])
    def test_length_preset_produces_output(self, preset: str) -> None:
        """Length presets through full pipeline: 'short' and 'feature' produce output."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--length", preset,
                "--no-preview",
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"


# ---------------------------------------------------------------------------
# Pipeline flags
# ---------------------------------------------------------------------------


class TestPipelineFlags:
    """Tests for various CLI flags through the pipeline."""

    def test_no_preview_flag(self) -> None:
        """--no-preview flag skips preview step."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
            ])
        assert result.exit_code == 0
        assert "Generate full piece?" not in result.stdout

    def test_output_flag_writes_to_file(self, tmp_path: Path) -> None:
        """--output flag writes generated piece to a temp file."""
        outfile = tmp_path / "output.md"
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
                "--output", str(outfile),
            ])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert outfile.exists()
        content = outfile.read_text(encoding="utf-8")
        assert "Algorithmic Threat" in content or "Democracy" in content

    def test_verbose_flag_produces_extra_output(self) -> None:
        """--verbose flag produces additional progress output."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
                "--verbose",
            ])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert "topic" in output_lower or "words" in output_lower or "domain" in output_lower

    def test_disclaimer_included_by_default(self) -> None:
        """Disclaimer is included in output by default."""
        with _full_pipeline_mocks() as (mock_llm, _):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
            ])
        assert result.exit_code == 0
        # The mandatory disclaimer must appear in the output
        assert "AI-assisted rhetorical controls" in result.stdout, (
            f"Mandatory disclaimer not found in output. Output: {result.stdout[:200]}"
        )


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    """Tests for error propagation through the CLI."""

    def test_invalid_mode_returns_exit_code_4(self) -> None:
        """Invalid mode ID through CLI returns exit code 4."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "nonexistent_mode_xyz",
                "--no-preview",
            ])
        assert result.exit_code == 4

    def test_serve_command_help(self) -> None:
        """'opinionforge serve --help' exits with code 0 and output contains --host and --port."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.stdout}"
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_missing_api_key_returns_exit_code_5(self) -> None:
        """Missing API key through CLI returns exit code 5.

        Patches create_llm_client to simulate the real behavior when
        no API key is configured (raises SystemExit(5)).
        """
        def _raise_no_key(settings=None):
            raise SystemExit(5)

        with patch("opinionforge.core.preview.create_llm_client", side_effect=_raise_no_key), \
             patch("opinionforge.core.generator.create_llm_client", side_effect=_raise_no_key):
            result = runner.invoke(app, [
                "write", "AI and democracy",
                "--mode", "analytical",
                "--no-preview",
                "--no-research",
            ])
        assert result.exit_code == 5, (
            f"Expected exit code 5, got {result.exit_code}: {result.stdout}"
        )
