"""Cross-module integration tests verifying OpinionForge components work together.

Tests internal component interactions without the CLI layer: topic ingestion,
mode assembly, prompt composition, research injection, and generation.
All LLM and network calls are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from opinionforge.core.generator import (
    MANDATORY_DISCLAIMER,
    compose_system_prompt,
    generate_piece,
    _parse_generated_output,
)
from opinionforge.core.topic import ingest_text
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece, SourceCitation
from opinionforge.models.topic import TopicContext
from opinionforge.utils.text import format_citations


# ---------------------------------------------------------------------------
# Shared mock LLM output
# ---------------------------------------------------------------------------

_MOCK_GENERATED = (
    "## The Crisis of Algorithmic Governance\n\n"
    "It would be comforting to believe that the machinery of self-governance "
    "is immune to the disruptions of artificial intelligence. Comforting, "
    "but dangerously naive. The evidence before us suggests that democratic "
    "institutions face a challenge unlike any they have encountered since "
    "the invention of the printing press.\n\n"
    "The first danger lies in AI systems generating misinformation at scale. "
    "Where once a propagandist required an army of scribes, a single algorithm "
    "can produce plausible falsehoods faster than any fact-checker can debunk.\n\n"
    "The deeper threat is subtler. When AI systems shape policy recommendations "
    "they encode biases into governance machinery. The democratic principle of "
    "equal voice is undermined when algorithms determine which voices are "
    "amplified and which suppressed.\n\n"
    "We must insist on transparency accountability and the primacy of human "
    "judgment over algorithmic convenience."
)


# ---------------------------------------------------------------------------
# Integration Tests: Topic -> Mode -> Prompt
# ---------------------------------------------------------------------------


class TestTopicToModeToPrompt:
    """Tests for topic ingestion -> mode assembly -> prompt composition."""

    def test_text_topic_to_mode_to_prompt(self) -> None:
        """Topic ingestion -> mode assembly -> prompt composition produces valid system prompt."""
        from opinionforge.core.mode_engine import blend_modes

        # 1. Ingest topic
        topic = ingest_text("The impact of artificial intelligence on democratic institutions")

        # 2. Load and blend modes
        mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
        mode_prompt = blend_modes(mode_config)

        # 3. Compose system prompt
        stance = StanceConfig(position=-20)
        system_prompt = compose_system_prompt(
            mode_prompt=mode_prompt, stance=stance, length_target=750,
        )

        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 200
        assert "OpinionForge" in system_prompt
        assert "Length Instructions" in system_prompt

    def test_blended_mode_prompt_composition(self) -> None:
        """Mode blending produces a coherent combined prompt."""
        from opinionforge.core.mode_engine import blend_modes

        mode_config = ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 40.0)])
        mode_prompt = blend_modes(mode_config)

        assert isinstance(mode_prompt, str)
        assert len(mode_prompt) > 50


# ---------------------------------------------------------------------------
# Integration Tests: Research Injection
# ---------------------------------------------------------------------------


class TestResearchInjection:
    """Tests for research results injection into the generation prompt."""

    def test_research_injected_into_prompt(self) -> None:
        """Research results are correctly injected into the generation prompt."""
        sources = [
            SourceCitation(
                claim="73 percent of agencies use AI tools.",
                source_name="The New York Times",
                source_url="https://www.nytimes.com/ai",
                accessed_at=datetime.now(timezone.utc),
                credibility_score=0.9,
            ),
        ]
        research_context = format_citations(sources)

        from opinionforge.core.mode_engine import blend_modes
        mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
        mode_prompt = blend_modes(mode_config)

        system_prompt = compose_system_prompt(
            mode_prompt=mode_prompt,
            stance=StanceConfig(position=0),
            length_target=750,
            research_context=research_context,
        )

        assert "Research Context" in system_prompt
        assert "73 percent" in system_prompt
        assert "nytimes.com" in system_prompt

    def test_citation_formatting_matches_prd(self) -> None:
        """Citation formatting from research results matches PRD format."""
        now = datetime(2025, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        sources = [
            SourceCitation(
                claim="AI tools are used by 73% of agencies.",
                source_name="The New York Times",
                source_url="https://www.nytimes.com/ai-democracy",
                accessed_at=now,
                credibility_score=0.9,
            ),
        ]

        formatted = format_citations(sources)
        assert '"AI tools are used by 73% of agencies."' in formatted
        assert "[The New York Times]" in formatted
        assert "(https://www.nytimes.com/ai-democracy)" in formatted
        assert "accessed 2025-03-15" in formatted


# ---------------------------------------------------------------------------
# Integration Tests: Generator Output
# ---------------------------------------------------------------------------


class TestGeneratorOutput:
    """Tests for generator output structure and content."""

    def test_generator_output_fields(self) -> None:
        """Generator output includes title, body, preview_text, disclaimer, and sources."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("The impact of AI on democratic institutions")
        mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
        stance = StanceConfig(position=0)

        piece = generate_piece(
            topic=topic,
            mode_config=mode_config,
            stance=stance,
            target_length=750,
            client=mock_client,
        )

        assert isinstance(piece, GeneratedPiece)
        assert piece.title  # Non-empty title
        assert piece.body  # Non-empty body
        assert piece.preview_text  # Non-empty preview
        assert piece.disclaimer == MANDATORY_DISCLAIMER
        assert isinstance(piece.sources, list)

    def test_word_count_within_tolerance(self) -> None:
        """Word count of generated output is within +/- 10% of target (controlled output)."""
        target = 750
        words = "word " * target
        controlled_output = f"## Test Title\n\n{words.strip()}"

        mock_client = MagicMock()
        mock_client.generate.return_value = controlled_output

        topic = ingest_text("Test topic for word count verification")
        piece = generate_piece(
            topic=topic,
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=target,
            client=mock_client,
        )

        tolerance = target * 0.10
        assert abs(piece.actual_length - target) <= tolerance, (
            f"Actual length {piece.actual_length} is not within 10% of target {target}"
        )

    def test_disclaimer_is_fixed_constant(self) -> None:
        """Disclaimer is the fixed mandatory constant, not writer-specific."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("Test topic for disclaimer check")
        piece = generate_piece(
            topic=topic,
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=500,
            client=mock_client,
        )

        assert piece.disclaimer == MANDATORY_DISCLAIMER
        assert "AI-assisted rhetorical controls" in piece.disclaimer
        assert "original content" in piece.disclaimer

    def test_screening_result_passed_for_clean_output(self) -> None:
        """Full pipeline with screening enabled produces a GeneratedPiece with screening_result.passed == True when mock output is clean."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("The impact of AI on democratic institutions")
        piece = generate_piece(
            topic=topic,
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=500,
            client=mock_client,
        )

        assert piece.screening_result is not None, (
            "GeneratedPiece must have a ScreeningResult after generation"
        )
        assert piece.screening_result.passed is True, (
            "Screening must pass for clean mock output"
        )

    def test_screening_failure_raises_runtime_error(self) -> None:
        """Full pipeline raises RuntimeError when mock screening returns passed=False."""
        from opinionforge.models.piece import ScreeningResult

        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        failing_result = ScreeningResult(
            passed=False,
            verbatim_matches=3,
            near_verbatim_matches=0,
            suppressed_phrase_matches=0,
            structural_fingerprint_score=0.0,
            rewrite_iterations=2,
            warning="Too many verbatim matches",
        )

        with patch("opinionforge.core.similarity.screen_output", return_value=failing_result):
            topic = ingest_text("The impact of AI on democratic institutions")
            with pytest.raises(RuntimeError) as exc_info:
                generate_piece(
                    topic=topic,
                    mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
                    stance=StanceConfig(position=0),
                    target_length=500,
                    client=mock_client,
                )
        assert "screening" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Integration Tests: Export Formats
# ---------------------------------------------------------------------------


class TestExportFormatsIntegration:
    """Tests for all 4 export formats end-to-end with mock LLM."""

    def _make_piece(self) -> GeneratedPiece:
        """Create a GeneratedPiece using the mock LLM for all export tests."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("AI and democratic governance")
        return generate_piece(
            topic=topic,
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=500,
            client=mock_client,
        )

    def test_substack_export_contains_disclaimer(self) -> None:
        """Substack export end-to-end with mock LLM contains the mandatory disclaimer."""
        from opinionforge.exporters import export

        piece = self._make_piece()
        output = export(piece, "substack")

        assert MANDATORY_DISCLAIMER in output, (
            "Substack export must contain the mandatory disclaimer"
        )

    def test_medium_export_contains_disclaimer(self) -> None:
        """Medium export end-to-end with mock LLM contains the mandatory disclaimer."""
        from opinionforge.exporters import export

        piece = self._make_piece()
        output = export(piece, "medium")

        assert MANDATORY_DISCLAIMER in output, (
            "Medium export must contain the mandatory disclaimer"
        )

    def test_wordpress_export_contains_disclaimer(self) -> None:
        """WordPress export end-to-end with mock LLM contains the mandatory disclaimer."""
        from opinionforge.exporters import export

        piece = self._make_piece()
        output = export(piece, "wordpress")

        assert MANDATORY_DISCLAIMER in output, (
            "WordPress export must contain the mandatory disclaimer"
        )

    def test_twitter_export_contains_disclaimer(self) -> None:
        """Twitter export end-to-end with mock LLM contains the mandatory disclaimer."""
        from opinionforge.exporters import export

        piece = self._make_piece()
        output = export(piece, "twitter")

        assert MANDATORY_DISCLAIMER in output, (
            "Twitter export must contain the mandatory disclaimer"
        )


# ---------------------------------------------------------------------------
# Integration Tests: Mode Blending
# ---------------------------------------------------------------------------


class TestModeBlendingIntegration:
    """Tests for 2-mode blending through the full pipeline."""

    def test_two_mode_blend_full_pipeline(self) -> None:
        """Mode blending (2-mode blend) succeeds through the full pipeline."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("AI and democratic governance")
        mode_config = ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 40.0)])

        piece = generate_piece(
            topic=topic,
            mode_config=mode_config,
            stance=StanceConfig(position=0),
            target_length=500,
            client=mock_client,
        )

        assert isinstance(piece, GeneratedPiece)
        assert piece.disclaimer == MANDATORY_DISCLAIMER
        assert piece.mode_config.modes == [("polemical", 60.0), ("analytical", 40.0)]


# ---------------------------------------------------------------------------
# Integration Tests: Stance Edge Values
# ---------------------------------------------------------------------------


class TestStanceEdgeValues:
    """Tests for stance at extreme values through the full pipeline."""

    @pytest.mark.parametrize("position", [-100, 100])
    def test_stance_at_extreme_values(self, position: int) -> None:
        """Stance at edge values (-100 and +100) through the full pipeline without errors."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("AI and democratic governance")
        mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
        stance = StanceConfig(position=position)

        piece = generate_piece(
            topic=topic,
            mode_config=mode_config,
            stance=stance,
            target_length=500,
            client=mock_client,
        )

        assert isinstance(piece, GeneratedPiece)
        assert piece.stance.position == position


# ---------------------------------------------------------------------------
# Integration Tests: Intensity Edge Values
# ---------------------------------------------------------------------------


class TestIntensityEdgeValues:
    """Tests for intensity at extreme values through the full pipeline."""

    @pytest.mark.parametrize("intensity", [0.0, 1.0])
    def test_intensity_at_edge_values(self, intensity: float) -> None:
        """Intensity at edge values (0.0 and 1.0) through the full pipeline without errors."""
        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("AI and democratic governance")
        mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
        stance = StanceConfig(position=0, intensity=intensity)

        piece = generate_piece(
            topic=topic,
            mode_config=mode_config,
            stance=stance,
            target_length=500,
            client=mock_client,
        )

        assert isinstance(piece, GeneratedPiece)
        assert piece.stance.intensity == intensity


# ---------------------------------------------------------------------------
# Integration Tests: Config Propagation
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    """Tests for config changes propagating correctly to LLM calls."""

    def test_provider_switch_propagates(self) -> None:
        """Config changes (provider switch) propagate correctly to LLM calls."""
        from opinionforge.config import Settings
        from opinionforge.core.preview import create_llm_client

        # Test Anthropic provider
        settings_anthropic = Settings(
            opinionforge_llm_provider="anthropic",
            anthropic_api_key="test-key-placeholder",
        )

        with patch("opinionforge.core.preview.AnthropicLLMClient") as mock_anthropic:
            client = create_llm_client(settings_anthropic)
            mock_anthropic.assert_called_once_with(api_key="test-key-placeholder")

        # Test OpenAI provider
        settings_openai = Settings(
            opinionforge_llm_provider="openai",
            openai_api_key="test-key-placeholder",
        )

        with patch("opinionforge.core.preview.OpenAILLMClient") as mock_openai:
            client = create_llm_client(settings_openai)
            mock_openai.assert_called_once_with(api_key="test-key-placeholder")


# ---------------------------------------------------------------------------
# Integration Tests: Research Pipeline
# ---------------------------------------------------------------------------


class TestResearchIntegration:
    """Tests for research pipeline integration with generation."""

    def test_research_into_generation(self) -> None:
        """Full flow: research results formatted and fed into generator."""
        from opinionforge.core.research import research_topic

        mock_search = MagicMock()
        mock_search.search.return_value = [
            MagicMock(
                url="https://www.nytimes.com/ai",
                title="AI and Democracy",
                snippet="73 percent of agencies use AI.",
                raw_content="According to a study, 73 percent of agencies use AI.",
            ),
        ]

        mock_fetcher = MagicMock(return_value=MagicMock(
            success=True,
            text="According to a study, 73 percent of agencies use AI tools.",
        ))

        topic = ingest_text("The impact of AI on democracy")
        stance = StanceConfig(position=0)

        result = research_topic(
            topic,
            stance,
            search_client=mock_search,
            fetcher=mock_fetcher,
            target_length=750,
        )

        assert len(result.sources) >= 1
        assert result.queries_used

        # Format and inject into system prompt
        from opinionforge.core.mode_engine import blend_modes
        mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
        mode_prompt = blend_modes(mode_config)
        context = format_citations(result.sources)
        system_prompt = compose_system_prompt(
            mode_prompt=mode_prompt,
            stance=stance,
            length_target=750,
            research_context=context,
        )
        assert "Research Context" in system_prompt


# ---------------------------------------------------------------------------
# Integration Tests: Output Parsing
# ---------------------------------------------------------------------------


class TestParsedOutput:
    """Tests for parsing generated output."""

    def test_parse_markdown_title(self) -> None:
        """Generated output with ## title is parsed correctly."""
        title, body = _parse_generated_output(
            "## My Great Title\n\nBody paragraph one.\n\nBody paragraph two."
        )
        assert title == "My Great Title"
        assert "Body paragraph one." in body

    def test_parse_hash_title(self) -> None:
        """Generated output with # title is parsed correctly."""
        title, body = _parse_generated_output(
            "# Single Hash Title\n\nBody content here."
        )
        assert title == "Single Hash Title"
        assert "Body content" in body

    def test_disclaimer_present_in_all_export_formats(self) -> None:
        """Mandatory disclaimer appears in all four export formats."""
        from opinionforge.exporters import export

        mock_client = MagicMock()
        mock_client.generate.return_value = _MOCK_GENERATED

        topic = ingest_text("AI and democratic governance")
        piece = generate_piece(
            topic=topic,
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=500,
            client=mock_client,
        )

        for fmt in ("substack", "medium", "wordpress", "twitter"):
            output = export(piece, fmt)
            assert MANDATORY_DISCLAIMER in output, (
                f"Mandatory disclaimer missing from {fmt} export"
            )


# ---------------------------------------------------------------------------
# Integration Tests: Web Pipeline Equivalence
# ---------------------------------------------------------------------------


class TestWebPipelineEquivalence:
    """Tests verifying the web UI generation path uses the same pipeline as the CLI."""

    def test_web_generate_calls_generate_piece_with_correct_config(self) -> None:
        """POST /generate with a given mode/stance/intensity calls generate_piece() with
        the equivalent ModeBlendConfig and StanceConfig that the CLI would construct."""
        from fastapi.testclient import TestClient
        from opinionforge.web.app import create_app

        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_GENERATED
        app = create_app(client=mock_llm)
        client = TestClient(app)

        captured_calls: list = []

        def mock_generate(**kwargs):
            captured_calls.append(kwargs)
            # Return a valid piece
            return generate_piece(
                topic=kwargs["topic"],
                mode_config=kwargs["mode_config"],
                stance=kwargs["stance"],
                target_length=kwargs.get("target_length", "standard"),
                research_context=kwargs.get("research_context"),
                client=mock_llm,
            )

        with patch("opinionforge.web.sse.generate_piece", side_effect=mock_generate):
            response = client.post(
                "/generate",
                data={
                    "topic": "AI and democracy",
                    "mode": "polemical:60,analytical:40",
                    "stance": "-30",
                    "intensity": "0.7",
                },
            )

        assert response.status_code == 200
        assert len(captured_calls) == 1

        call = captured_calls[0]
        # Verify ModeBlendConfig matches what CLI would construct
        assert call["mode_config"].modes == [("polemical", 60.0), ("analytical", 40.0)]
        # Verify StanceConfig matches
        assert call["stance"].position == -30
        assert call["stance"].intensity == 0.7

    def test_web_generate_runs_screening(self) -> None:
        """The web UI generation path runs screen_output() (screening is not bypassed)."""
        from fastapi.testclient import TestClient
        from opinionforge.web.app import create_app

        mock_llm = MagicMock()
        mock_llm.generate.return_value = _MOCK_GENERATED
        app = create_app(client=mock_llm)
        client = TestClient(app)

        with patch("opinionforge.core.similarity.screen_output") as mock_screen:
            from opinionforge.models.piece import ScreeningResult
            mock_screen.return_value = ScreeningResult(
                passed=True,
                verbatim_matches=0,
                near_verbatim_matches=0,
                suppressed_phrase_matches=0,
                structural_fingerprint_score=0.1,
                rewrite_iterations=0,
                warning=None,
            )
            response = client.post(
                "/generate",
                data={
                    "topic": "Screening pipeline test",
                    "mode": "analytical",
                    "stance": "0",
                    "intensity": "0.5",
                },
            )

        assert response.status_code == 200
        # screen_output should have been called during generation
        assert mock_screen.called, (
            "screen_output() was not called during web UI generation — "
            "screening is being bypassed"
        )
