# PRD: OpinionForge v1.0.0 (Risk-Reduction Refactor)

**PRD Version:** 4.0
**Last Updated:** 2026-03-25
**Product Version:** 1.0.0

**Revision History:**

| PRD Version | Date | Description |
|-------------|------|-------------|
| 1.0 | 2026-03-25 | Original Phase 1 MVP spec (PRD Writer output) |
| 2.0 | 2026-03-25 | Legal refactor rewrite — 12 rhetorical modes replace 100 named profiles |
| 3.0 | 2026-03-25 | 9 amendments from legal counsel: tightened safeguards, removed escape hatches, blocked output on screening failure, removed named-writer mappings from spec |
| 4.0 | 2026-03-25 | Amendment: Web UI added to v1.0.0 scope (Feature 9, Sprints 8-10) |

---

## Executive Summary

OpinionForge v1.0.0 is a risk-reduction refactor of the existing v0.2.0 editorial craft engine. It replaces 100 named writer voice profiles with 12 abstract rhetorical modes, renames the political spectrum slider to stance-and-intensity controls, makes disclaimers mandatory and non-removable, adds similarity screening and confusability testing, and removes all writer names from the public codebase. The product repositions from "voice cloning engine" to "editorial craft engine" -- better arguments, sharper rhetoric, stronger evidence, clearer structure.

## Problem Statement

OpinionForge v0.2.0 works well technically but creates legal and PR risk that blocks public release:

1. **Named writer profiles constitute identifiable voice emulation.** Files like `hitchens.yaml` and `buckley.yaml` with `system_prompt_fragment` fields engineered to reproduce a specific writer's style are evidence of intent to emulate real people, regardless of disclaimers.
2. **The public repo is evidence of intent.** 100 YAML files named after real writers, organized by category ("Mid-Century Titans," "Modern Heavyweights"), with fields like `signature_moves` and `few_shot_examples` quoting recognizable prose patterns.
3. **The success metric is inverted.** The v0.2.0 architecture implicitly measures "how recognizable is this as Writer X?" -- the legal standard requires the opposite: "how unlikely is this to be confused with Writer X?"
4. **The political spectrum slider (-100 to +100 labeled "far left" to "far right") is a PR risk.** Labels like `far_left`, `left`, `center`, `right`, `far_right` invite controversy and mischaracterize the tool's actual function, which is controlling argumentative emphasis and rhetorical heat.
5. **The `--no-disclaimer` flag allows suppression of attribution notices,** removing the only runtime safeguard against misuse.

Current solutions fail because no amount of disclaimer text fixes the fundamental architecture: the product is built around named-person emulation, and the codebase proves it.

## Target Users

- **Opinion writers and editors** at digital publications (Substack, Medium, WordPress) who want AI-assisted drafting with precise rhetorical control -- not a specific person's voice, but a specific rhetorical approach (polemical, analytical, satirical, etc.).
- **Content strategists and communications professionals** who need to produce persuasive editorial content with controllable argumentative intensity and evidence style.
- **Solo newsletter writers** who want to experiment with different rhetorical approaches to find their own editorial voice.
- **Academic and policy writers** who want to translate research into accessible opinion formats with appropriate rhetorical framing.

## Goals

- Materially reduce legal and reputational risk associated with named-person voice emulation, and remove named-person emulation from the shipped public release so v1.0.0 can be released publicly.
- Replace 100 named writer profiles with 12 abstract rhetorical modes that deliver equivalent rhetorical control without referencing any identifiable person
- Rename the political spectrum slider to stance-and-intensity controls that describe what the feature actually does (argumentative emphasis and rhetorical heat) without political alignment labels
- Make disclaimers mandatory on every output with no opt-out mechanism
- Add a similarity screening module that blocks verbatim or near-verbatim reuse of known source texts and suppresses signature catchphrases
- Add a confusability test suite that validates outputs stay below a confusability threshold for identifiable authors
- Achieve feature parity with v0.2.0 for all non-removed features: generation pipeline, research, export, image prompts, CLI
- Ship a product that positions as "editorial craft engine" in all user-facing text: CLI help, README, docs, landing page

## Non-Goals

- This is NOT a rewrite of the generation engine. The core pipeline (topic ingestion -> mode loading -> stance/intensity application -> preview -> research -> generation -> export) stays the same.
- This does NOT add new export formats beyond the existing four (Substack, Medium, WordPress, Twitter).
- This does NOT add a web UI. The product remains CLI-only.
- This does NOT change the LLM provider abstraction (Anthropic/OpenAI) or search provider (Tavily).
- This does NOT implement generation history storage (that remains a future feature).
- This does NOT add new length presets or change the word count system.
- This does NOT change the image prompt generator beyond removing any writer name references (already absent in v0.2.0).
- Internal research notes referencing writers by name may exist only in a private `research/` directory that is `.gitignore`-excluded, never shipped, never loaded at runtime, never referenced by tests, examples, fixtures, prompts, or generated artifacts, and never required for operation of the public release. This PRD does not govern internal research workflow except to prohibit any runtime or shipped dependency on those materials.

## Core Features

### Feature 1: Abstract Rhetorical Modes (replaces Named Writer Profiles)

The 100 named writer YAML profiles are replaced by 12 abstract rhetorical mode YAML profiles. Each mode is synthesized from the rhetorical patterns of multiple writers but references no writer by name anywhere in the shipped codebase.

**The 12 modes:**

| Mode ID | Display Name | Rhetorical Character | Internal Source Basis |
|---------|-------------|---------------------|-----------------------------------------------------|
| `polemical` | Polemical | Aggressive argumentation, moral clarity, rhetorical force, pointed rhetoric | Aggressive polemical traditions |
| `analytical` | Analytical | Evidence-driven, systems thinking, nuanced qualification, policy focus | Evidence-driven policy commentary traditions |
| `populist` | Populist | Street-level voice, vernacular authority, institutional skepticism, everyman framing | Urban populist column traditions |
| `satirical` | Satirical | Humor as weapon, ironic juxtaposition, absurdist exposure, comedic timing | Satirical opinion traditions |
| `forensic` | Forensic | Investigative rigor, document-driven, follow-the-money, systematic exposure | Document-driven investigative commentary traditions |
| `oratorical` | Oratorical | Elevated diction, rhetorical elegance, Latinate precision, epigrammatic summation | Classical rhetorical and oratorical traditions |
| `narrative` | Narrative | Story-first structure, scene-setting, character-driven, emotional arc | Immersive narrative journalism traditions |
| `data-driven` | Data-Driven | Statistics-forward, chart-logic prose, quantitative framing, empirical authority | Quantitative and empirical commentary traditions |
| `aphoristic` | Aphoristic | Compact wit, maxim construction, epigrammatic density, memorable one-liners | Epigrammatic and aphoristic traditions |
| `dialectical` | Dialectical | Thesis-antithesis structure, steelmanning, good-faith engagement, synthesis-seeking | Thesis-antithesis dialectical traditions |
| `provocative` | Provocative | Contrarian framing, deliberate disruption, convention-challenging, attention-forcing | Combined patterns from boundary-pushing voices |
| `measured` | Measured | Deliberative tone, balanced acknowledgment, careful qualification, bridge-building | Combined patterns from centrist, bridge-building voices |

**Mode profile structure** (replaces `VoiceProfile`):

Each mode YAML file contains the same structural fields as the current `VoiceProfile` minus the person-specific fields (`name`, `wikipedia_url`, `era`, `publication`, `ideological_baseline`). New fields are added for legal compliance.

```yaml
id: polemical
display_name: Polemical
description: "Aggressive argumentation with moral clarity and rhetorical force"
category: confrontational  # one of: confrontational, investigative, deliberative, literary

prose_patterns:
  avg_sentence_length: varied
  paragraph_length: medium
  uses_fragments: true
  uses_lists: false
  opening_style: provocative_declaration
  closing_style: moral_summation

rhetorical_devices:
  - moral_inversion
  - rhetorical_questions
  - accumulation
  - antithesis
  - devastating_qualifier

vocabulary_register:
  formality: formal
  word_origin_preference: mixed
  jargon_level: moderate
  profanity: rare
  humor_frequency: regular

argument_structure:
  approach: deductive
  evidence_style: mixed
  concession_pattern: brief_dismiss
  thesis_placement: first_paragraph

# NEW: No ideological_baseline. Modes are ideologically neutral.
# Stance is controlled entirely by --stance and --intensity flags.

signature_patterns:  # renamed from signature_moves -- no person reference
  - "Opening with a provocative moral declaration that forces the reader to take sides"
  - "Deploying rhetorical questions that expose the absurdity of the opposing position"
  - "Building accumulative evidence chains that create a sense of overwhelming proof"
  - "Closing with a compact moral judgment that crystallizes the entire argument"

# NEW: suppressed_phrases list for similarity screening
suppressed_phrases:
  - []  # populated during confusability testing with any phrases that trigger false positives

system_prompt_fragment: |
  Write with aggressive rhetorical force and moral clarity. Take firm
  positions and argue them with conviction. Deploy rhetorical questions,
  moral inversions, and devastating qualifiers. Open with a provocative
  declaration that frames the argument in moral terms. Build arguments
  through accumulation of evidence, each point reinforcing the thesis.
  Treat the opposing position not merely as wrong but as consequentially
  wrong. Close with a compact moral summation. The overall effect should
  be of a mind that has examined the evidence, reached a verdict, and
  delivers it with prosecutorial force.

few_shot_examples:
  - "The policy fails not because it is impractical -- impractical policies can be revised -- but because it is built on a premise so fundamentally dishonest that no amount of revision can rescue it from its own contradictions."
  - "Consider the arithmetic of this proposal, which its advocates would prefer you did not: for every dollar it promises to save, it requires three dollars of new obligation, a ratio that in any other context would be called what it is -- a confidence trick."
```

**Key constraints:**
- No writer name appears anywhere in any mode YAML file, in the `system_prompt_fragment`, in `few_shot_examples`, in comments, or in the filename.
- `few_shot_examples` are original compositions that demonstrate the rhetorical mode, NOT quotes from real writers.
- The `suppressed_phrases` field contains catchphrases identified during confusability testing that must be blocked from output.
- Mode files live at `opinionforge/modes/profiles/<mode_id>.yaml`.
- A `categories.yaml` file groups modes into 4 categories: `confrontational`, `investigative`, `deliberative`, `literary`.

### Feature 2: Stance and Intensity Controls (replaces Political Spectrum Slider)

The `--spectrum` flag (single integer from -100 to +100) is replaced by two separate controls:

- `--stance` (integer, -100 to +100): Controls argumentative emphasis direction. Negative values emphasize equity, collective action, systemic analysis. Positive values emphasize individual liberty, market mechanisms, institutional continuity. Zero is balanced.
- `--intensity` (float, 0.0 to 1.0, default 0.5): Controls rhetorical heat independently of direction. 0.0 = deliberative and measured. 1.0 = maximum conviction and force.

**Why two controls instead of one:** The v0.2.0 design couples direction and intensity into a single number where `abs(position)` determines intensity. This means you cannot write a strongly-worded centrist piece (stance=0, intensity=1.0) or a gently-worded progressive piece (stance=-60, intensity=0.2). Separating the controls gives users actual rhetorical control rather than a political alignment dial.

**Renamed internal concepts:**
- `SpectrumConfig` -> `StanceConfig` with fields `position: int` and `intensity: float`
- The `_direction_label()` function in `spectrum.py` (renamed to `stance.py`) removes political labels. New labels: "strongly equity-focused", "equity-leaning", "balanced", "liberty-leaning", "strongly liberty-focused"
- Source preference instructions remove named political publications and instead describe evidence categories: "peer-reviewed research, government data, investigative journalism" vs. "market analysis, institutional research, empirical case studies"
- The `ideological_baseline` field is removed from profiles entirely. Modes have no ideological default -- all ideological positioning comes from the user's `--stance` and `--intensity` flags.

**StanceConfig model:**

```python
class StanceConfig(BaseModel):
    position: int = Field(default=0, ge=-100, le=100)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def label(self) -> str:
        if self.position < -60:
            return "strongly_equity_focused"
        if self.position < -20:
            return "equity_leaning"
        if self.position <= 20:
            return "balanced"
        if self.position <= 60:
            return "liberty_leaning"
        return "strongly_liberty_focused"
```

### Feature 3: Mandatory Non-Removable Disclaimers

Every output includes the following disclaimer with no opt-out:

> This piece was generated with AI-assisted rhetorical controls. It is original content and is not written by, endorsed by, or affiliated with any real person.

**Implementation:**
- The `--no-disclaimer` CLI flag is removed entirely.
- The `disclaimer` field on `GeneratedPiece` is always populated with the above text. It is not configurable.
- All four exporters (Substack, Medium, WordPress, Twitter) include the disclaimer in their output. For Twitter, the disclaimer is appended as the final tweet in the thread.
- The disclaimer text is a constant, not constructed from writer names (eliminating the current `_build_disclaimer()` function that lists writer names).
- Tests assert that every code path producing output includes the disclaimer string.

### Feature 4: Similarity Screening Module

A new module `opinionforge/core/similarity.py` screens generated output before delivery to the user.

**What it screens for:**
1. **Verbatim reuse:** N-gram matching (n=5) against a library of known source texts. Any 5+ word verbatim match from the research context triggers a rewrite of that sentence.
2. **Near-verbatim reuse:** Normalized n-gram matching (lowercase, stripped punctuation, n=6) catches close paraphrases of source text.
3. **Signature catchphrase suppression:** A global `suppressed_phrases.yaml` file plus per-mode `suppressed_phrases` lists contain phrases that are too closely associated with identifiable writers. If any appear in the output, they are flagged for rewrite.
4. **Structural fingerprinting:** Detects if the sentence rhythm pattern (measured by syllable-count sequences) of a paragraph matches a known writer's signature pattern above a configurable threshold (default: 0.85 cosine similarity).

**Screening pipeline:**
```
generate_piece() -> raw_output
  -> similarity_screen(raw_output, research_sources, mode_config)
    -> check_verbatim(raw_output, research_texts)
    -> check_near_verbatim(raw_output, research_texts)
    -> check_suppressed_phrases(raw_output, suppressed_phrases)
    -> check_structural_fingerprint(raw_output, known_patterns)
  -> if violations found:
    -> rewrite flagged passages (single LLM call with rewrite instructions)
    -> re-screen rewritten passages (max 2 iterations)
  -> return screened_output
```

**Data files:**
- `opinionforge/data/suppressed_phrases.yaml`: Global suppressed phrases list.
- Each mode YAML has a `suppressed_phrases` field for mode-specific suppressions.
- `opinionforge/data/structural_fingerprints.yaml`: Known structural patterns to avoid (populated by confusability testing).

**Behavior on failure:** If screening cannot resolve a violation after 2 rewrite iterations, the output is blocked from public delivery and export. The system returns a screening failure with a summary of the violation type and prompts the user to regenerate or revise the request. In developer-only local test environments, a gated override may be available, but no such override exists in shipped builds.

### Feature 5: Confusability Test Suite

A new test module `tests/test_confusability.py` validates that outputs from each rhetorical mode stay below a confusability threshold for identifiable authors.

**How it works:**
1. For each of the 12 rhetorical modes, generate 3 sample pieces on standardized topics at different stance/intensity settings.
2. Run each sample through a confusability evaluation pipeline consisting of: (1) a prompted LLM judge, (2) a lexical/stylistic similarity scorer, and (3) a curated regression set of previously flagged outputs. A test fails if any evaluator indicates likely confusability with a specific identifiable author above threshold.
3. The test PASSES if no evaluator indicates confusability above threshold for any sample. The test FAILS if any sample triggers a confident identification by any evaluator.
4. Threshold: No evaluator may exceed the configured confusability threshold for any specific identifiable author. Initial default threshold is 0.6 for the LLM judge, with stricter thresholds established for auxiliary similarity checks during implementation.

**Test matrix:**
- 12 modes x 3 topics x 3 stance/intensity combos = 108 test cases
- Topics: "universal basic income," "artificial intelligence regulation," "urban housing policy" (standardized, politically contested topics that exercise the full rhetorical range)
- Stance/intensity combos: (0, 0.5), (-60, 0.8), (60, 0.8)

**Integration into CI:** These tests are marked `@pytest.mark.slow` and `@pytest.mark.confusability` so they can be run separately from the fast unit test suite. They require an LLM API key and are expected to run in CI on a scheduled basis (nightly), not on every commit.

**Success metric inversion:** The v0.2.0 implicit metric is "recognizability above threshold" (higher is better). The v1.0.0 metric is "confusability below threshold" (lower is better). This is the correct legal standard.

### Feature 6: Mode Blending (replaces Voice Blending)

The existing blend system (`--voice hitchens:60,ivins:40`) is preserved but adapted for modes:

- `--mode polemical:60,narrative:40` blends two rhetorical modes
- Maximum 3 modes in a blend (unchanged)
- Weights must sum to 100 (unchanged)
- The `BlendConfig` model is renamed to `ModeBlendConfig` but the validation logic is identical
- `blend_voices()` is renamed to `blend_modes()` with the same algorithm
- Single mode at 100% returns the mode's `system_prompt_fragment` unmodified (unchanged)

### Feature 7: CLI Refactor

All user-facing CLI text is updated to reflect the editorial craft engine positioning. No writer names appear anywhere in help text, error messages, or examples.

**Renamed flags:**
| v0.2.0 | v1.0.0 | Notes |
|--------|--------|-------|
| `--voice hitchens` | `--mode polemical` | Mode ID instead of writer name |
| `--voice hitchens:60,ivins:40` | `--mode polemical:60,narrative:40` | Blend syntax unchanged |
| `--spectrum -30` | `--stance -30` | Renamed, same range |
| (implicit from spectrum) | `--intensity 0.7` | New explicit control |
| `--no-disclaimer` | (removed) | No opt-out |

**Renamed commands:**
| v0.2.0 | v1.0.0 | Notes |
|--------|--------|-------|
| `opinionforge voices` | `opinionforge modes` | Lists available modes |
| `opinionforge voices --detail hitchens` | `opinionforge modes --detail polemical` | Mode detail view |

**Updated app help text:**
```
opinionforge - AI-powered editorial craft engine for generating opinion pieces
with precise rhetorical control.
```

### Feature 8: Codebase Sanitization

All references to real writer names are removed from the shipped codebase:

**Files removed:**
- All 100 files in `opinionforge/voices/profiles/` (replaced by 12 files in `opinionforge/modes/profiles/`)
- `opinionforge/voices/categories.yaml` (replaced by `opinionforge/modes/categories.yaml`)
- `opinionforge/voices/__init__.py` (replaced by `opinionforge/modes/__init__.py`)
- `docs/VOICES.md` (replaced by `docs/MODES.md`)

**Files renamed/refactored:**
- `opinionforge/core/voice.py` -> `opinionforge/core/mode_engine.py`
- `opinionforge/models/voice.py` -> `opinionforge/models/mode.py`
- `opinionforge/core/spectrum.py` -> `opinionforge/core/stance.py`
- `opinionforge/models/config.py`: `BlendConfig` -> `ModeBlendConfig`, `SpectrumConfig` -> `StanceConfig`

**Files updated:**
- `README.md`: Complete rewrite. No writer names. Positions as editorial craft engine.
- `docs/index.html`: Landing page rewrite. No writer names.
- `docs/terms.html`: Updated terms of service.
- `pyproject.toml`: Updated description, version to 1.0.0.
- All test files: Updated to use mode names instead of writer names.

**Grep verification:** Before any release, `grep -r` the entire repository for all 100 writer surnames. Zero matches required outside of `.gitignore`-excluded directories.

## CLI Interface

### Commands

```
opinionforge write TOPIC [OPTIONS]    # Generate an opinion piece
opinionforge preview TOPIC [OPTIONS]  # Generate a tone preview (2-3 sentences)
opinionforge modes [OPTIONS]          # List available rhetorical modes
opinionforge export PIECE_ID [OPTIONS] # Export a piece (future: requires history storage)
opinionforge config [OPTIONS]         # Show or modify configuration
```

### Options (write command)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode`, `-m` | `str` | `analytical` | Rhetorical mode or blend (e.g., `polemical` or `polemical:60,narrative:40`) |
| `--stance`, `-s` | `int` | `0` | Argumentative emphasis direction (-100 to +100) |
| `--intensity`, `-i` | `float` | `0.5` | Rhetorical heat (0.0 to 1.0) |
| `--length`, `-l` | `str` | `standard` | Length preset (short/standard/long/essay/feature) or word count |
| `--url` | `str` | `None` | Ingest topic from a URL |
| `--file`, `-f` | `Path` | `None` | Ingest topic from a local file |
| `--no-preview` | `bool` | `False` | Skip tone preview |
| `--research/--no-research` | `bool` | `True` | Enable/disable source research |
| `--output`, `-o` | `Path` | `None` | Write output to file |
| `--verbose` | `bool` | `False` | Show progress details |
| `--export` | `str` | `None` | Export format (substack/medium/wordpress/twitter) |
| `--image-prompt` | `bool` | `False` | Generate header image prompt |
| `--image-platform` | `str` | `substack` | Target platform for image dimensions |
| `--image-style` | `str` | `editorial` | Visual style for image prompt |

No user-facing or shipped CLI flag exists to bypass similarity screening. Local development bypass, if any, must be gated by a non-default developer environment variable unavailable in release builds and prohibited in CI, packaged distributions, and published examples.

### Options (modes command)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--search`, `-s` | `str` | `None` | Filter modes by search query |
| `--category`, `-c` | `str` | `None` | Filter by category (confrontational/investigative/deliberative/literary) |
| `--detail`, `-d` | `str` | `None` | Show full detail for a mode |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General / generation error |
| 2 | Invalid arguments |
| 3 | Network error |
| 4 | Mode not found |
| 5 | API key not configured |
| 6 | Rate limit exceeded |
| 7 | Content policy violation |
| 8 | Similarity screening failure (output blocked) |

## Data Models

### ModeProfile (replaces VoiceProfile)

```python
class ModeProfile(BaseModel):
    id: str                          # e.g., "polemical"
    display_name: str                # e.g., "Polemical"
    description: str                 # One-sentence description
    category: str                    # confrontational/investigative/deliberative/literary

    prose_patterns: ProsePatterns    # Unchanged from v0.2.0
    rhetorical_devices: list[str]    # Unchanged
    vocabulary_register: VocabularyRegister  # Unchanged
    argument_structure: ArgumentStructure    # Unchanged

    signature_patterns: list[str]    # Renamed from signature_moves; no person references
    suppressed_phrases: list[str]    # NEW: phrases to block from output
    system_prompt_fragment: str      # Unchanged structure, new content
    few_shot_examples: list[str]     # Original compositions, not real quotes
```

**Removed fields** (vs. VoiceProfile): `name`, `wikipedia_url`, `era`, `publication`, `ideological_baseline`.

### StanceConfig (replaces SpectrumConfig)

```python
class StanceConfig(BaseModel):
    position: int = Field(default=0, ge=-100, le=100)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
```

### ModeBlendConfig (replaces BlendConfig)

```python
class ModeBlendConfig(BaseModel):
    modes: list[tuple[str, float]]  # renamed from voices

    # Same validation: 1-3 modes, weights sum to 100
```

### GeneratedPiece (updated)

```python
class GeneratedPiece(BaseModel):
    # ... existing fields ...
    mode_config: ModeBlendConfig     # renamed from voice_config
    stance: StanceConfig             # renamed from spectrum
    disclaimer: str                  # always the fixed disclaimer string
    screening_result: ScreeningResult | None  # NEW
```

### ScreeningResult (new)

```python
class ScreeningResult(BaseModel):
    passed: bool
    verbatim_matches: int
    near_verbatim_matches: int
    suppressed_phrase_matches: int
    structural_fingerprint_score: float
    rewrite_iterations: int
    warning: str | None
```

## Architecture

The processing pipeline is the same as v0.2.0 with two additions (similarity screening, confusability testing) and renamed components:

```
CLI (Typer)
  |
  v
Topic Ingestion (unchanged)
  |
  v
Mode Loading (was: Voice Loading)
  - loads ModeProfile from opinionforge/modes/profiles/<id>.yaml
  - blend_modes() composes multi-mode fragments
  |
  v
Stance Application (was: Spectrum Application)
  - apply_stance() modifies the mode prompt with argumentative emphasis
  - uses separate position and intensity parameters
  |
  v
Tone Preview (unchanged pipeline, new mode prompts)
  |
  v
Source Research (unchanged)
  |
  v
Generation (compose_system_prompt -> LLM call)
  |
  v
*** NEW: Similarity Screening ***
  - check_verbatim()
  - check_near_verbatim()
  - check_suppressed_phrases()
  - check_structural_fingerprint()
  - rewrite if violations found (max 2 iterations)
  |
  v
Disclaimer Injection (mandatory, fixed text)
  |
  v
Export (unchanged: Substack, Medium, WordPress, Twitter)
  |
  v
Image Prompt Generation (unchanged, optional)
```

### Module Map

```
opinionforge/
  __init__.py
  __main__.py
  cli.py                    # Updated CLI with new flags
  config.py                 # Unchanged
  core/
    __init__.py
    generator.py            # Updated: uses ModeBlendConfig, StanceConfig
    image_prompt.py         # Unchanged
    length.py               # Unchanged
    mode_engine.py          # NEW (was voice.py): load_mode, blend_modes
    preview.py              # Unchanged pipeline
    research.py             # Unchanged
    similarity.py           # NEW: similarity screening module
    stance.py               # NEW (was spectrum.py): apply_stance
    topic.py                # Unchanged
  data/
    suppressed_phrases.yaml # NEW: global suppressed phrases
    structural_fingerprints.yaml  # NEW: known patterns to avoid
  exporters/
    __init__.py             # Unchanged
    base.py                 # Unchanged
    medium.py               # Unchanged
    substack.py             # Unchanged
    twitter.py              # Unchanged
    wordpress.py            # Unchanged
  modes/                    # NEW (replaces voices/)
    __init__.py             # list_modes, load_mode
    categories.yaml         # 4 categories
    profiles/
      analytical.yaml
      aphoristic.yaml
      data_driven.yaml
      dialectical.yaml
      forensic.yaml
      measured.yaml
      narrative.yaml
      oratorical.yaml
      polemical.yaml
      populist.yaml
      provocative.yaml
      satirical.yaml
  models/
    __init__.py
    config.py               # ModeBlendConfig, StanceConfig, ImagePromptConfig
    mode.py                 # NEW (was voice.py): ModeProfile, ProsePatterns, etc.
    piece.py                # Updated: ModeBlendConfig, StanceConfig, ScreeningResult
    topic.py                # Unchanged
  utils/
    __init__.py
    fetcher.py              # Unchanged
    search.py               # Unchanged
    text.py                 # Unchanged
```

## Tech Stack

- Language: Python 3.11+
- Framework: Typer for CLI, Pydantic v2 for models
- Build: hatchling
- Testing: pytest with pytest-asyncio, pytest-mock
- LLM: Anthropic Claude API (primary), OpenAI API (secondary)
- Search: Tavily API
- Content extraction: trafilatura
- HTTP: httpx

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| typer | >=0.9.0 | CLI framework |
| rich | >=13.0 | Terminal output formatting |
| pydantic | >=2.0 | Data validation and models |
| pydantic-settings | >=2.0 | Environment-based configuration |
| httpx | >=0.24 | HTTP client |
| trafilatura | >=1.6 | Web content extraction |
| anthropic | >=0.30 | Anthropic Claude API client |
| openai | >=1.0 | OpenAI API client |
| tavily-python | >=0.3 | Tavily search API client |
| pyyaml | >=6.0 | YAML profile loading |
| python-dotenv | >=1.0 | Environment variable loading |

No new runtime dependencies are required. The similarity screening module uses only stdlib and existing dependencies (n-gram matching is pure Python; structural fingerprinting uses basic math).

## Testing Strategy

### Unit Tests (fast, no API calls)

- **Mode loading:** All 12 mode YAML files parse into valid `ModeProfile` instances. All required fields present. No writer names in any field (automated grep test).
- **Mode blending:** Single-mode returns unmodified fragment. Multi-mode blending produces deterministic output. Weight validation. Max-3-mode limit.
- **Stance application:** Position and intensity produce correct instruction text. Direction labels are non-political. Edge cases: (0, 0.0), (-100, 1.0), (100, 1.0), (0, 1.0).
- **Disclaimer enforcement:** Every code path that produces output includes the fixed disclaimer string. No `--no-disclaimer` flag exists.
- **Similarity screening:** Verbatim detection catches 5+ word matches. Near-verbatim catches normalized matches. Suppressed phrases blocked. Structural fingerprint scored.
- **CLI argument parsing:** `--mode`, `--stance`, `--intensity` parse correctly. Old flags `--voice`, `--spectrum`, `--no-disclaimer` produce clear error messages pointing to new syntax.
- **Export:** All four exporters include disclaimer in output.
- **Name sanitization test:** Automated test that `grep -r` searches the entire `opinionforge/` directory and `tests/` directory for all 100 writer surnames from the v0.2.0 codebase. Zero matches required.
- **Runtime isolation test:** Public release builds must succeed without any files in `research/`. Tests assert that no module imports, file loads, prompt builders, fixtures, or examples depend on `.gitignore`-excluded research materials.

### Integration Tests (mock LLM, no API calls)

- **Full pipeline:** Topic -> mode loading -> stance -> preview -> research (mocked) -> generation (mocked) -> screening -> output. End-to-end with mock LLM client.
- **Screening integration:** Generated mock output with planted verbatim matches triggers rewrite. Screening with clean output passes through.

### Confusability Tests (slow, requires API key)

- **108 test cases** as described in Feature 5. Marked `@pytest.mark.slow` and `@pytest.mark.confusability`.
- **Run in CI nightly**, not on every commit.

### Coverage Target

- 90%+ line coverage for all modules except confusability tests.
- Every public function has at least one test.
- Every CLI flag has at least one test.
- Every exit code has at least one test.

### Edge Cases to Cover

- Mode blend where all weights are equal (e.g., 3 modes at 33.3/33.3/33.4)
- Stance at extremes with intensity at zero (should produce balanced output regardless of position)
- Intensity at 1.0 with stance at 0 (strong rhetoric, balanced framing)
- Empty topic string with `--url` flag
- Similarity screening with no research context (should still check suppressed phrases)
- Mode not found with close-match suggestions
- All 12 modes produce non-empty system prompt fragments

## Feature 9: Web UI

**Amendment (PRD v4.0):** The web UI ships as part of v1.0.0, not as a future phase.

### Stack

- **FastAPI** — backend API wrapping the existing generation pipeline
- **HTMX** — interactive frontend without a JavaScript framework
- **Jinja2** — server-side templates
- Zero additional JavaScript frameworks. HTMX handles all dynamic behavior.

### Pages and Routes

| Route | Page | Description |
|-------|------|-------------|
| `GET /` | Home | Mode browser, topic input, stance/intensity controls, generate button |
| `POST /generate` | (HTMX partial) | Streams generation progress, returns the finished piece |
| `GET /preview` | (HTMX partial) | Tone preview before committing to full generation |
| `GET /modes` | Mode catalog | Browse all 12 modes with descriptions, categories, rhetorical devices |
| `GET /modes/{mode_id}` | Mode detail | Full profile view for a single mode |
| `POST /export` | (HTMX partial) | Export the current piece to Substack/Medium/WordPress/Twitter format |
| `GET /about` | About | What OpinionForge is, how it works, disclaimer policy |

### Home Page Controls

The home page is the primary interface. It provides:

1. **Topic input** — text area, URL field, or file upload. Tabs for each input mode.
2. **Mode selector** — visual grid of 12 modes organized by category (confrontational, investigative, deliberative, literary). Click to select. Click a second mode to blend. Weights adjustable via sliders when blending.
3. **Stance slider** — horizontal slider from -100 to +100. Labels update dynamically: "strongly equity-focused" through "balanced" through "strongly liberty-focused." No political party labels.
4. **Intensity slider** — horizontal slider from 0.0 to 1.0. Labels: "measured" through "moderate" through "forceful" through "maximum conviction."
5. **Length selector** — dropdown with presets (short, standard, long, essay, feature) or custom word count input.
6. **Export format** — dropdown: plain text, Substack, Medium, WordPress, Twitter thread.
7. **Image prompt toggle** — checkbox to generate a header image prompt. When enabled, shows style selector (photorealistic, editorial, cartoon, minimalist, vintage, abstract) and platform selector (Substack, Medium, WordPress, Facebook, Twitter, Instagram).
8. **Generate button** — launches the pipeline. Disabled until a topic is provided.

### Generation Flow

1. User fills in topic and selects controls.
2. User clicks "Preview Tone" (optional) — HTMX call to `/preview`, returns a 2-3 sentence tone preview inline without page reload.
3. User clicks "Generate" — HTMX call to `/generate` with SSE streaming.
4. Progress indicators show: "Researching sources..." → "Generating piece..." → "Screening for originality..." → "Done."
5. Finished piece appears in a reading pane with title, body, sources, and disclaimer.
6. Export buttons appear below the piece.

### Disclaimer Policy in the Web UI

The disclaimer is included in every generated piece by default. The terms of service and about page clearly state that the disclaimer must be preserved in any published version of the output. The UI does not provide a button or toggle to remove it. However, once a user exports or copies the text, they have the output — OpinionForge states the requirement but does not enforce it beyond the application boundary. The responsibility for compliance with the disclaimer policy rests with the user after export.

This is the same approach as the CLI: the disclaimer is always present in the output. There is no --no-disclaimer flag. But OpinionForge is a tool, not a policeman.

### Screening in the Web UI

Similarity screening runs on every generation, same as CLI. If screening fails after 2 rewrite attempts, the UI displays a clear message: "This output did not pass originality screening. Please try regenerating with different settings or a revised topic." The blocked output is not shown to the user. This matches CLI exit code 8 behavior.

### Design Requirements

- Clean, editorial aesthetic consistent with the landing page (Georgia serif, cream/ink palette)
- Responsive — works on desktop and mobile
- No JavaScript frameworks — HTMX only
- Mode cards should show the mode's rhetorical character at a glance
- The stance and intensity sliders should show real-time label updates as the user drags
- Generated output should render markdown properly (headers, lists, blockquotes)
- Copy-to-clipboard button on the output pane
- Dark/light mode toggle (nice-to-have, not required for v1)

### API Configuration

The web UI reads the same environment variables as the CLI:

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
- `OPINIONFORGE_LLM_PROVIDER`
- `OPINIONFORGE_SEARCH_API_KEY`
- `OPINIONFORGE_SEARCH_PROVIDER`

An additional variable controls the web server:

- `OPINIONFORGE_HOST` — default `127.0.0.1`
- `OPINIONFORGE_PORT` — default `8000`

### CLI Integration

The web UI is launched via a new CLI command:

```
opinionforge serve [--host HOST] [--port PORT]
```

This starts the FastAPI server. The `write`, `preview`, `modes`, and `export` CLI commands continue to work independently.

### Testing

- API route tests using FastAPI's TestClient
- HTMX response tests (verify correct partial HTML responses)
- Integration test: generate via web UI with mock LLM, verify output includes disclaimer
- Integration test: screening failure returns error message, not blocked output
- All web UI tests use mock LLM/search clients — zero real API calls

---

## Version Roadmap

| Version | Phase | What Ships |
|---------|-------|------------|
| 0.1.0 | MVP (shipped, private) | Core generation engine, 10 writer profiles, spectrum slider, CLI basics |
| 0.2.0 | Phase 2 (shipped, private) | 100 writer profiles, voice blending, export formats, image prompts, research engine |
| 1.0.0 | Risk-reduction refactor + Web UI (this PRD) | 12 abstract rhetorical modes; stance/intensity controls; mandatory disclaimers; similarity screening; confusability testing; name sanitization; web UI with FastAPI + HTMX; editorial craft engine positioning |

- `0.x.x` = pre-release, private; API and interfaces may change between versions
- `1.0.0` = materially risk-reduced public release with stable API, mandatory disclosure, blocked screening failures, no shipped named-person emulation surface, and full web interface.

## Success Criteria

- **Zero writer names in shipped code.** Automated grep test passes: no matches for any of the 100 writer surnames in `opinionforge/`, `tests/`, `docs/`, `README.md`, or `pyproject.toml`.
- **Confusability test suite passes.** All 108 test cases score below 0.6 confidence for any specific author identification.
- **Similarity screening catches planted matches.** Unit tests with known verbatim passages achieve 100% detection rate for 5+ word matches.
- **Disclaimer present in all outputs.** No code path exists that produces user-facing text without the mandatory disclaimer.
- **Feature parity.** All v0.2.0 functionality (generation, preview, research, export, image prompts, blending) works with the new mode/stance interface.
- **CLI backward compatibility messaging.** Users of v0.2.0 flags (`--voice`, `--spectrum`, `--no-disclaimer`) receive clear error messages pointing to the new syntax.
- **All existing tests pass** (after updating for new interface names) with 90%+ coverage.
- **No public named-person mappings.** No shipped docs, tests, examples, config files, fixtures, comments, or help text contain mappings from rhetorical modes to specific real writers.

## Deliverables Checklist

- [ ] Source code with type hints and docstrings for all public functions
- [ ] 12 rhetorical mode YAML profiles with original (non-derivative) few-shot examples
- [ ] Similarity screening module (`opinionforge/core/similarity.py`)
- [ ] Stance/intensity module (`opinionforge/core/stance.py`)
- [ ] Mode engine module (`opinionforge/core/mode_engine.py`)
- [ ] Updated CLI with new flags and help text
- [ ] Updated exporters with mandatory disclaimer
- [ ] Suppressed phrases data file (`opinionforge/data/suppressed_phrases.yaml`)
- [ ] Structural fingerprints data file (`opinionforge/data/structural_fingerprints.yaml`)
- [ ] Name sanitization test (`tests/test_name_sanitization.py`)
- [ ] Confusability test suite (`tests/test_confusability.py`)
- [ ] Similarity screening tests (`tests/test_similarity.py`)
- [ ] Updated unit tests for all renamed modules
- [ ] Updated integration tests
- [ ] Test suite with 90%+ coverage
- [ ] README.md (complete rewrite, no writer names, editorial craft engine positioning)
- [ ] CLI help text on all commands/options
- [ ] pyproject.toml updated to version 1.0.0
- [ ] `docs/MODES.md` documenting all 12 rhetorical modes
- [ ] `docs/index.html` landing page (rewritten, no writer names)
- [ ] `.gitignore` updated to exclude `research/` directory (internal research notes)
- [ ] Web UI: FastAPI backend (`opinionforge/web/app.py`)
- [ ] Web UI: HTMX templates (`opinionforge/web/templates/`)
- [ ] Web UI: Static assets (`opinionforge/web/static/`)
- [ ] Web UI: `opinionforge serve` CLI command
- [ ] Web UI: Route tests with FastAPI TestClient
- [ ] Web UI: Integration tests (generate, preview, export, screening failure)
- [ ] Web UI: Disclaimer present in all web-generated output
- [ ] Web UI: Responsive layout (desktop + mobile)
