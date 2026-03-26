# OpinionForge — Extended Documentation

**Version 2.0.0** | **Python 3.11+** | **MIT License**

A local desktop editorial craft engine for generating publication-ready opinion pieces with precise rhetorical control. Runs in your browser, stores everything locally, connects to any LLM.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start Tutorial](#quick-start-tutorial)
- [Web UI](#web-ui)
- [LLM Providers](#llm-providers)
- [Local Storage](#local-storage)
- [Desktop App](#desktop-app)
- [CLI Reference](#cli-reference)
  - [write](#write--generate-an-opinion-piece)
  - [preview](#preview--generate-a-tone-preview)
  - [modes](#modes--list-rhetorical-modes)
  - [export](#export--export-a-previously-generated-piece)
  - [config](#config--show-or-modify-configuration)
- [The 12 Rhetorical Modes](#the-12-rhetorical-modes)
- [Mode Blending](#mode-blending)
- [Stance and Intensity](#stance-and-intensity)
- [Length Presets](#length-presets)
- [Topic Ingestion](#topic-ingestion)
- [Source Research Pipeline](#source-research-pipeline)
- [Similarity Screening](#similarity-screening)
- [Export Formats](#export-formats)
- [Image Prompt Generation](#image-prompt-generation)
- [Configuration Reference](#configuration-reference)
- [Architecture Overview](#architecture-overview)
- [API Reference](#api-reference)
- [Exit Codes](#exit-codes)
- [Troubleshooting](#troubleshooting)
- [Running Tests](#running-tests)
- [Mandatory Disclaimer](#mandatory-disclaimer)

---

## Overview

OpinionForge is an AI-powered editorial craft engine. It takes a topic (provided as text, URL, or file) and generates an opinion piece using a selected rhetorical mode. You control three independent dimensions:

- **Mode** -- the rhetorical approach (12 built-in modes across four categories)
- **Stance** -- argumentative emphasis direction from -100 (equity-focused) to +100 (liberty-focused)
- **Intensity** -- rhetorical heat from 0.0 (measured, deliberative) to 1.0 (maximum conviction)

OpinionForge is not a voice-cloning or writer-imitation tool. It uses abstract rhetorical modes that define argumentative structure, prose style, and rhetorical techniques without replicating the style or identity of any real individual.

Every generated piece includes a mandatory, non-removable disclaimer identifying it as AI-assisted content.

---

## Installation

```bash
pip install opinionforge
```

OpinionForge requires Python 3.11 or later.

### Dependencies

OpinionForge installs the following dependencies automatically:

- `typer>=0.9.0` -- CLI framework
- `rich>=13.0` -- Terminal formatting
- `pydantic>=2.0` -- Data validation
- `pydantic-settings>=2.0` -- Environment-based configuration
- `httpx>=0.24` -- HTTP client
- `trafilatura>=1.6` -- Article content extraction
- `anthropic>=0.30` -- Anthropic API client
- `openai>=1.0` -- OpenAI API client
- `tavily-python>=0.3` -- Tavily search API client
- `pyyaml>=6.0` -- YAML profile parsing
- `python-dotenv>=1.0` -- .env file loading
- `fastapi>=0.110` -- Web framework
- `uvicorn[standard]>=0.27` -- ASGI server
- `jinja2>=3.1` -- Template engine
- `sse-starlette>=1.6` -- Server-sent events for streaming
- `aiosqlite>=0.19` -- Async SQLite for local storage
- `platformdirs>=4.0` -- Platform-appropriate data directories
- `cryptography>=41.0` -- API key encryption at rest

Optional desktop extras (system tray icon):

```bash
pip install opinionforge[desktop]
```

- `pystray>=0.19` -- System tray integration
- `Pillow>=10.0` -- Tray icon rendering

### Development dependencies

```bash
pip install -e ".[dev]"
```

Development extras include pytest, pytest-asyncio, pytest-mock, ruff, mypy, and coverage.

### API Key Setup

Configure your LLM provider. You can do this via environment variables, a `.env` file, or through the web UI Settings page:

```bash
# For Anthropic (default cloud provider):
export ANTHROPIC_API_KEY=your-key-here

# For OpenAI:
export OPENAI_API_KEY=your-key-here
export OPINIONFORGE_LLM_PROVIDER=openai

# For Ollama (free, local, no API key needed):
export OPINIONFORGE_LLM_PROVIDER=ollama
```

For source research, optionally configure a search API key:

```bash
export OPINIONFORGE_SEARCH_API_KEY=your-tavily-key-here
```

Or skip environment variables entirely — the first-run onboarding wizard in the web UI will walk you through all of this.

---

## Quick Start Tutorial

### 1. Generate your first piece

```bash
opinionforge write "The rise of algorithmic governance" --mode polemical --no-preview
```

This generates a polemical opinion piece with default stance (balanced) and intensity (0.5), skipping the interactive tone preview.

### 2. Preview the tone before generating

```bash
opinionforge write "Climate policy tradeoffs" --mode analytical --stance -20 --intensity 0.4
```

Without `--no-preview`, OpinionForge first shows a 2-3 sentence tone preview and asks you to confirm before generating the full piece.

### 3. Use a standalone tone preview

```bash
opinionforge preview "The attention economy" --mode satirical --intensity 0.9
```

This produces only the tone preview (no full generation), useful for experimenting with modes and settings.

### 4. Blend two modes

```bash
opinionforge write "Immigration policy" --mode polemical:60,analytical:40 --no-preview
```

This produces a piece that is 60% polemical and 40% analytical. Up to three modes can be blended; weights must sum to 100.

### 5. Ingest from a URL

```bash
opinionforge write --url "https://example.com/article" --mode forensic --no-preview
```

OpinionForge fetches the article, extracts its content with trafilatura, and uses it as the topic.

### 6. Ingest from a file

```bash
opinionforge write --file ./my-article.txt --mode narrative --no-preview
```

### 7. Export to a platform format

```bash
opinionforge write "The state of journalism" --mode oratorical --export substack --no-preview
```

### 8. Generate with a header image prompt

```bash
opinionforge write "Tech regulation" --mode analytical --image-prompt --image-style photorealistic --image-platform medium --no-preview
```

### 9. Save output to a file

```bash
opinionforge write "Housing policy" --mode data_driven --output piece.md --no-preview
```

### 10. View all available modes

```bash
opinionforge modes
opinionforge modes --category literary
opinionforge modes --detail forensic
```

---

## CLI Reference

### `write` -- Generate an opinion piece

```
opinionforge write [TOPIC] [OPTIONS]
```

Runs the full pipeline: topic ingestion, mode loading, stance adjustment, tone preview, source research, and generation. Optionally exports to a platform format and generates an image prompt.

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `TOPIC` | No | Topic text for the opinion piece. Can be omitted when using `--url` or `--file`. |

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode` | `-m` | `analytical` | Rhetorical mode ID or blend (e.g., `polemical` or `polemical:60,narrative:40`) |
| `--stance` | `-s` | `0` | Argumentative emphasis direction: -100 (equity-focused) to +100 (liberty-focused) |
| `--intensity` | `-i` | `0.5` | Rhetorical heat: 0.0 (measured) to 1.0 (maximum conviction) |
| `--length` | `-l` | `standard` | Length preset (`short`, `standard`, `long`, `essay`, `feature`) or custom word count (200-8000) |
| `--url` | | | Ingest topic from a URL |
| `--file` | `-f` | | Ingest topic from a local file |
| `--no-preview` | | | Skip tone preview and generate immediately |
| `--research` / `--no-research` | | `--research` | Enable or disable source research |
| `--output` | `-o` | | Write output to a file instead of stdout |
| `--verbose` | | | Show research progress and generation details |
| `--export` | | | Export format: `substack`, `medium`, `wordpress`, `twitter` |
| `--image-prompt` | | | Generate a header image prompt (DALL-E / Midjourney / Stable Diffusion) |
| `--image-platform` | | `substack` | Target platform for image dimensions: `substack`, `medium`, `wordpress`, `facebook`, `twitter`, `instagram` |
| `--image-style` | | `editorial` | Visual style: `photorealistic`, `editorial`, `cartoon`, `minimalist`, `vintage`, `abstract` |

**Deprecated flags (produce clear error messages):**

- `--voice` -- removed; use `--mode` instead
- `--spectrum` -- removed; use `--stance` instead
- `--no-disclaimer` -- removed; the disclaimer is mandatory and cannot be suppressed

### `preview` -- Generate a tone preview

```
opinionforge preview [TOPIC] [OPTIONS]
```

Produces a 2-3 sentence preview in the selected rhetorical mode without performing full research or generation.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode` | `-m` | `analytical` | Rhetorical mode or blend |
| `--stance` | `-s` | `0` | Argumentative emphasis direction (-100 to +100) |
| `--intensity` | `-i` | `0.5` | Rhetorical heat (0.0 to 1.0) |
| `--url` | | | Ingest topic from a URL |
| `--file` | `-f` | | Ingest topic from a local file |

### `modes` -- List rhetorical modes

```
opinionforge modes [OPTIONS]
```

List and search available rhetorical mode profiles.

| Option | Short | Description |
|--------|-------|-------------|
| `--search` | `-s` | Filter modes by search query (searches ID, name, category, description) |
| `--category` | `-c` | Filter modes by category |
| `--detail` | `-d` | Show full detail for a specific rhetorical mode |

### `export` -- Export a previously generated piece

```
opinionforge export PIECE_ID --format FORMAT
```

This command is designed to export a piece from generation history. Generation history storage is a Phase 3 feature not yet implemented. The command currently displays guidance to use the `--export` flag on the `write` command instead.

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--format` | `-f` | Yes | Export format: `substack`, `medium`, `wordpress`, `twitter` |

### `config` -- Show or modify configuration

```
opinionforge config [OPTIONS]
```

Without options, displays all current configuration values with API keys masked. With `--set`, updates a configuration value (session-only; update your `.env` file for persistence).

| Option | Description |
|--------|-------------|
| `--set KEY` | Configuration key to set. Settable keys: `opinionforge_llm_provider`, `opinionforge_search_provider` |

API keys cannot be set via the CLI for security reasons. Set them in your `.env` file or as environment variables.

---

## Web UI

The web UI is the primary interface for OpinionForge. It runs locally in your browser — no terminal needed after the initial launch.

### Starting the Server

```bash
opinionforge serve
```

The web UI is available at **http://127.0.0.1:8000** by default.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | `-h` | `127.0.0.1` | Host to bind the web server to |
| `--port` | `-p` | `8000` | Port to bind the web server to |

### Web UI Pages

- **Home** (`/`) -- Generation form with topic input (text, URL, or file), mode selector grid, stance slider, intensity slider, length presets, export format selection, and image prompt toggle. Recent pieces sidebar and topic suggestions.
- **Mode browser** (`/modes`) -- All 12 rhetorical modes in a filterable grid organized by category, with links to individual mode detail pages
- **Mode detail** (`/modes/{id}`) -- Full profile for each mode including rhetorical devices, prose patterns, vocabulary register, argument structure, and signature patterns
- **History** (`/history`) -- Browse, search, filter, sort, and manage all generated pieces. Filter by mode, stance range, date range. Bulk select and delete. Click any piece to re-read or re-export.
- **Settings** (`/settings`) -- Configure LLM provider and model, enter API keys (encrypted at rest), set up search provider, customize default mode/stance/intensity/length/theme, export all data as JSON, clear history.
- **Setup** (`/setup`) -- First-run onboarding wizard: welcome, LLM provider configuration with connection test, search provider setup, quick tour, test generation.
- **About** (`/about`) -- Project overview, the 12-mode system, disclaimer policy.

### Generation Flow

When you submit a topic through the web form, OpinionForge streams progress via Server-Sent Events (SSE):

1. **Researching** -- Preparing research context
2. **Generating** -- Generating the opinion piece using the selected mode, stance, and intensity
3. **Screening** -- Running originality screening
4. **Done** -- Piece delivered with mandatory disclaimer, copy button, export buttons

If screening fails, an error event is returned instead of the blocked output. Generated pieces are automatically saved to the local database.

---

## LLM Providers

OpinionForge is provider-agnostic. Configure your provider in the Settings page or via environment variables.

| Provider | Type | Requirements |
|----------|------|-------------|
| Ollama | Local, free | Install Ollama, pull a model |
| Anthropic (Claude) | Cloud, per-token | `ANTHROPIC_API_KEY` |
| OpenAI (GPT-4o) | Cloud, per-token | `OPENAI_API_KEY` |
| OpenAI-compatible | Cloud/local, varies | `OPENAI_API_KEY` + base URL |

Switch providers at any time from the Settings page. All providers use the same rhetorical mode system.

---

## Local Storage

All data is stored in a local SQLite database at the platform-standard location:

- **Linux**: `~/.local/share/opinionforge/opinionforge.db`
- **macOS**: `~/Library/Application Support/opinionforge/opinionforge.db`
- **Windows**: `%LOCALAPPDATA%/opinionforge/opinionforge.db`

The database stores:

- **Pieces** -- every generated piece with full metadata (topic, mode, stance, intensity, sources, screening results)
- **Exports** -- export records linked to parent pieces
- **Settings** -- provider config, user preferences, custom key-value pairs

API keys are encrypted at rest using Fernet symmetric encryption.

---

## Desktop App

Install the desktop extras and launch from the system tray:

```bash
pip install opinionforge[desktop]
opinionforge desktop
```

The tray icon provides one-click access to start/stop the web server and open the browser. The app runs in the background until quit.

---

## The 12 Rhetorical Modes

OpinionForge ships with 12 rhetorical mode profiles organized into four categories.

### Confrontational

| ID | Display Name | Description |
|----|-------------|-------------|
| `polemical` | Polemical | Combative moral urgency that names targets and demands the reader choose sides |
| `populist` | Populist | Plain-spoken, ground-level clarity grounded in concrete human detail against abstract power |
| `provocative` | Provocative | Deliberately unsettling claims designed to unsettle conventional thinking |

### Investigative

| ID | Display Name | Description |
|----|-------------|-------------|
| `forensic` | Forensic | Evidence-first argumentation that treats opinion as a prosecutorial brief |
| `data_driven` | Data-Driven | Quantitative framing where statistics anchor every claim |

### Deliberative

| ID | Display Name | Description |
|----|-------------|-------------|
| `analytical` | Analytical | Rigorous evidence-first reasoning with measured, authoritative tone |
| `dialectical` | Dialectical | Thesis-antithesis-synthesis reasoning that acknowledges genuine complexity |
| `measured` | Measured | Calm, earned authority that conveys confidence through restraint and precision |

### Literary

| ID | Display Name | Description |
|----|-------------|-------------|
| `satirical` | Satirical | Irony, wit, and exaggeration to expose absurdity |
| `oratorical` | Oratorical | Grand, elevated classical rhetoric designed to be spoken aloud |
| `narrative` | Narrative | Story-led opinion that arrives at argument through scene and detail |
| `aphoristic` | Aphoristic | Compressed, epigram-driven writing where each sentence carries maximum weight |

Each mode profile defines:

- **Prose patterns** -- sentence length, paragraph length, fragment usage, opening and closing styles
- **Rhetorical devices** -- the specific devices the mode favors
- **Vocabulary register** -- formality, word origin preference, jargon level, profanity, humor frequency
- **Argument structure** -- approach (deductive/inductive/dialectical/narrative/mixed), evidence style, concession pattern, thesis placement
- **Signature patterns** -- distinctive stylistic habits unique to this mode
- **Suppressed phrases** -- phrases and constructions the mode avoids
- **System prompt fragment** -- the core prompt engineering text
- **Few-shot examples** -- short original-composition examples of the mode's style

Use `opinionforge modes --detail <mode-id>` for full profile details.

---

## Mode Blending

You can blend up to three modes using colon-separated weights that must sum to 100:

```bash
# 60% polemical, 40% analytical
opinionforge write "Topic" --mode polemical:60,analytical:40

# Three-way blend
opinionforge write "Topic" --mode forensic:50,narrative:30,measured:20
```

When a single mode is used without weights, it is treated as 100% of that mode. In blended mode, the dominant mode drives the overall structure and argument approach, while secondary modes contribute their rhetorical qualities in proportion to their weight.

---

## Stance and Intensity

Stance and intensity are independent controls that modify any rhetorical mode.

### Stance (--stance, -s)

The stance parameter controls argumentative emphasis direction on a scale from -100 to +100:

| Range | Label | Description |
|-------|-------|-------------|
| -100 to -50 | Strongly equity-focused | Foregrounds systemic analysis, collective solutions, structural reform |
| -49 to -25 | Equity-leaning | Emphasizes equity-based and systemic arguments |
| -24 to +25 | Balanced | Synthesizes both equity and liberty perspectives |
| +26 to +49 | Liberty-leaning | Emphasizes individual agency and market-based solutions |
| +50 to +100 | Strongly liberty-focused | Foregrounds individual agency, market mechanisms, institutional continuity |

Stance modifies five dimensions of the generated piece:

1. **Argument selection** -- which arguments receive primary emphasis
2. **Framing** -- how facts and statistics are contextualized
3. **Source preference** -- which evidence types are prioritized
4. **Rhetorical intensity** -- how forcefully arguments are made
5. **Counterargument handling** -- how opposing views are treated

### Intensity (--intensity, -i)

Intensity controls rhetorical heat independently of stance direction:

| Range | Description |
|-------|-------------|
| 0.0 - 0.19 | Measured, deliberative tone; acknowledges complexity |
| 0.2 - 0.49 | Moderate conviction; analytical confidence over polemical force |
| 0.5 - 0.79 | Strong conviction; vivid language and pointed rhetoric |
| 0.8 - 1.0 | Maximum conviction; aggressive rhetoric and uncompromising language |

---

## Length Presets

Use the `--length` flag with a preset name or a custom word count (200-8000):

| Preset | Word Count |
|--------|-----------|
| `short` | 500 |
| `standard` | 750 |
| `long` | 1200 |
| `essay` | 2500 |
| `feature` | 5000 |

The generator targets the specified word count within a 10% tolerance range. Structure guidance is adjusted based on the target length:

- **Short** (up to 800 words): Focus on a single strongest argument with minimal preamble
- **Standard** (up to 1200 words): Clear thesis, 2-3 supporting arguments, strong conclusion
- **Extended** (up to 2500 words): Multiple arguments with supporting evidence, substantive body sections
- **Long-form** (2500+ words): Section breaks, subheadings, deeply developed arguments, rich evidence

Custom word counts are accepted as integers:

```bash
opinionforge write "Topic" --length 1500 --no-preview
```

---

## Topic Ingestion

OpinionForge accepts topics through three input methods:

### Plain text

```bash
opinionforge write "The case for universal basic income"
```

### URL

```bash
opinionforge write --url "https://example.com/article"
```

The URL is fetched with httpx and article content is extracted using trafilatura. If the fetch fails, the URL itself is used as a topic reference.

### File

```bash
opinionforge write --file ./my-topic.txt
```

Reads a local text file and uses its contents as the topic.

### How ingestion works

For all input methods, OpinionForge produces a normalized `TopicContext` with:

- **Title** -- extracted from the first line or sentence
- **Summary** -- first 2-3 sentences
- **Key claims** -- sentences containing statistical or factual patterns
- **Key entities** -- capitalized multi-word sequences (basic named-entity recognition)
- **Subject domain** -- keyword-based classification (politics, economics, technology, health, environment, culture, education, foreign_policy, or general)

Source text is truncated to 10,000 words if it exceeds that limit.

---

## Source Research Pipeline

When `--research` is enabled (the default), OpinionForge conducts automated web research before generating:

1. **Query generation** -- produces at least 5-8 search queries covering factual background, recent developments, expert opinions, statistics, and counterarguments
2. **Web search** -- executes searches using the configured provider (Tavily, Brave Search, or SerpAPI)
3. **Source scoring** -- each result is scored on three dimensions:
   - **Relevance** (60% weight) -- keyword overlap with the topic
   - **Credibility** (40% weight) -- domain reputation from a built-in credibility map of 30+ known publications
   - **Spectrum weight** -- slight adjustment based on source political lean relative to the requested stance
4. **Content fetching** -- top results are fetched and processed with trafilatura
5. **Claim extraction** -- sentences containing statistics, studies, data, and expert references are extracted
6. **Citation formatting** -- sources are formatted with claim text, source name, URL, and access date

The minimum number of sources scales with target piece length: 3 for short op-eds (800 words or fewer), 8 for features (2500+ words), linearly interpolated between.

If research is disabled with `--no-research`, generation proceeds without source context.

---

## Similarity Screening

Every generated piece is screened for originality before delivery. The screening module runs four checks:

1. **Verbatim matching** -- 5-word n-gram matching against research source texts (exact tokens, case-sensitive)
2. **Near-verbatim matching** -- 6-word n-gram matching after text normalization (lowercase, punctuation stripped)
3. **Suppressed phrase matching** -- case-insensitive check against a global list of suppressed phrases (stored in `opinionforge/data/suppressed_phrases.yaml`) and per-mode suppressed phrases from each mode profile
4. **Structural fingerprint scoring** -- cosine similarity of the piece's sentence-level syllable-count sequence against known structural fingerprints (stored in `opinionforge/data/structural_fingerprints.yaml`). The blocking threshold is 0.85.

If violations are found and an LLM client is available, the screening module performs up to 2 automatic rewrite passes. Each pass instructs the LLM to rephrase only the flagged passages while preserving argument and structure.

If the piece still fails screening after rewrite attempts, generation is blocked (exit code 8).

---

## Export Formats

Use `--export` with the `write` command to produce platform-optimized output:

### Substack (`--export substack`)

Clean ATX-style markdown with no raw HTML tags. Includes title, body, sources appendix, mandatory disclaimer, and optional image prompt.

### Medium (`--export medium`)

Medium-compatible markdown with:

- `> DROP CAP` marker before the first paragraph
- Heading levels demoted by one to avoid conflicting with the piece title
- Pull quote blockquote for pieces over 300 words
- No raw HTML tags

### WordPress (`--export wordpress`)

Gutenberg block-editor markup (`<!-- wp:paragraph -->` / `<!-- /wp:paragraph -->`) with:

- Featured image placeholder comment
- SEO meta description comment (first 160 characters)
- Image prompt comment (when generated)
- `<h1>` title in a `wp:heading` block
- Body paragraphs each wrapped in `wp:paragraph` blocks
- Disclaimer with `opinionforge-disclaimer` CSS class

### Twitter/X (`--export twitter`)

Numbered tweet thread with:

- Tweet 1: title + opening hook
- Tweets 2 through N-1: key argument paragraphs
- Final tweet: mandatory disclaimer
- No tweet exceeds 280 characters (including the `N/` prefix)
- Thread length: 5 to 15 tweets
- Sentences are never split mid-word; breaks happen at sentence or clause boundaries

---

## Image Prompt Generation

Use `--image-prompt` with the `write` command to generate a header image prompt suitable for DALL-E, Midjourney, or Stable Diffusion.

The image prompt includes:

- A vivid, topic-specific subject description generated by the LLM
- Style directives for the chosen visual style
- Composition guidance (no text overlays)
- Aspect ratio and pixel dimensions for the target platform

### Image Styles

| Style | Description |
|-------|-------------|
| `photorealistic` | Documentary photograph style with natural lighting |
| `editorial` | Professional editorial art with bold composition |
| `cartoon` | Illustrated style with bold outlines and vibrant colors |
| `minimalist` | Clean composition with geometric simplicity and negative space |
| `vintage` | Retro aesthetic with aged texture and nostalgic color palette |
| `abstract` | Non-representational conceptual imagery |

### Image Platform Dimensions

| Platform | Aspect Ratio | Dimensions (px) |
|----------|-------------|-----------------|
| `substack` | 16:9 | 1456 x 819 |
| `medium` | 16:9 | 1400 x 788 |
| `wordpress` | 16:9 | 1200 x 675 |
| `facebook` | 1.91:1 | 1200 x 628 |
| `twitter` | 16:9 | 1600 x 900 |
| `instagram` | 1:1 | 1080 x 1080 |

---

## Configuration Reference

All configuration is via environment variables or a `.env` file in your working directory.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ANTHROPIC_API_KEY` | string | (none) | API key for Anthropic Claude models. Required when using the Anthropic provider. |
| `OPENAI_API_KEY` | string | (none) | API key for OpenAI models. Required when using the OpenAI provider. |
| `OPINIONFORGE_LLM_PROVIDER` | `anthropic` or `openai` | `anthropic` | Which LLM provider to use for generation. |
| `OPINIONFORGE_SEARCH_API_KEY` | string | (none) | API key for the web search provider. Required for source research. |
| `OPINIONFORGE_SEARCH_PROVIDER` | `tavily`, `brave`, or `serpapi` | `tavily` | Which search provider to use for source research. |

### LLM Models

- **Anthropic**: Uses `claude-sonnet-4-20250514` by default
- **OpenAI**: Uses `gpt-4o` by default

### Viewing current configuration

```bash
opinionforge config
```

This displays all settings with API keys masked (showing only the last 4 characters).

### Setting configuration values

```bash
opinionforge config --set opinionforge_llm_provider openai
```

Settable keys via CLI: `opinionforge_llm_provider`, `opinionforge_search_provider`. API keys must be set via environment variables or `.env` file for security.

Note: CLI config changes are session-only. For persistence, update your `.env` file.

---

## Architecture Overview

```
opinionforge/
├── __init__.py             # Package init, __version__ = "2.0.0"
├── __main__.py             # Entry point: python -m opinionforge
├── cli.py                  # Typer CLI application (5 commands)
├── config.py               # Settings via pydantic-settings
├── core/
│   ├── generator.py        # Main LLM generation engine
│   ├── mode_engine.py      # Mode loading, blending, fuzzy-match suggestions
│   ├── stance.py           # Stance modifier (5 dimensions)
│   ├── preview.py          # Tone preview + LLM client protocol/implementations
│   ├── research.py         # Source research pipeline
│   ├── similarity.py       # Similarity screening (4 checks + rewrite loop)
│   ├── length.py           # Length presets and validation
│   ├── topic.py            # Topic ingestion (text/URL/file)
│   └── image_prompt.py     # Image prompt generation
├── exporters/
│   ├── base.py             # Abstract base exporter
│   ├── substack.py         # Substack markdown exporter
│   ├── medium.py           # Medium markdown exporter (drop cap, pull quotes)
│   ├── wordpress.py        # WordPress Gutenberg block markup exporter
│   └── twitter.py          # Twitter/X numbered thread exporter
├── modes/
│   ├── __init__.py         # load_mode() and list_modes()
│   ├── categories.yaml     # Category-to-mode mapping
│   └── profiles/           # 12 YAML mode profiles
│       ├── analytical.yaml
│       ├── aphoristic.yaml
│       ├── data_driven.yaml
│       ├── dialectical.yaml
│       ├── forensic.yaml
│       ├── measured.yaml
│       ├── narrative.yaml
│       ├── oratorical.yaml
│       ├── polemical.yaml
│       ├── populist.yaml
│       ├── provocative.yaml
│       └── satirical.yaml
├── models/
│   ├── config.py           # ModeBlendConfig, StanceConfig, ImagePromptConfig
│   ├── mode.py             # ModeProfile, ProsePatterns, VocabularyRegister, ArgumentStructure
│   ├── piece.py            # GeneratedPiece, ScreeningResult, SourceCitation
│   └── topic.py            # TopicContext
├── providers/              # LLM provider backends
│   ├── base.py             # Abstract provider interface (LLMProvider protocol)
│   ├── registry.py         # Provider discovery, instantiation, connection testing
│   ├── anthropic.py        # Anthropic Claude adapter
│   ├── openai_provider.py  # OpenAI adapter
│   ├── ollama.py           # Ollama local LLM adapter
│   └── openai_compatible.py # OpenAI-compatible endpoint adapter
├── storage/                # Local SQLite persistence
│   ├── database.py         # Connection manager, schema, migrations
│   ├── pieces.py           # Piece CRUD (save, get, list, search, filter, delete)
│   ├── exports.py          # Export record CRUD
│   ├── settings.py         # Key-value settings + typed ProviderConfig/UserPreferences
│   └── encryption.py       # API key encryption at rest (Fernet)
├── desktop/                # Desktop tray app (optional)
│   ├── tray.py             # System tray icon and menu (Pystray)
│   └── browser.py          # Browser launch helper
├── web/
│   ├── app.py              # FastAPI application factory (create_app)
│   ├── sse.py              # SSE streaming helper for generation pipeline
│   ├── templates/          # Jinja2 templates (home, modes, history, settings, setup, about, partials)
│   └── static/             # CSS, JS, and static assets
├── utils/
│   ├── fetcher.py          # URL fetching with httpx + trafilatura extraction
│   ├── search.py           # Search clients (Tavily, Brave, SerpAPI)
│   └── text.py             # word_count, truncate_text, format_citations
└── data/
    ├── suppressed_phrases.yaml       # Global suppressed catchphrases
    └── structural_fingerprints.yaml  # Structural fingerprint definitions
```

### Generation Pipeline

The `write` command executes these steps in order:

1. **Input validation** -- mode blend, stance, intensity, length, export format, image config
2. **Topic ingestion** -- normalize text/URL/file into a TopicContext
3. **Mode loading** -- load YAML profiles, blend if multiple modes
4. **Stance modification** -- append stance-aware instructions to mode prompt
5. **Tone preview** (unless `--no-preview`) -- generate 2-3 sentence preview, prompt for confirmation
6. **Source research** (unless `--no-research`) -- multi-query web search, fetch, score, extract claims
7. **Generation** -- compose system prompt, call LLM, parse output into title + body
8. **Similarity screening** -- check for verbatim/near-verbatim/suppressed phrases/structural fingerprints; rewrite if violations found
9. **Image prompt generation** (if `--image-prompt`) -- generate visual prompt using LLM
10. **Output** -- format with disclaimer, optionally export to platform format, write to stdout or file

---

## API Reference

### Models

All models are Pydantic BaseModel subclasses available from `opinionforge.models`:

**`TopicContext`** -- normalized topic representation with title, summary, key_claims, key_entities, subject_domain, and optional source_url, source_text, fetched_at.

**`ModeBlendConfig`** -- holds a list of (mode_id, weight) tuples. Validates: at least 1 mode, at most 3, weights sum to 100.

**`StanceConfig`** -- position (-100 to +100) and intensity (0.0 to 1.0). The `label` property returns a non-political descriptor (e.g., "balanced", "equity_leaning").

**`ImagePromptConfig`** -- style, platform, optional custom_keywords. Properties: `aspect_ratio`, `dimensions`.

**`ModeProfile`** -- full profile with id, display_name, description, category, prose_patterns, rhetorical_devices, vocabulary_register, argument_structure, signature_patterns, suppressed_phrases, system_prompt_fragment, few_shot_examples.

**`GeneratedPiece`** -- the complete output: id, created_at, topic, mode_config, stance, target/actual length, title, body, preview_text, sources, research_queries, image_prompt, image_platform, disclaimer, screening_result.

**`ScreeningResult`** -- passed (bool), verbatim_matches, near_verbatim_matches, suppressed_phrase_matches, structural_fingerprint_score, rewrite_iterations, warning.

**`SourceCitation`** -- claim, source_name, source_url, accessed_at, political_lean, credibility_score, recency_score.

### LLM Client Protocol

The `LLMClient` protocol (in `opinionforge.core.preview`) defines a single method:

```python
def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str
```

Two implementations are provided:

- `AnthropicLLMClient` -- uses `claude-sonnet-4-20250514`
- `OpenAILLMClient` -- uses `gpt-4o`

Both accept dependency injection for testing.

### Key Functions

- `opinionforge.core.generator.generate_piece()` -- orchestrates full generation
- `opinionforge.core.mode_engine.blend_modes()` -- produces blended prompt from ModeBlendConfig
- `opinionforge.core.stance.apply_stance()` -- modifies mode prompt with stance instructions
- `opinionforge.core.research.research_topic()` -- conducts multi-query research
- `opinionforge.core.similarity.screen_output()` -- screens output for originality
- `opinionforge.core.topic.ingest_text()`, `ingest_url()`, `ingest_file()` -- topic normalization
- `opinionforge.core.image_prompt.generate_image_prompt()` -- generates image prompts
- `opinionforge.exporters.export()` -- dispatches to platform exporters

---

## Exit Codes

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
| 8 | Similarity screening failure |

---

## Troubleshooting

### "ANTHROPIC_API_KEY is not set"

You need to configure your LLM API key. Set it as an environment variable or in a `.env` file:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

### "OPINIONFORGE_SEARCH_API_KEY is not set"

Source research requires a search API key. Set it or disable research:

```bash
export OPINIONFORGE_SEARCH_API_KEY=your-tavily-key-here
# or skip research:
opinionforge write "Topic" --no-research --no-preview
```

### "Mode 'X' not found"

The mode ID does not match any installed profile. OpinionForge provides fuzzy-match suggestions. Check available modes:

```bash
opinionforge modes
```

### "Blend weights must sum to 100"

When blending modes, weights must total exactly 100:

```bash
# Correct:
--mode polemical:60,analytical:40

# Wrong (sums to 90):
--mode polemical:50,analytical:40
```

### "Word count X is outside the allowed range"

Custom word counts must be between 200 and 8000.

### "--voice is not a valid option"

The `--voice` flag was removed in v1.0.0. Use `--mode` instead.

### "--spectrum is not a valid option"

The `--spectrum` flag was removed in v1.0.0. Use `--stance` instead.

### "--no-disclaimer is not a valid option"

The disclaimer is mandatory in v1.0.0 and cannot be suppressed.

### "Similarity screening failed -- output blocked"

The generated piece failed originality screening after automatic rewrite attempts. Try regenerating with a different mode, stance, or topic phrasing.

### Generation is slow

The full pipeline (research + generation) requires multiple API calls. To speed things up:

- Use `--no-research` to skip web research
- Use `--no-preview` to skip the tone preview step

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite (excluding slow LLM tests)
pytest tests/ -m "not slow"
```

The test suite contains 990 tests covering:

- Unit tests for all core modules
- Topic ingestion and normalization
- Mode loading and blending
- Length validation and resolution
- Stance modification
- Similarity screening
- Export formatting
- Search client adapters
- Text utility functions
- CLI integration tests
- Web UI integration, E2E, and HTMX tests

All tests use mocked LLM and search clients -- no real API calls are made during testing.

Test markers:

- `slow` -- marks tests that make real API calls (excluded by default)
- `confusability` -- marks tests in the confusability regression suite

---

## Mandatory Disclaimer

Every piece generated by OpinionForge includes this mandatory disclaimer:

> This piece was generated with AI-assisted rhetorical controls. It is original content and is not written by, endorsed by, or affiliated with any real person.

The disclaimer cannot be suppressed. The `--no-disclaimer` flag was removed in v1.0.0 and produces an error if used. This policy exists because OpinionForge generates original AI content with defined rhetorical character, and readers deserve to know what they are reading.
