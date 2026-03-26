# OpinionForge

**An editorial craft engine for generating opinion pieces with precise rhetorical control.**

OpinionForge is an AI-powered tool for producing publication-ready opinion content — op-eds, columns, and long-form opinion pieces — using a system of 12 rhetorical modes with independently tunable argumentative stance and rhetorical intensity. It is not about imitating any particular person: it is about choosing a rhetorical approach and arguing it well.

Version: **1.0.0**

---

## What It Does

OpinionForge takes a topic (text, URL, or file) and generates an opinion piece in a selected rhetorical mode. You control three dimensions:

- **`--mode`** — The rhetorical approach (e.g., polemical, analytical, satirical)
- **`--stance`** — Argumentative emphasis direction from -100 (equity-focused) to +100 (liberty-focused)
- **`--intensity`** — Rhetorical heat from 0.0 (measured) to 1.0 (maximum conviction)

Every generated piece includes a mandatory disclaimer identifying it as AI-assisted content.

---

## Installation

```bash
pip install opinionforge
```

Configure your API key before using:

```bash
export ANTHROPIC_API_KEY=your-key-here
# or for OpenAI:
export OPENAI_API_KEY=your-key-here
export OPINIONFORGE_LLM_PROVIDER=openai
```

---

## Quick Start

**Generate an opinion piece:**

```bash
opinionforge write "The rise of algorithmic governance" --mode polemical --no-preview
```

**Choose a more measured analytical approach:**

```bash
opinionforge write "Climate policy tradeoffs" --mode analytical --stance -20 --intensity 0.4
```

**Use a satirical mode with strong conviction:**

```bash
opinionforge write "The latest tech regulation proposal" --mode satirical --intensity 0.9 --no-preview
```

---

## The 12 Rhetorical Modes

OpinionForge ships with 12 mode profiles organized into four rhetorical categories:

### Confrontational

| ID | Display Name | Description |
|----|-------------|-------------|
| `polemical` | Polemical | Combative moral urgency that names targets and demands the reader choose sides |
| `populist` | Populist | Ground-level clarity grounded in concrete human detail against abstract power |
| `provocative` | Provocative | Deliberate counter-intuitive claims designed to unsettle conventional thinking |

### Investigative

| ID | Display Name | Description |
|----|-------------|-------------|
| `forensic` | Forensic | Evidence-first argumentation that treats opinion as prosecutorial brief |
| `data_driven` | Data-Driven | Quantitative framing where statistics anchor every claim |

### Deliberative

| ID | Display Name | Description |
|----|-------------|-------------|
| `analytical` | Analytical | Structured logical argumentation with measured, authoritative tone |
| `dialectical` | Dialectical | Thesis-antithesis-synthesis reasoning that acknowledges genuine complexity |
| `measured` | Measured | Scoped, careful, acknowledges limits — the essayist's careful voice |

### Literary

| ID | Display Name | Description |
|----|-------------|-------------|
| `satirical` | Satirical | Irony, wit, and exaggeration to expose absurdity |
| `oratorical` | Oratorical | Speechmaking cadences and elevated register for maximum rhetorical effect |
| `narrative` | Narrative | Story-led opinion that arrives at argument through scene and detail |
| `aphoristic` | Aphoristic | Compressed, epigram-driven writing where each sentence carries maximum weight |

List all modes:

```bash
opinionforge modes
```

Filter by category:

```bash
opinionforge modes --category confrontational
```

View full details for a mode:

```bash
opinionforge modes --detail polemical
```

---

## Web UI

OpinionForge includes a browser-based web interface for interactive opinion piece generation.

### Starting the Web Server

```bash
opinionforge serve
```

The web UI is available at **http://127.0.0.1:8000** by default.

| Option | Default | Description |
|--------|---------|-------------|
| `--host`, `-h` | `127.0.0.1` | Host to bind the server to |
| `--port`, `-p` | `8000` | Port to bind the server to |

The web UI provides:

- **Mode browser** — browse all 12 rhetorical modes with full profile details at `/modes`
- **Topic input** — enter a topic, paste a URL, or upload a file
- **Stance and intensity controls** — interactive sliders for real-time adjustment
- **Generation with streaming progress** — SSE-powered progress indicators for each pipeline stage
- **Export** — export generated pieces to Substack, Medium, WordPress, or Twitter format

The web UI uses the same environment variables as the CLI (`ANTHROPIC_API_KEY`, `OPINIONFORGE_LLM_PROVIDER`, etc.). No additional configuration is required.

---

## CLI Reference

### `write` — Generate an opinion piece

```
opinionforge write [TOPIC] [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode`, `-m` | `analytical` | Rhetorical mode or blend (e.g., `polemical` or `polemical:60,narrative:40`) |
| `--stance`, `-s` | `0` | Argumentative direction: -100 (equity-focused) to +100 (liberty-focused) |
| `--intensity`, `-i` | `0.5` | Rhetorical heat: 0.0 (measured) to 1.0 (maximum conviction) |
| `--length`, `-l` | `standard` | Length preset: `short`, `standard`, `long`, `essay`, `feature` or word count |
| `--url` | — | Ingest topic from a URL |
| `--file`, `-f` | — | Ingest topic from a local file |
| `--no-preview` | — | Skip tone preview and generate immediately |
| `--no-research` | — | Skip source research |
| `--export` | — | Export format: `substack`, `medium`, `wordpress`, `twitter` |
| `--image-prompt` | — | Generate a header image prompt |
| `--output`, `-o` | — | Write output to a file |
| `--verbose` | — | Show research progress and generation details |

**Mode blending:**

```bash
opinionforge write "Immigration policy" --mode polemical:60,analytical:40 --no-preview
```

### `preview` — Generate a tone preview

```bash
opinionforge preview "Topic text" --mode satirical --stance 30
```

### `modes` — List rhetorical modes

```bash
opinionforge modes
opinionforge modes --category literary
opinionforge modes --detail analytical
```

### `config` — Show or modify configuration

```bash
opinionforge config
```

---

## Usage Examples

**1. Generate a forensic investigation-style piece:**

```bash
opinionforge write "Corporate lobbying and climate legislation" \
  --mode forensic \
  --stance -40 \
  --intensity 0.7 \
  --no-preview
```

**2. Export to Substack format:**

```bash
opinionforge write "The attention economy" \
  --mode analytical \
  --export substack \
  --no-preview
```

**3. Generate from a URL with image prompt:**

```bash
opinionforge write \
  --url "https://example-news.org/article" \
  --mode oratorical \
  --intensity 0.8 \
  --image-prompt \
  --no-preview
```

---

## Configuration

Set via environment variables or a `.env` file in your working directory:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required when using Anthropic provider) |
| `OPENAI_API_KEY` | OpenAI API key (required when using OpenAI provider) |
| `OPINIONFORGE_LLM_PROVIDER` | LLM provider: `anthropic` (default) or `openai` |
| `OPINIONFORGE_SEARCH_API_KEY` | Search API key for source research |
| `OPINIONFORGE_SEARCH_PROVIDER` | Search provider: `tavily` (default), `brave`, or `serpapi` |

---

## Mandatory Disclaimer

Every piece generated by OpinionForge includes this mandatory disclaimer:

> This piece was generated with AI-assisted rhetorical controls. It is original content and is not written by, endorsed by, or affiliated with any real person.

The disclaimer cannot be suppressed. This is by design.

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite (excluding slow LLM tests)
pytest tests/ -m "not slow"
```

The test suite contains 1098 tests (990 fast + 108 confusability) covering unit tests, integration tests, end-to-end CLI tests, web UI tests, and confusability regression tests. All fast tests use mocked LLM and search clients — no real API calls. Confusability tests require an API key and are excluded by default.

---

## Architecture Overview

```
opinionforge/
├── cli.py                  # Typer CLI application
├── config.py               # Settings (pydantic-settings)
├── core/
│   ├── generator.py        # Main LLM generation engine
│   ├── mode_engine.py      # Mode loading and blending
│   ├── stance.py           # Stance modifier
│   ├── preview.py          # Tone preview + LLM clients
│   ├── research.py         # Source research pipeline
│   ├── similarity.py       # Similarity screening
│   └── topic.py            # Topic ingestion (text/URL/file)
├── exporters/              # Platform exporters (Substack, Medium, WordPress, Twitter)
├── modes/
│   ├── profiles/           # 12 YAML mode profiles
│   └── categories.yaml     # Category assignments
├── models/                 # Pydantic data models
├── web/                    # FastAPI web UI (HTMX frontend, SSE streaming)
└── data/                   # Suppressed phrases and structural fingerprints
```

---

## License

MIT License. See LICENSE for details.
