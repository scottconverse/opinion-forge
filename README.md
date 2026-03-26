# OpinionForge

**An editorial craft engine for generating opinion pieces with precise rhetorical control.**

OpinionForge is an AI-powered tool for producing publication-ready opinion content — op-eds, columns, and long-form opinion pieces — using a system of 12 rhetorical modes with independently tunable argumentative stance and rhetorical intensity. It is not about imitating any particular person: it is about choosing a rhetorical approach and arguing it well.

Version: **2.0.0**

---

## What's New in v2.0.0

- **Multi-provider LLM support** — Anthropic, OpenAI, Ollama, and OpenAI-compatible endpoints. Switch providers at runtime via CLI flags or the settings UI
- **Local SQLite storage** — All generated pieces, exports, and settings persist automatically in a platform-appropriate data directory
- **History** — Browse, search, filter, and re-export past pieces through the web UI
- **Settings UI** — Configure provider, model, default mode, stance, intensity, theme, and other preferences from the browser
- **Desktop app mode** — System tray icon (via Pystray) with one-click server start and browser launch
- **First-run onboarding** — Guided setup flow for API key, provider selection, and default preferences
- **API key encryption** — Keys stored at rest are encrypted with the `cryptography` library

---

## Installation

```bash
pip install opinionforge
```

For the desktop tray icon (optional):

```bash
pip install opinionforge[desktop]
```

Configure your API key before using:

```bash
export ANTHROPIC_API_KEY=your-key-here
# or for OpenAI:
export OPENAI_API_KEY=your-key-here
export OPINIONFORGE_LLM_PROVIDER=openai
```

Requires Python 3.11+.

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

**Specify a provider and model:**

```bash
opinionforge write "Local journalism's future" --mode narrative --provider ollama --model llama3
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
- **History** — browse, search, and filter all past pieces at `/history`
- **Settings** — configure provider, model, defaults, and theme at `/settings`
- **Onboarding** — first-run guided setup for new users

The web UI uses the same environment variables as the CLI (`ANTHROPIC_API_KEY`, `OPINIONFORGE_LLM_PROVIDER`, etc.). No additional configuration is required.

---

## Desktop App

Install the desktop extras and launch from the system tray:

```bash
pip install opinionforge[desktop]
opinionforge desktop
```

The tray icon provides one-click access to start/stop the web server and open the browser. The app runs in the background until explicitly quit.

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
| `--provider` | — | LLM provider: `anthropic`, `openai`, `ollama`, `openai_compatible` |
| `--model` | — | Model identifier (overrides default) |
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

### `serve` — Start the web UI

```bash
opinionforge serve
opinionforge serve --port 9000
```

### `desktop` — Launch the desktop tray app

```bash
opinionforge desktop
```

### `config` — Show or modify configuration

```bash
opinionforge config
```

---

## LLM Providers

OpinionForge supports multiple LLM backends:

| Provider | Config Value | Requirements |
|----------|-------------|--------------|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Ollama | `ollama` | Local Ollama server running |
| OpenAI-compatible | `openai_compatible` | `OPENAI_API_KEY` + `OPINIONFORGE_BASE_URL` |

Set the provider via environment variable, CLI flag, or the settings UI:

```bash
# Environment variable
export OPINIONFORGE_LLM_PROVIDER=ollama

# CLI flag
opinionforge write "Topic" --provider ollama --model llama3

# Or configure in the web UI at /settings
```

---

## Storage

OpinionForge stores all data in a local SQLite database at the platform-standard location:

- **Linux**: `~/.local/share/opinionforge/opinionforge.db`
- **macOS**: `~/Library/Application Support/opinionforge/opinionforge.db`
- **Windows**: `%LOCALAPPDATA%/opinionforge/opinionforge.db`

The database stores:

- **Pieces** — every generated opinion piece with full metadata (topic, mode, stance, intensity, sources, screening results)
- **Exports** — export records linked to their parent pieces
- **Settings** — provider configuration, user preferences, and custom key-value pairs

API keys stored in settings are encrypted at rest using the `cryptography` library.

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

**4. Use Ollama with a local model:**

```bash
opinionforge write "Housing affordability" \
  --provider ollama \
  --model llama3 \
  --mode populist \
  --no-preview
```

---

## Configuration

Set via environment variables or a `.env` file in your working directory:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required when using Anthropic provider) |
| `OPENAI_API_KEY` | OpenAI API key (required when using OpenAI provider) |
| `OPINIONFORGE_LLM_PROVIDER` | LLM provider: `anthropic` (default), `openai`, `ollama`, `openai_compatible` |
| `OPINIONFORGE_BASE_URL` | Custom endpoint URL (for ollama or openai_compatible) |
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

The test suite contains 1322 fast tests plus 108 confusability tests (1430 total) covering unit tests, integration tests, end-to-end CLI tests, web UI tests, storage integration tests, provider tests, desktop tests, and confusability regression tests. All fast tests use mocked LLM and search clients — no real API calls. Confusability tests require an API key and are excluded by default.

---

## Architecture Overview

```
opinionforge/
├── __main__.py             # Entry point
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
├── providers/              # LLM backends (Anthropic, OpenAI, Ollama, OpenAI-compatible)
│   ├── base.py             # Abstract provider interface
│   ├── registry.py         # Provider discovery and instantiation
│   ├── anthropic.py
│   ├── openai_provider.py
│   ├── ollama.py
│   └── openai_compatible.py
├── storage/                # Local SQLite persistence
│   ├── database.py         # Connection manager and schema
│   ├── pieces.py           # Piece CRUD
│   ├── exports.py          # Export record CRUD
│   ├── settings.py         # Key-value settings store
│   └── encryption.py       # API key encryption
├── exporters/              # Platform exporters (Substack, Medium, WordPress, Twitter)
├── desktop/                # Desktop tray app (Pystray)
│   ├── tray.py             # System tray icon and menu
│   └── browser.py          # Browser launch helper
├── web/                    # FastAPI web UI (HTMX frontend, SSE streaming)
│   ├── app.py              # FastAPI application
│   ├── sse.py              # Server-sent events for streaming
│   ├── templates/          # Jinja2 templates (home, modes, history, settings, onboarding)
│   └── static/             # CSS, JS, icons
├── modes/
│   ├── profiles/           # 12 YAML mode profiles
│   └── categories.yaml     # Category assignments
├── models/                 # Pydantic data models
├── utils/                  # Utilities (fetcher, search, text processing)
└── data/                   # Suppressed phrases and structural fingerprints
```

---

## License

MIT License. See LICENSE for details.
