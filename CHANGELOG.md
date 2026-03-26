# Changelog

All notable changes to OpinionForge are documented in this file.

---

## [2.0.0] - 2026-03-26

Local desktop application with pluggable LLM backends. OpinionForge now runs entirely on the user's machine and connects to whatever LLM the user already has.

### Added

- **Pluggable LLM providers**: Ollama (local, free), Anthropic Claude, OpenAI, and any OpenAI-compatible endpoint (LM Studio, vLLM, etc.)
- **Provider registry** with auto-detection (finds running Ollama instances), connection testing, and model listing
- **Local SQLite database** for persistent storage of generated pieces, settings, and export history
- **First-run onboarding wizard** -- 5-step setup: welcome, LLM provider config with connection test, search provider setup, quick tour, test generation
- **History page** -- browse, search, filter, sort, and re-export all generated pieces
- **Settings page** -- LLM provider config, search provider config, user preferences, data export, history management
- **API key encryption** at rest using Fernet symmetric encryption with machine-specific key
- **Desktop integration** -- system tray icon (optional, via pystray), browser auto-open on launch, port fallback (8484-8494)
- **`opinionforge` bare command** launches the web UI (no subcommand needed)
- **`opinionforge config`** reads from SQLite storage with `--provider`/`--model` flags
- **Home page improvements** -- mode cards with descriptions and category colors, slider tooltips, help icons, recent pieces sidebar, topic suggestions
- **HX-Request detection** -- routes return partials for HTMX requests, full pages for direct navigation
- **SSE generation** renders full `piece_result.html` template (sources, copy button, export buttons)

### Changed

- Architecture changed from hosted web server to local desktop application
- LLM calls are now provider-agnostic via the `LLMProvider` protocol
- Version bumped to 2.0.0
- Default port changed from 8000 to 8484

### Security

- API keys encrypted at rest in local SQLite database
- API keys displayed masked (last 4 characters only) in settings UI
- No data leaves the user's machine except LLM API calls to their chosen provider

---

## [1.0.0] - 2026-03-25

Initial public release. OpinionForge v1.0.0 is a complete rewrite from the private v0.2.0 prototype, redesigned around abstract rhetorical modes rather than named writer profiles.

### Added

- **12 rhetorical modes** organized into four categories:
  - Confrontational: polemical, populist, provocative
  - Investigative: forensic, data_driven
  - Deliberative: analytical, dialectical, measured
  - Literary: satirical, oratorical, narrative, aphoristic
- **Mode blending** -- combine up to 3 modes with weighted percentages (e.g., `polemical:60,analytical:40`)
- **Stance control** (`--stance`) -- argumentative emphasis direction from -100 (equity-focused) to +100 (liberty-focused), modifying five independent dimensions: argument selection, framing, source preference, rhetorical intensity, and counterargument handling
- **Intensity control** (`--intensity`) -- rhetorical heat from 0.0 (measured) to 1.0 (maximum conviction), independent of stance direction
- **Five length presets** (`--length`): short (500), standard (750), long (1200), essay (2500), feature (5000), plus custom word counts from 200-8000
- **Topic ingestion** from three sources: plain text, URL (via httpx + trafilatura), and local files
- **Source research pipeline** with multi-query search, domain credibility scoring, political lean tagging, relevance scoring, recency scoring, and claim extraction
- **Three search provider adapters**: Tavily, Brave Search, SerpAPI
- **Two LLM provider integrations**: Anthropic (Claude Sonnet) and OpenAI (GPT-4o)
- **Similarity screening** with four checks: verbatim n-gram matching (n=5), near-verbatim normalized matching (n=6), suppressed phrase matching, and structural fingerprint scoring (cosine similarity). Automatic rewrite loop (up to 2 iterations) for flagged content.
- **Four export formats**: Substack (clean markdown), Medium (drop cap + pull quotes), WordPress (Gutenberg block markup), Twitter/X (numbered thread with 280-character limits)
- **Image prompt generation** (`--image-prompt`) for DALL-E, Midjourney, and Stable Diffusion with six visual styles (photorealistic, editorial, cartoon, minimalist, vintage, abstract) and six platform dimension presets (Substack, Medium, WordPress, Facebook, Twitter, Instagram)
- **Tone preview** -- 2-3 sentence preview before full generation, with interactive confirmation
- **CLI application** with five commands: `write`, `preview`, `modes`, `export`, `config`
- **Mandatory disclaimer** on all output -- cannot be suppressed
- **Fuzzy-match suggestions** when a mistyped mode ID is provided
- **Rich terminal output** with formatted panels, tables, and progress spinners
- **Comprehensive test suite** with 1040 tests, all using mocked API clients

### Changed (from v0.2.0 private prototype)

- Replaced named writer profiles with abstract rhetorical modes -- no writer names appear anywhere in the system
- Replaced `--voice` flag with `--mode` (deprecated flag produces clear error)
- Replaced `--spectrum` flag with `--stance` (deprecated flag produces clear error)
- Replaced ideological baseline concept with non-political equity/liberty vocabulary
- Made the disclaimer mandatory and non-removable (deprecated `--no-disclaimer` flag produces clear error)
- Redesigned stance modifier to use five independent dimensions instead of a single spectrum scale
- Added similarity screening as a mandatory pipeline step

### Removed

- Named writer profile system (replaced by abstract rhetorical modes)
- `--voice` CLI flag (replaced by `--mode`)
- `--spectrum` CLI flag (replaced by `--stance`)
- `--no-disclaimer` CLI flag (disclaimer is now mandatory)
- All writer names, publication references, and biographical data from mode profiles
- Ideological baseline fields from mode profiles

### Security

- API keys cannot be set via CLI; must be configured via environment variables or `.env` file
- API keys are masked in `config` display output (showing only the last 4 characters)
- Similarity screening blocks generated content that reuses source text verbatim

---

## [0.2.0] - Private Prototype

Private development prototype. Not publicly released. Superseded by v1.0.0.
