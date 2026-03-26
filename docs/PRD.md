# PRD: OpinionForge v2.0.0 (Local Desktop Application)

**PRD Version:** 5.0
**Last Updated:** 2026-03-25
**Product Version:** 2.0.0

**Revision History:**

| PRD Version | Date | Description |
|-------------|------|-------------|
| 1.0 | 2026-03-25 | Original Phase 1 MVP spec (PRD Writer output) |
| 2.0 | 2026-03-25 | Legal refactor rewrite — 12 rhetorical modes replace 100 named profiles |
| 3.0 | 2026-03-25 | 9 amendments from legal counsel: tightened safeguards, removed escape hatches, blocked output on screening failure, removed named-writer mappings from spec |
| 4.0 | 2026-03-25 | Amendment: Web UI added to v1.0.0 scope (Feature 9, Sprints 8-10) |
| 5.0 | 2026-03-25 | Architecture change: local desktop application with pluggable LLM backends. Replaces hosted web app model. |

---

## Executive Summary

OpinionForge v2.0.0 is a local desktop application that helps writers produce opinion pieces with precise rhetorical control, sourced research, and originality screening. It runs entirely on the user's machine and connects to whatever LLM backend the user already has — Ollama running locally, Claude API, OpenAI API, or any compatible provider.

The user installs it, opens it in their browser, and writes. No account creation. No subscription. No server to maintain. OpinionForge provides the rhetorical intelligence — 12 modes, stance/intensity controls, source research, similarity screening, export formatting — and the user's LLM provides the generation.

This replaces the v1.0.0 architecture (FastAPI web server requiring deployment and API key management) with a local-first design that eliminates hosting, auth, billing, and rate limiting entirely.

## Problem Statement

OpinionForge v1.0.0 has a working engine — 12 rhetorical modes, stance/intensity controls, similarity screening, confusability testing, 4 export formats, image prompt generation — but no way for a non-developer to use it. The current architecture requires:

1. Python 3.11+ installed
2. Git to clone the repo
3. A virtual environment
4. An Anthropic API key configured as an environment variable
5. Running `opinionforge serve` from the command line

That's not a product. That's a source code checkout.

The v1.0.0 web UI (FastAPI + HTMX) assumed a hosted deployment model, which introduced a cascade of infrastructure requirements: hosting, auth, billing, rate limiting, user management. None of those are built.

The fundamental insight: the product's value is in the rhetorical engine, not in hosting an LLM. The LLM is a commodity the user can provide themselves. OpinionForge should be a local tool that plugs into whatever LLM the user already has, the same way a code editor plugs into whatever compiler the user has installed.

## Target Users

- **Opinion writers and editors** at digital publications (Substack, Medium, WordPress) who want AI-assisted drafting with precise rhetorical control.
- **Solo newsletter writers** who want to experiment with different rhetorical approaches to find their own editorial voice.
- **Content strategists and communications professionals** who need persuasive editorial content with controllable argumentative intensity and evidence style.
- **Academic and policy writers** who want to translate research into accessible opinion formats.
- **Anyone who already uses a local LLM** (Ollama, LM Studio, etc.) and wants a purpose-built writing tool on top of it.

## Goals

- A non-technical user can install and use OpinionForge in under 5 minutes
- The application runs locally with no external dependencies beyond the user's chosen LLM provider
- Support at minimum: Ollama (local), Anthropic Claude API, OpenAI API
- First-run onboarding guides the user through LLM provider setup and explains modes, stance, and intensity
- Generated pieces are saved locally with full history and search
- The application is a single install — no Python knowledge, no terminal commands, no environment variables configured by hand
- All v1.0.0 safeguards carry forward: mandatory disclaimers, similarity screening, no named-writer references
- Position as "editorial craft engine" in all user-facing text

## Non-Goals

- This is NOT a hosted service. No server deployment, no cloud infrastructure, no user accounts.
- This does NOT require the user to have Python installed. The application bundles its own runtime.
- This does NOT add new rhetorical modes beyond the existing 12.
- This does NOT change the similarity screening, confusability testing, or disclaimer system.
- This does NOT add collaborative features (shared documents, team accounts, etc.).
- This does NOT add real-time co-editing or multiplayer functionality.
- Internal research notes referencing writers by name may exist only in a private `research/` directory that is `.gitignore`-excluded, never shipped, never loaded at runtime, never referenced by tests, examples, fixtures, prompts, or generated artifacts, and never required for operation of the public release. This PRD does not govern internal research workflow except to prohibit any runtime or shipped dependency on those materials.

## Architecture Change: Why Local

| Concern | v1.0.0 (Hosted) | v2.0.0 (Local) |
|---------|-----------------|-----------------|
| Hosting | You run a server | User's machine |
| Auth | You build it | Not needed |
| Billing | You build it | User pays their own LLM provider |
| Rate limiting | You build it | Not needed |
| API keys | User configures env vars | First-run setup wizard |
| LLM provider | Hardcoded Anthropic/OpenAI | Pluggable — Ollama, Claude, OpenAI, any OpenAI-compatible |
| Installation | `git clone && pip install .` | Download installer or `pip install opinionforge` |
| Data storage | None (stateless) | Local SQLite database |
| Updates | Redeploy server | `pip install --upgrade opinionforge` or auto-update |
| Offline capability | None | Works offline with local LLM (Ollama) |

## Core Features

### Feature 1: Pluggable LLM Backend

OpinionForge connects to whatever LLM the user has. A provider adapter layer abstracts the LLM call so the rest of the engine is provider-agnostic.

**Supported providers:**

| Provider | Type | Requirements | Notes |
|----------|------|--------------|-------|
| Ollama | Local | Ollama installed, model pulled | Free, offline capable, no API key |
| Anthropic Claude | Cloud | API key | Highest quality for rhetorical tasks |
| OpenAI | Cloud | API key | GPT-4o, GPT-4-turbo |
| OpenAI-compatible | Cloud/Local | Base URL + optional API key | LM Studio, vLLM, text-generation-inference, any provider with OpenAI-compatible API |

**Provider interface:**

```python
class LLMProvider(Protocol):
    """Any LLM backend must implement this interface."""

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text from a system prompt and user prompt."""
        ...

    async def stream(self, system_prompt: str, user_prompt: str, max_tokens: int) -> AsyncIterator[str]:
        """Stream text token by token."""
        ...

    def model_name(self) -> str:
        """Return the model identifier for logging/display."""
        ...
```

**Provider implementations:**

- `opinionforge/providers/ollama.py` — Uses Ollama's REST API (`http://localhost:11434/api/generate`)
- `opinionforge/providers/anthropic.py` — Uses Anthropic SDK (existing code, extracted)
- `opinionforge/providers/openai.py` — Uses OpenAI SDK (existing code, extracted)
- `opinionforge/providers/openai_compatible.py` — Uses OpenAI SDK with custom `base_url` for any compatible provider

**Model recommendations:**

Not all models handle rhetorical tasks equally. The onboarding wizard suggests models known to work well:

| Provider | Recommended Models | Minimum |
|----------|--------------------|---------|
| Ollama | `llama3.1:70b`, `qwen2.5:32b`, `mistral-large` | `llama3.1:8b` (reduced quality) |
| Anthropic | `claude-sonnet-4-6`, `claude-opus-4-6` | `claude-haiku-4-5` |
| OpenAI | `gpt-4o`, `gpt-4-turbo` | `gpt-4o-mini` (reduced quality) |

### Feature 2: Local Data Storage

All generated pieces, settings, and history are stored in a local SQLite database. No cloud sync. No data leaves the user's machine (except LLM API calls to their chosen provider).

**Database location:** `~/.opinionforge/opinionforge.db` (or platform-appropriate app data directory)

**Tables:**

```sql
-- Generated pieces with full metadata
CREATE TABLE pieces (
    id TEXT PRIMARY KEY,              -- UUID
    created_at TIMESTAMP NOT NULL,
    topic TEXT NOT NULL,
    topic_source TEXT,                -- 'text', 'url', 'file'
    source_url TEXT,
    mode_config TEXT NOT NULL,        -- JSON: mode blend config
    stance_position INTEGER NOT NULL,
    stance_intensity REAL NOT NULL,
    length_preset TEXT,
    word_count INTEGER,
    title TEXT,
    subtitle TEXT,
    body TEXT NOT NULL,
    sources TEXT,                     -- JSON: list of sources
    disclaimer TEXT NOT NULL,
    screening_passed BOOLEAN NOT NULL,
    screening_details TEXT,           -- JSON: ScreeningResult
    image_prompt TEXT,
    provider TEXT NOT NULL,           -- e.g., 'ollama/llama3.1:70b'
    generation_time_ms INTEGER,
    exported_formats TEXT             -- JSON: list of formats exported to
);

-- User preferences
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Export history
CREATE TABLE exports (
    id TEXT PRIMARY KEY,
    piece_id TEXT NOT NULL REFERENCES pieces(id),
    format TEXT NOT NULL,             -- 'substack', 'medium', 'wordpress', 'twitter'
    exported_at TIMESTAMP NOT NULL,
    content TEXT NOT NULL
);
```

**Piece history features:**
- Browse all generated pieces with search and filter
- Filter by mode, stance range, date range, topic keywords
- Re-export any saved piece to a different format
- Delete individual pieces or bulk delete
- Pieces are never auto-deleted

### Feature 3: First-Run Onboarding

The first time a user opens OpinionForge, they see a setup wizard — not a blank form. The wizard:

1. **Welcome screen** — "OpinionForge is an editorial craft engine. It helps you write opinion pieces with precise rhetorical control." One paragraph, no jargon.

2. **LLM provider setup** — "OpinionForge needs a language model to generate text. Which do you have?"
   - **Ollama (local, free)** — Detects if Ollama is installed. If yes, shows available models. If no, links to install instructions. Recommends pulling `llama3.1:70b` or `qwen2.5:32b`.
   - **Anthropic Claude (cloud)** — API key input field. "Paste your API key from console.anthropic.com." Test connection button.
   - **OpenAI (cloud)** — API key input field. "Paste your API key from platform.openai.com." Test connection button.
   - **Other (OpenAI-compatible)** — Base URL + optional API key fields. Test connection button.
   - Connection test runs immediately. Green checkmark or red error with specific fix instructions.

3. **Search provider setup** (optional) — "OpinionForge can research sources for your topics. This requires a search API key."
   - Tavily (recommended, free tier available)
   - Brave Search
   - SerpAPI
   - "Skip for now" — research features disabled, generation still works.

4. **Quick tour** — Interactive walkthrough of the main interface:
   - "These are rhetorical modes. Pick one to set the style of your piece."
   - "This slider controls your argumentative stance."
   - "This slider controls intensity — how forceful the rhetoric is."
   - "Type a topic and click Generate."
   - Tour is skippable and can be replayed from settings.

5. **Test generation** — "Let's make sure everything works. We'll generate a short piece on a sample topic." Runs a real generation with the configured provider. If it works, the user sees their first piece and knows the tool is functional. If it fails, clear error with fix instructions.

### Feature 4: Improved Web UI

The existing HTMX web UI is preserved and improved. It runs as a local server (localhost) and opens in the user's default browser.

**New pages/features beyond v1.0.0:**

| Page | Description |
|------|-------------|
| `/setup` | First-run onboarding wizard |
| `/history` | Browse, search, and filter all generated pieces |
| `/history/{id}` | View a saved piece with re-export options |
| `/settings` | LLM provider config, search provider config, UI preferences |

**Home page improvements:**
- Mode cards show a one-sentence description and category color
- Tooltips on stance and intensity sliders explain what the values mean
- "What's this?" help links on every control that open inline explanations
- Recent pieces sidebar (last 5) for quick reference
- Topic input remembers the last 10 topics as suggestions

**History page:**
- Card grid of all generated pieces showing title, mode, date, word count
- Search by topic keywords
- Filter by mode, stance range, date range
- Sort by date, word count, or mode
- Bulk select and delete
- Click any piece to view it full-screen with re-export buttons

**Settings page:**
- LLM provider selection and configuration (same options as onboarding wizard)
- Test connection button
- Search provider configuration
- Default mode, stance, intensity, length preferences
- Theme preference (light/dark)
- Data management: export all pieces as JSON, clear history, database location

### Feature 5: Desktop Integration

OpinionForge launches like a desktop application, not a CLI tool.

**Installation methods:**

1. **pip install** (for Python users):
   ```
   pip install opinionforge
   opinionforge
   ```
   Opens browser to `http://localhost:8484`. Port 8484 chosen to avoid conflicts with common dev servers on 3000, 5000, 8000, 8080.

2. **Standalone installer** (for non-Python users):
   - Windows: `.exe` installer (PyInstaller or Briefcase)
   - macOS: `.dmg` (Briefcase)
   - Linux: AppImage or `.deb`
   - Bundles Python runtime — user doesn't need Python installed
   - Desktop shortcut, start menu entry, system tray icon

**System tray behavior:**
- OpinionForge icon in system tray while running
- Click to open in browser
- Right-click menu: Open, Settings, Quit
- Closing the browser tab doesn't stop the server — it keeps running in the tray
- "Quit" from the tray actually stops the process

**Auto-launch (optional):**
- Setting to start OpinionForge on system boot (minimized to tray)
- Disabled by default

### Feature 6: Rhetorical Modes (carried forward from v1.0.0)

No changes. The 12 abstract rhetorical modes are identical to v1.0.0:

`polemical`, `analytical`, `populist`, `satirical`, `forensic`, `oratorical`, `narrative`, `data-driven`, `aphoristic`, `dialectical`, `provocative`, `measured`

Organized into 4 categories: confrontational, investigative, deliberative, literary.

Mode blending (up to 3 modes with weights summing to 100) is unchanged.

### Feature 7: Stance and Intensity Controls (carried forward)

No changes. `--stance` (-100 to +100) and `--intensity` (0.0 to 1.0) work identically to v1.0.0.

### Feature 8: Mandatory Disclaimers (carried forward)

No changes. Every output includes the fixed disclaimer. No opt-out in the UI or CLI. The disclaimer is always present in generated and exported content. OpinionForge states the requirement but does not enforce it beyond the application boundary — once a user exports text, the responsibility for compliance rests with them.

### Feature 9: Similarity Screening (carried forward)

No changes. Verbatim detection, near-verbatim detection, suppressed phrase blocking, structural fingerprinting, max 2 rewrite iterations, output blocked on failure.

### Feature 10: Export Formats (carried forward)

No changes. Substack, Medium, WordPress, Twitter/X. All include mandatory disclaimer.

### Feature 11: Image Prompt Generator (carried forward)

No changes. 6 styles, 6 platform dimensions.

### Feature 12: Source Research (carried forward)

No changes. Tavily, Brave, or SerpAPI. Claim-to-source linkage, credibility scoring, real URLs only.

### Feature 13: CLI (preserved)

The CLI continues to work for users who prefer it. All commands unchanged:

```
opinionforge write TOPIC [OPTIONS]
opinionforge preview TOPIC [OPTIONS]
opinionforge modes [OPTIONS]
opinionforge serve [OPTIONS]
opinionforge config [OPTIONS]
```

New addition: `opinionforge` with no arguments launches the web UI (equivalent to `opinionforge serve` with browser auto-open).

New flags on all commands:
- `--provider` — override the configured LLM provider for this run
- `--model` — override the configured model for this run

## Data Models

### Carried forward from v1.0.0 (no changes)

- `ModeProfile` — 12 rhetorical mode definitions
- `ModeBlendConfig` — mode blending with weights
- `StanceConfig` — position (-100 to +100) and intensity (0.0 to 1.0)
- `GeneratedPiece` — generated output with metadata
- `ScreeningResult` — similarity screening results
- `ImagePromptConfig` — image generation parameters

### New models

```python
class ProviderConfig(BaseModel):
    """LLM provider configuration."""
    provider_type: str          # 'ollama', 'anthropic', 'openai', 'openai_compatible'
    model: str                  # e.g., 'llama3.1:70b', 'claude-sonnet-4-6'
    api_key: str | None = None  # encrypted at rest for cloud providers
    base_url: str | None = None # for ollama or openai-compatible
    max_tokens: int = 4096

class SearchConfig(BaseModel):
    """Search provider configuration."""
    provider: str               # 'tavily', 'brave', 'serpapi', 'none'
    api_key: str | None = None

class UserPreferences(BaseModel):
    """Stored user defaults."""
    default_mode: str = "analytical"
    default_stance: int = 0
    default_intensity: float = 0.5
    default_length: str = "standard"
    theme: str = "light"        # 'light' or 'dark'
    auto_launch: bool = False
    onboarding_completed: bool = False
```

## Module Map

```
opinionforge/
  __init__.py
  __main__.py                   # Entry point: launches web UI by default
  cli.py                        # CLI commands (preserved)
  config.py                     # Settings via pydantic-settings

  core/                         # Generation engine (unchanged)
    generator.py
    image_prompt.py
    length.py
    mode_engine.py
    preview.py
    research.py
    similarity.py
    stance.py
    topic.py

  providers/                    # NEW: pluggable LLM backends
    __init__.py
    base.py                     # LLMProvider protocol
    ollama.py
    anthropic.py
    openai_provider.py
    openai_compatible.py
    registry.py                 # Provider discovery and instantiation

  storage/                      # NEW: local persistence
    __init__.py
    database.py                 # SQLite connection, migrations
    pieces.py                   # CRUD for generated pieces
    settings.py                 # CRUD for user preferences
    exports.py                  # CRUD for export history

  web/                          # Web UI (enhanced)
    __init__.py
    app.py                      # FastAPI app factory
    sse.py                      # SSE streaming
    templates/
      base.html
      home.html
      modes.html
      mode_detail.html
      about.html
      setup.html                # NEW: onboarding wizard
      history.html              # NEW: piece history
      history_detail.html       # NEW: single piece view
      settings.html             # NEW: settings page
      partials/
        piece_result.html
        preview_result.html
        export_result.html
        error.html
        progress.html
        piece_card.html         # NEW: history card
    static/
      style.css
      onboarding.js             # NEW: setup wizard logic (vanilla JS)

  exporters/                    # (unchanged)
    substack.py
    medium.py
    wordpress.py
    twitter.py

  modes/                        # (unchanged)
    profiles/
      <12 mode YAML files>
    categories.yaml

  data/                         # (unchanged)
    suppressed_phrases.yaml
    structural_fingerprints.yaml

  models/                       # (updated)
    config.py                   # + ProviderConfig, SearchConfig, UserPreferences
    mode.py
    piece.py
    topic.py
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | Existing codebase |
| CLI | Typer | Existing |
| Web framework | FastAPI + HTMX + Jinja2 | Existing, enhanced |
| Data validation | Pydantic v2 | Existing |
| Database | SQLite via aiosqlite | Zero config, ships with Python, perfect for local app |
| LLM (local) | Ollama REST API | Most popular local LLM runtime |
| LLM (cloud) | Anthropic SDK, OpenAI SDK | Existing |
| Search | Tavily, Brave, SerpAPI | Existing |
| Packaging | PyInstaller (Windows), Briefcase (macOS/Linux) | Standalone installers without requiring Python |
| Build | hatchling | Existing |
| Testing | pytest | Existing |

## Dependencies

Carried forward from v1.0.0:

| Package | Purpose |
|---------|---------|
| typer | CLI |
| rich | Terminal output |
| pydantic / pydantic-settings | Models and config |
| httpx | HTTP client |
| trafilatura | Content extraction |
| anthropic | Claude API |
| openai | OpenAI API |
| tavily-python | Search |
| pyyaml | Mode profiles |
| python-dotenv | Env vars |
| fastapi | Web framework |
| uvicorn | ASGI server |
| jinja2 | Templates |
| sse-starlette | Server-sent events |
| python-multipart | Form handling |

New dependencies:

| Package | Purpose |
|---------|---------|
| aiosqlite | Async SQLite for local storage |
| cryptography | Encrypt API keys at rest |
| platformdirs | Cross-platform app data directory |

## Testing Strategy

### Carried forward from v1.0.0
- Mode loading, blending, stance, disclaimer, screening, CLI, export, name sanitization, runtime isolation, confusability — all tests preserved.

### New tests

- **Provider tests:** Each provider adapter is tested with mock HTTP responses. Ollama, Anthropic, OpenAI, OpenAI-compatible. Tests verify correct request format, error handling, streaming.
- **Provider registry:** Test auto-detection (Ollama running? Which models available?), fallback behavior, invalid config handling.
- **Database tests:** CRUD operations on pieces, settings, exports. Migration tests. Concurrent access. Database creation on first run.
- **Onboarding tests:** Each wizard step renders correctly. Connection test works for each provider type. Skip behavior. Resume after partial completion.
- **History tests:** Search, filter, sort, pagination, delete, bulk delete.
- **Settings tests:** Provider config save/load. Preference save/load. API key encryption/decryption.
- **Desktop integration tests:** `opinionforge` with no args launches server and opens browser. System tray behavior. Port conflict handling.
- **Installer tests:** Standalone binary starts correctly. Database created in correct location. First-run detection works.

### Coverage target
- 90%+ line coverage for all modules except confusability tests
- Every provider adapter has connection test, generate test, stream test, error test
- Every database operation has happy path and error path test

## Version Roadmap

| Version | Description |
|---------|-------------|
| 0.1.0 | MVP — CLI, 10 writer profiles, spectrum slider (private, archived) |
| 0.2.0 | 100 writer profiles, exports, image prompts (private, archived) |
| 1.0.0 | Legal refactor — 12 modes, screening, disclaimers, web UI (current codebase) |
| **2.0.0** | **Local desktop app — pluggable LLM, local storage, onboarding, installer (this PRD)** |

## Success Criteria

- **5-minute install to first piece.** A non-technical user with Ollama already installed can download OpinionForge, run it, complete onboarding, and generate their first piece in under 5 minutes.
- **Works offline.** With Ollama and a local model, OpinionForge generates pieces with no internet connection (research features disabled).
- **Provider-agnostic.** The same piece generated with the same settings produces comparable rhetorical quality across Ollama (large model), Claude, and GPT-4o.
- **Pieces persist.** Every generated piece is saved automatically. Users can find, re-read, and re-export any piece from their history.
- **All v1.0.0 safeguards pass.** Zero writer names, confusability below threshold, screening catches planted matches, disclaimer always present.
- **No public named-person mappings.** No shipped docs, tests, examples, config files, fixtures, comments, or help text contain mappings from rhetorical modes to specific real writers.
- **Materially risk-reduced public release** with mandatory disclosure, blocked screening failures, and no shipped named-person emulation surface.

## Deliverables Checklist

- [ ] Provider adapter layer with Ollama, Anthropic, OpenAI, OpenAI-compatible implementations
- [ ] Provider registry with auto-detection and connection testing
- [ ] SQLite database with pieces, settings, exports tables
- [ ] Database migrations for future schema changes
- [ ] API key encryption at rest
- [ ] First-run onboarding wizard (5 steps)
- [ ] History page with search, filter, sort, delete
- [ ] Settings page with provider config, preferences, data management
- [ ] `opinionforge` bare command launches web UI with browser auto-open
- [ ] System tray integration (Windows, macOS, Linux)
- [ ] Standalone installers for Windows (.exe), macOS (.dmg), Linux (AppImage)
- [ ] PyPI package (`pip install opinionforge`)
- [ ] All v1.0.0 features preserved (modes, stance, screening, disclaimers, exports, image prompts, CLI)
- [ ] Provider-specific tests with mock HTTP
- [ ] Database CRUD tests
- [ ] Onboarding flow tests
- [ ] History and settings tests
- [ ] 90%+ test coverage
- [ ] Updated README for local-first installation
- [ ] Updated landing page reflecting desktop application positioning
- [ ] Updated docs/terms.html
