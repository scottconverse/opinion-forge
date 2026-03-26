# OpinionForge — Rhetorical Mode Catalog

OpinionForge ships with 12 rhetorical modes organized into four categories. Each mode defines a distinct argumentative approach, prose style, and rhetorical toolkit.

Use the `--mode` flag to select a mode when generating a piece:

```bash
opinionforge write "your topic" --mode polemical
```

You can also blend modes using colon-separated weights:

```bash
opinionforge write "your topic" --mode polemical:60,analytical:40
```

---

## Confrontational Modes

These modes take an aggressive or oppositional stance toward their subject.

---

### Polemical (`polemical`)

**Category:** confrontational

**Description:**
A combative, morally urgent mode that names enemies, indicts positions as contemptible, and demands the reader choose sides.

**Rhetorical Devices:**
- Rhetorical questions used as indictments
- Antithesis to sharpen moral contrast
- Periodic sentences that build to a damning conclusion
- Direct address to challenge the reader
- Anaphora for cumulative emphasis

---

### Populist (`populist`)

**Category:** confrontational

**Description:**
A plain-spoken, us-versus-them mode that amplifies the voice of ordinary people against remote, impersonal power. Grounds every abstraction in concrete human detail.

**Rhetorical Devices:**
- Second-person direct address to the reader
- Concrete specific detail standing in for the systemic
- Repetition of the name of the institution or elite being indicted
- Short declarative sentences that simulate plain speech
- The list of grievances that accumulates into an indictment

---

### Provocative (`provocative`)

**Category:** confrontational

**Description:**
A deliberately unsettling mode that states unfashionable truths, attacks comfortable assumptions, and forces readers to confront implications they would prefer to avoid.

**Rhetorical Devices:**
- The flat contradiction of received wisdom
- The inconvenient implication — following the comfortable premise to its uncomfortable conclusion
- The inversion — turning a virtue into a liability, or a vice into an asset
- Concessive openings that grant the premise before attacking it
- The extended hypothetical that reveals absurdity

---

## Investigative Modes

These modes prioritize evidence, documentation, and methodical construction of argument.

---

### Forensic (`forensic`)

**Category:** investigative

**Description:**
A methodical investigative mode that assembles documentary evidence, follows the money, exposes contradictions, and builds a prosecutorial case from primary sources.

**Rhetorical Devices:**
- Attribution of claims to specific primary sources
- Chronological reconstruction of events
- Juxtaposition of contradictory statements by the same actor
- The document — quoting the record to make it indict itself
- The timeline that reveals pattern

---

### Data-Driven (`data_driven`)

**Category:** investigative

**Description:**
A quantitative mode that leads with numbers, visualizes trends, and uses statistical precision to anchor argument in empirical reality.

**Rhetorical Devices:**
- Specific quantification rather than vague magnitude
- Comparison to baseline or benchmark to establish significance
- Trend description using precise time ranges
- The unexpected number that reframes the issue
- Juxtaposition of statistics that reveals disparity

---

## Deliberative Modes

These modes reason through complex issues with care, balance, and intellectual rigor.

---

### Analytical (`analytical`)

**Category:** deliberative

**Description:**
A rigorous, evidence-first mode that breaks complex issues into component parts, weighs competing explanations, and arrives at measured conclusions.

**Rhetorical Devices:**
- Precise qualification to bound claims
- Parallel structure to compare alternatives
- Topic sentences that state the paragraph's logical contribution
- The acknowledged counterargument that strengthens the main thesis
- Hedged assertion that signals calibrated confidence

---

### Dialectical (`dialectical`)

**Category:** deliberative

**Description:**
A thesis-antithesis-synthesis mode that takes opposing arguments seriously, follows the logic where it leads, and arrives at a position through genuine engagement with difficulty.

**Rhetorical Devices:**
- Steelmanning — presenting the opposition at its most compelling
- The turn — "and yet," "but this misses something important"
- Concessive clauses that grant without surrendering
- The synthesis that genuinely incorporates the opposing insight
- Numbered or structured tension between positions

---

### Measured (`measured`)

**Category:** deliberative

**Description:**
A calm, earned-authority mode that conveys confidence through restraint, precision, and a refusal to claim more than the evidence warrants.

**Rhetorical Devices:**
- The deliberate qualification — "to the extent that," "in the cases where"
- The acknowledged limit — "this argument does not address"
- The earned assertion — stating only what the preceding argument has established
- Understatement that implies more than it claims
- The long sentence that earns its complexity

---

## Literary Modes

These modes draw on narrative craft, wit, and elevated rhetorical tradition.

---

### Satirical (`satirical`)

**Category:** literary

**Description:**
A mode that uses irony, exaggeration, and comic incongruity to expose the absurdity of its target. The argument is made by letting the target's logic destroy itself.

**Rhetorical Devices:**
- Sustained irony and deadpan understatement
- Hyperbole that extends the target's logic to its ridiculous conclusion
- Mock praise that operates as condemnation
- The earnest explanation of something obviously absurd
- Bathos — the sudden deflation of inflated language

---

### Oratorical (`oratorical`)

**Category:** literary

**Description:**
A grand, elevated mode that deploys the full arsenal of classical rhetoric — anaphora, periodic sentences, tricolon — to produce writing that feels designed to be spoken aloud.

**Rhetorical Devices:**
- Anaphora for cumulative moral force
- Periodic sentences that withhold the main verb for emphasis
- Tricolon and rule of three
- The rhetorical question that implies its own answer
- The peroration — the rising close that builds to a declaration

---

### Narrative (`narrative`)

**Category:** literary

**Description:**
A story-first mode that uses scene, character, and personal testimony to make abstract arguments vivid and emotionally resonant. The argument emerges from the story rather than preceding it.

**Rhetorical Devices:**
- In medias res — beginning in the middle of an event
- Specific sensory detail to anchor abstraction in physical reality
- Indirect discourse — reporting speech that reveals character
- The anecdote that stands for the systemic
- The return — ending where the story began, with altered meaning

---

### Aphoristic (`aphoristic`)

**Category:** literary

**Description:**
A compressed, epigrammatic mode that delivers argument through maxims, paradoxes, and sentences designed to be memorable. Every word carries maximum weight.

**Rhetorical Devices:**
- Chiasmus — reversal of grammatical structure
- Paradox that reveals a deeper truth
- Compression — removing all words that are not load-bearing
- The gnomic present — stating the particular as universal truth
- The antithetical pair that illuminates through contrast

---

## Quick Reference

| Mode ID | Display Name | Category |
|---------|-------------|----------|
| `analytical` | Analytical | deliberative |
| `aphoristic` | Aphoristic | literary |
| `data_driven` | Data-Driven | investigative |
| `dialectical` | Dialectical | deliberative |
| `forensic` | Forensic | investigative |
| `measured` | Measured | deliberative |
| `narrative` | Narrative | literary |
| `oratorical` | Oratorical | literary |
| `polemical` | Polemical | confrontational |
| `populist` | Populist | confrontational |
| `provocative` | Provocative | confrontational |
| `satirical` | Satirical | literary |

---

## Browsing Modes via the Web UI

Modes can also be browsed interactively through the web interface. Start the web server with `opinionforge serve` and visit `/modes` to explore all 12 modes with full profile details, category filtering, and direct links to use each mode in the generation form.

---

## Using Modes via the CLI

```bash
# List all modes
opinionforge modes

# Filter by category
opinionforge modes --category confrontational

# View full details for a specific mode
opinionforge modes --detail forensic

# Generate using a mode
opinionforge write "your topic here" --mode forensic --no-preview

# Blend two modes
opinionforge write "your topic" --mode polemical:70,analytical:30 --no-preview
```
