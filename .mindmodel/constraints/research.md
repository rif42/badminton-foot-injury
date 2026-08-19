# Research & Design Constraints

## Research Documentation (`research_badminton_injury_params/`)
- Each parameter has a dedicated findings file: `findings_*.md` containing key findings, evidence snippets with source URLs, and severity linkage.
- A `research_plan.md` defines the main question, subtopics, synthesis method, and expected outputs before research begins.

## Evidence Standards
- Every claim must cite at least one peer-reviewed source (DOI or URL).
- Source snippets are quoted verbatim where possible to preserve scientific precision.
- Findings distinguish between: direct badminton evidence vs. extrapolated from general landing biomechanics, and high-severity acute injuries vs. chronic overload risk.

## Design Documentation (`docs/superpowers/specs/`)
- Design specs follow a structured format: objective, context, selected approach (with alternatives documented), components, data flow, error handling, dependencies, out-of-scope items, future hooks.
- Designs are date-stamped and versioned as `YYYY-MM-DD-{topic}-design.md`.

## Separation of Concerns Between Research and Runtime
- **Research files** (`*.md`) contain no executable Python code — they document what should be implemented.
- **Runtime modules** (`*.py`) contain implementation only — biomechanical thresholds live in the runtime module, not duplicated in research notes.
- The pose detector intentionally excludes injury scoring; this is a documented boundary, not an oversight.
