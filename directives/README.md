# Layer 1: Directives

Directives are Standard Operating Procedures (SOPs) written in Markdown. They define **what to do**, leaving the orchestration to the agent and the execution to deterministic Python scripts in `execution/`.

## Anatomy of a Directive

Each directive should define:
1. **Goal**: High-level objective.
2. **Inputs**: Parameters, file paths, credentials, or context required.
3. **Execution Scripts**: Which deterministic tools in `execution/` should be called, and in what sequence.
4. **Outputs / Deliverables**: Intermediate files in `.tmp/` or deliverables (e.g. Google Sheets, Slides, database records, web assets).
5. **Edge Cases & Learnings**: Known constraints, API limits, failure modes, and self-annealed fixes.

## Guidelines
- Directives are **living documents**. When you encounter API limits, timeouts, or edge cases, update the directive.
- Directives should read like instructions for a capable mid-level employee.
- See `_template.md` for a starting scaffold.
