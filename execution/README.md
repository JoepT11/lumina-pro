# Layer 3: Execution

This directory contains deterministic Python scripts that perform the actual work.

## Principles
1. **Deterministic & Testable**: Avoid embedding probabilistic LLM logic in execution scripts. Keep business logic, image processing, API calls, and transformations deterministic.
2. **Environment Variables**: Load secrets and configuration from `.env` (e.g. using `python-dotenv`).
3. **CLI Arguments**: Scripts should accept arguments via `argparse` or `sys.argv` so the orchestrator (Layer 2) can invoke them reliably.
4. **Intermediate Files**: Save working files, dumps, and cache to `.tmp/`.
5. **Self-Annealing**: When an error occurs, inspect the traceback, fix the script, verify that it runs cleanly, and record any learnings back into the corresponding directive.
