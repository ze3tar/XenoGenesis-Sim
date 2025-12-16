# AGENTS.md for XenoGenesis-Sim

This file contains **mandatory instructions for any coding agent** working in this repository. Its scope is the entire project.

## Mission
XenoGenesis-Sim is a Linux-first artificial life playground combining digital organisms, continuous cellular automata, and soft-body evolution. Maintain clear documentation, modular code, and reproducible workflows.

## Coding conventions
- Prefer Python 3.9+ features; include type hints and docstrings for new public functions.
- Keep modules small and composable; avoid deep inheritance.
- Favor explicit configuration objects over magic constants. Add default values in a central config when possible.
- Never wrap imports in try/except.
- When touching existing code, preserve current logging and argument patterns unless there is a clear bug.

## Demo and deliverable expectations
- The README must always describe the one-command demo (`make demo`) that builds native code, installs the editable package, runs a CA simulation, saves outputs (plots + Parquet/JSON), and prints a summary table.
- Keep `scripts/run_demo.sh` and `Makefile` targets in sync with documented behavior; highlight checkpoint/resume/analyze flows when touched.

## Documentation expectations
- Update README.md when user-facing behavior or project scope changes.
- Add inline comments only where they clarify non-obvious logic.
- Include short module-level summaries for new files.

## Testing
- Run relevant fast checks. At minimum, run `python -m compileall src examples` after modifying Python code.
- List every command you ran in the final response under **Testing**.

## Git and PR guidance
- Keep commits focused; do not mix unrelated changes.
- After committing, call the `make_pr` tool with a succinct title and body summarizing the change and tests.
- Do not fabricate test results; report failures or limitations honestly.

## Communication
- Follow the system/developer/user message ordering. If instructions conflict, the higher-priority source wins.
- Be concise but explicit in the final summary. Include file path citations per the system format.
