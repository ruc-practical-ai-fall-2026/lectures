---
name: format-ruff
description: Format and auto-fix this repository's Python code with Ruff using uv.
---

# Format with Ruff

Use Git Bash as the shell for repository formatting. Run these commands from the repository root, in order:

```bash
uv run ruff format .
uv run ruff check --fix .
```

Report the result of each command. Do not run additional formatters or make unrelated edits.
