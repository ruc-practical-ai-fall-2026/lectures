---
name: run-pytest
description: Run and report PyTest unit tests for this repository when tests need to be verified.
---

# Run PyTest

Use Git Bash as the shell for test automation. Run the relevant tests with:

```bash
uv run pytest
```

Use a narrower path or test expression when the request targets specific tests. Report the command and result, including failures with their file and test names. Do not modify tests or production code while running them.
