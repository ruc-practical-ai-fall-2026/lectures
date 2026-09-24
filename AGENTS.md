# AGENTS.md

This is a repository with lecture material and executable example code for the Practical AI course, a one-semester course on implementing AI tools that work in real-world environments. These rules override default behavior.

## 0. Human Interaction

- Perform only the steps requested.
- Do not perform extra steps.

## 0.1 Automation Shell

- Use Git Bash as the default shell for all automation commands.
- Prefer Bash syntax and Git Bash paths when running commands.
- Use the full path to Git Bash when running Git Bash commands.
- Use another shell only when Git Bash cannot perform the requested operation.

## 1. Core Philosophy

- Prefer the smallest working solution.
- Readability is more important than cleverness.
- Avoid over-engineering.
- Choose the simplest approach and justify added complexity.
- Do not optimize prematurely.
- Make the code **instructive**. The reader of the code should be able to learn from it.

## 2. Architecture Rules

- Prefer functions over classes.
- Avoid OOP unless it is clearly justified.
- Do not use unnecessary layers, abstractions, or patterns.
- Do not use dependency injection, factories, or frameworks unless required.
- Keep structure flat and direct.

## 3. Code Style

- Use idiomatic Python.
- Follow the Google style guide.
- Ensure code will pass mypy and ruff checks.
- Avoid over-riding ruff checks when possible.
- Prefer the standard library.
- Minimize dependencies.
- Use clear, descriptive names in snake case (do not use clever abbreviations).
- Keep functions small (target: <30 lines).
- Limit nesting (max ~2 levels where possible).

## 4. Modification Rules

When editing existing code:

- Make the smallest possible change.
- Do not refactor unrelated parts.
- Do not rewrite working code without clear benefit.
- Preserve existing structure unless there is a strong reason.

## 5. Comments and Documentation

- Comment only when intent is not obvious.
- Do not restate what the code already shows.
- Prefer clear code over explanatory comments.

## 6. Repository Structure

- Top level folders correspond to classes in the course, e.g., `class-01`, `class-02`, etc.
- Each folder contains documentation in the form of markdown documents, and executable examples in the form of Python modules and Jupyter notebooks.
- Within each class folder, related examples are grouped together into sub-folders, e.g., `class-02/basic_packages_introduction/01_numpy.ipynb`, `class-02/basic_packages_introduction/02_pandas.ipynb`.
- Sub-folders may be installable Python packages if required.
- Top-level folders use hyphenated, all lower-case names, e.g., `class-01`, while lower level folders are idiomatic for the languages and tools they contain, e.g., `python_introduction`.
- Adhere to this structure unless explicitly requested to change it.

## 7. Python Environment and Dependency Management

- The Python environment and dependencies are managed with uv. Do not use or reference other dependency management tools.
- Use `uv run program` to run Python programs using the virtual environment.
- Use `uv run pytest` to run unit tests.
