---
name: ort-lint
description: Lint and format ONNX Runtime code. Use this skill when asked to lint, format, or check code style for C++ or Python files in ONNX Runtime.
---

# Linting and Formatting ONNX Runtime Code

ONNX Runtime uses [lintrunner](https://github.com/suo/lintrunner) to manage linting for both C++ (clang-format) and Python (ruff).

## Setup (one-time)

Install lintrunner and initialize it:

```bash
pip install -r requirements-lintrunner.txt
lintrunner init
```

This downloads and configures the required linting tools (clang-format, ruff, etc.).

## Common commands

```bash
# Auto-fix lint issues in changed files (staged + unstaged vs HEAD)
lintrunner -a

# Auto-fix lint issues in ALL files
lintrunner -a --all-files

# Format Python files only
lintrunner f --all-files

# Check without fixing (dry run)
lintrunner

# Lint specific files
lintrunner -a path/to/file.py path/to/other_file.cc
```

## Style rules

### C++
- Google C++ Style with modifications
- Max line length: 120 characters
- Configured in `.clang-format` and `.clang-tidy`

### Python
- Google Python Style Guide (extension of PEP 8)
- Max line length: 120 characters
- Formatter/linter: ruff (configured in `pyproject.toml`)

## Workflow

1. If lintrunner is not yet set up, install dependencies and run `lintrunner init`.
2. Determine scope: changed files only (`lintrunner -a`) or all files (`lintrunner -a --all-files`).
3. Run the appropriate lint command.
4. If there are unfixable issues, report them to the user with file locations and descriptions.
