---
name: ort-lint
description: Lint and format ONNX Runtime code. Use this skill when asked to lint, format, or check code style for C++ or Python files in ONNX Runtime.
---

# Linting and Formatting ONNX Runtime Code

ONNX Runtime uses [lintrunner](https://github.com/suo/lintrunner) to manage linting for both C++ (clang-format) and Python (ruff).

## Initial Setup

Install lintrunner and its dependencies, then initialize:

```bash
pip install -r requirements-lintrunner.txt
lintrunner init
```

This setup only needs to be done once.

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

1. **Activate a Python virtual environment** before installing dependencies. See the "Python Environment" section in `AGENTS.md` for instructions.
2. If lintrunner is not yet set up, install and initialize. See the [initial setup section](#initial-setup) for instructions.
3. Determine scope: changed files only (`lintrunner -a`) or all files (`lintrunner -a --all-files`).
4. Run the appropriate lint command.
5. If there are unfixable issues, report them to the user with file locations and descriptions.
