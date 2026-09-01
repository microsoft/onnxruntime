---
name: ort-lint
description: Lint and format ONNX Runtime code. Use this skill when asked to lint, format, or check code style for C++ or Python files in ONNX Runtime.
---

# Linting and Formatting ONNX Runtime Code

ONNX Runtime uses [lintrunner](https://github.com/suo/lintrunner) for both C++ (clang-format) and Python (ruff).

## Setup (one-time)

```bash
pip install -r requirements-lintrunner.txt
lintrunner init
```

## Commands

```bash
lintrunner -a                                        # auto-fix changed files
lintrunner -a --all-files                            # auto-fix all files
lintrunner -a path/to/file.py path/to/other_file.cc  # auto-fix specific files
lintrunner f --all-files                             # format Python files only
lintrunner                                           # check without fixing (dry run)
```

## Style rules

### C++
- Google C++ Style with modifications (see `docs/Coding_Conventions_and_Standards.md` for full details)
- Max line length: 120 characters, but **aim for 80** when possible
- Configured in `.clang-format` and `.clang-tidy`

### Python
- Google Python Style Guide (extension of PEP 8)
- Max line length: 120 characters
- Configured in `pyproject.toml`

## Agent tips

- **Activate a Python virtual environment** before installing dependencies. See "Python > Virtual environment" in `AGENTS.md`.
- If lintrunner is not yet set up, install and initialize first (see [Setup](#setup-one-time)).
- Prefer `lintrunner -a` (changed files only) over `--all-files` unless the user asks for a full sweep.
