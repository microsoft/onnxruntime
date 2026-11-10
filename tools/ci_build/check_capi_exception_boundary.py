#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Check that C-API function bodies balance their exception-boundary macros.

Exception-handling C-API function bodies use ``API_IMPL_BEGIN ... API_IMPL_END``
(or the ``TENSOR_READ_API_BEGIN`` / ``TENSOR_READWRITE_API_BEGIN`` openers, which both
expand to ``API_IMPL_BEGIN``). That funnel catches C++ exceptions so they never cross the
C ABI. With exceptions enabled, the compiler already rejects an unmatched ``try`` /
``catch`` pair. This lightweight source check can catch macro-count mismatches before a
build, but does not prove that every function has an exception boundary.

This script counts, per ``.cc`` file, the opening macros vs ``API_IMPL_END`` and fails if
they are unbalanced. Macro *definitions* (``#define`` lines) are ignored. Equal counts
within a file do not establish correct pairing within individual functions.

Usage::

    python tools/ci_build/check_capi_exception_boundary.py
    python tools/ci_build/check_capi_exception_boundary.py --root onnxruntime/core/session --list

Exit code is non-zero when any file is unbalanced.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_OPENING_RE = re.compile(r"\b(?:API_IMPL_BEGIN|TENSOR_READ_API_BEGIN|TENSOR_READWRITE_API_BEGIN)\b")
_CLOSING_RE = re.compile(r"\bAPI_IMPL_END\b")


def count_boundaries(text: str) -> tuple[int, int]:
    """Return (opening_count, closing_count) over call sites, ignoring #define bodies.

    Multi-line macro definitions (e.g. TENSOR_READ_API_BEGIN, which itself contains
    API_IMPL_BEGIN on a backslash-continuation line) are skipped in full so the macro
    definitions are not miscounted as call sites.
    """
    opens = closes = 0
    in_define = False
    for line in text.splitlines():
        if in_define:
            in_define = line.rstrip().endswith("\\")
            continue
        if line.lstrip().startswith("#define"):
            in_define = line.rstrip().endswith("\\")
            continue
        # Strip line comments so macro names mentioned in comments are not counted.
        code = line.split("//", 1)[0]
        opens += len(_OPENING_RE.findall(code))
        closes += len(_CLOSING_RE.findall(code))
    return opens, closes


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "onnxruntime" / "core" / "session",
        help="Directory to scan for C-API .cc files (relative to the repository; default: onnxruntime/core/session).",
    )
    parser.add_argument("--list", action="store_true", help="Print per-file counts for files that use the macros.")
    args = parser.parse_args()
    args.root = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()

    if not args.root.is_dir():
        print(f"error: root directory not found: {args.root}", file=sys.stderr)
        return 2

    unbalanced: list[str] = []
    checked = 0
    for path in sorted(args.root.rglob("*.cc")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - defensive I/O guard
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        opens, closes = count_boundaries(text)
        if opens == 0 and closes == 0:
            continue  # file does not use the boundary macros
        checked += 1
        rel = display_path(path, repo_root)
        if args.list:
            print(f"{rel}: open={opens} close={closes}")
        if opens != closes:
            unbalanced.append(f"{rel}: {opens} opening macro(s) vs {closes} API_IMPL_END")

    print(f"Checked {checked} C-API source file(s) under {display_path(args.root, repo_root)}.")

    if unbalanced:
        print(
            "error: unbalanced C-API exception-boundary macro counts:",
            file=sys.stderr,
        )
        for item in unbalanced:
            print(f"  {item}", file=sys.stderr)
        print(
            "Check that each API_IMPL_BEGIN (or TENSOR_*_API_BEGIN) is paired with API_IMPL_END "
            "in the same function body.",
            file=sys.stderr,
        )
        return 1

    return 0


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
