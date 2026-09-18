#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Ratchet on the number of GoogleTest ``DISABLED_`` tests.

GoogleTest silently excludes any test whose name is prefixed with ``DISABLED_``
(only a run-time stderr note is emitted), so disabled tests rot indefinitely with
no owner or linked issue. This script counts them and fails when the total grows
beyond a baseline, so the count can only go down over time.

Usage::

    python tools/ci_build/check_disabled_tests.py            # check against baseline
    python tools/ci_build/check_disabled_tests.py --list     # also list every site
    python tools/ci_build/check_disabled_tests.py --update-baseline   # print new baseline

Exit code is non-zero when the count exceeds ``--baseline`` (CI gate), or when the
count drops below it (a reminder to lower the baseline so the ratchet stays tight).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches a disabled test declaration, e.g. ``TEST_F(SuiteName, DISABLED_Foo)``.
# Mirrors GoogleTest's TEST / TEST_F / TEST_P / TYPED_TEST(_P) macros.
_DISABLED_RE = re.compile(r"\b(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(\s*\w+\s*,\s*DISABLED_\w*")

# Default baseline measured from the working tree (TEST/TEST_F/TEST_P/TYPED_TEST*
# with a DISABLED_ name). Lower this (never raise it) whenever disabled tests are
# re-enabled or removed.
_DEFAULT_BASELINE = 170

_SOURCE_SUFFIXES = (".cc", ".cpp", ".cxx", ".cu")


def find_disabled_tests(test_dir: Path) -> list[tuple[Path, int, str]]:
    """Return (path, line_number, matched_text) for every disabled test under test_dir."""
    hits: list[tuple[Path, int, str]] = []
    for path in sorted(test_dir.rglob("*")):
        if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - defensive I/O guard
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = _DISABLED_RE.search(line)
            if match:
                hits.append((path, lineno, match.group(0).strip()))
    return hits


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=repo_root / "onnxruntime" / "test",
        help="Directory to scan for disabled tests (default: onnxruntime/test).",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=_DEFAULT_BASELINE,
        help=f"Maximum allowed disabled tests (default: {_DEFAULT_BASELINE}).",
    )
    parser.add_argument("--list", action="store_true", help="Print every disabled test site.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Print the suggested baseline line for the current count and exit 0.",
    )
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        print(f"error: test directory not found: {args.test_dir}", file=sys.stderr)
        return 2

    hits = find_disabled_tests(args.test_dir)
    count = len(hits)
    file_count = len({path for path, _, _ in hits})

    if args.list:
        for path, lineno, text in hits:
            print(f"{path.relative_to(repo_root).as_posix()}:{lineno}: {text}")

    print(f"Disabled tests: {count} across {file_count} files (baseline {args.baseline}).")

    if args.update_baseline:
        print(f"Suggested: _DEFAULT_BASELINE = {count}")
        return 0

    if count > args.baseline:
        print(
            f"error: disabled-test count increased to {count} (baseline {args.baseline}). "
            "Re-enable a test or fix the regression before adding new DISABLED_ tests.",
            file=sys.stderr,
        )
        return 1

    if count < args.baseline:
        print(
            f"note: disabled-test count dropped to {count}; lower the baseline to {count} "
            "in tools/ci_build/check_disabled_tests.py to keep the ratchet tight.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
