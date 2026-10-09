#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Ratchet on the number of GoogleTest ``DISABLED_`` declarations.

GoogleTest silently excludes any test or suite prefixed with ``DISABLED_``
(only a run-time stderr note is emitted), so disabled tests rot indefinitely with
no owner or linked issue. This script counts them and fails when the total grows
beyond a baseline, so the count can only go down over time.

Counts TEST, TEST_F, TEST_P, TYPED_TEST and TYPED_TEST_P declarations in all
preprocessor branches, including multiline declarations and disabled suites.
Comments and string literals are excluded; macro aliases are not expanded.

Usage::

    python tools/ci_build/check_disabled_tests.py            # check against baseline
    python tools/ci_build/check_disabled_tests.py --list     # also list every site
    python tools/ci_build/check_disabled_tests.py --update-baseline   # print new baseline

Exit code is non-zero when the count exceeds ``--baseline`` (CI gate), or when the
count drops below it (a reminder to lower the baseline so the ratchet stays tight).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_TEST_RE = re.compile(r"\b(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")
_NON_CODE_RE = re.compile(
    r'R"([^ ()\\\t\r\n]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
    re.DOTALL,
)

# Initial baseline is frozen against main 3d9d664a45. Count declarations, not
# parameterized instances; inspect all preprocessor branches. Lower this when
# disabled declarations are re-enabled or removed.
_DEFAULT_BASELINE = 188

_SOURCE_SUFFIXES = (".cc", ".cpp", ".cxx", ".cu")


def find_disabled_tests(test_dir: Path) -> list[tuple[Path, int, str]]:
    """Return (path, line_number, matched_text) for every disabled test under test_dir."""

    def raise_walk_error(error: OSError) -> None:
        raise error

    hits: list[tuple[Path, int, str]] = []
    for directory, _, filenames in os.walk(test_dir, onerror=raise_walk_error):
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix not in _SOURCE_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            code = _NON_CODE_RE.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), text)
            for match in _TEST_RE.finditer(code):
                if any(name.startswith("DISABLED_") for name in match.groups()):
                    lineno = code.count("\n", 0, match.start()) + 1
                    declaration = " ".join(match.group().split())
                    hits.append((path, lineno, declaration))
    return sorted(hits)


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
        help=f"Exact expected disabled declaration count (default: {_DEFAULT_BASELINE}).",
    )
    parser.add_argument("--list", action="store_true", help="Print every disabled test site.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Print the suggested baseline line for the current count and exit 0.",
    )
    args = parser.parse_args()
    if args.baseline < 0:
        parser.error("--baseline must be non-negative")
    args.test_dir = args.test_dir.resolve()

    if not args.test_dir.is_dir():
        print(f"error: test directory not found: {args.test_dir}", file=sys.stderr)
        return 2

    try:
        hits = find_disabled_tests(args.test_dir)
    except (OSError, UnicodeError) as exc:
        print(f"error: cannot scan disabled tests: {exc}", file=sys.stderr)
        return 2
    count = len(hits)
    file_count = len({path for path, _, _ in hits})

    if args.list:
        for path, lineno, text in hits:
            display_path = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
            print(f"{display_path.as_posix()}:{lineno}: {text}")

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
