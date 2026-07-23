#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Map CUDA kernel source files to the operators they register and, given an
operator allow-list, report the source files that can be excluded from the build.

Used by the CUDA plugin EP build (cmake/onnxruntime_providers_cuda_plugin.cmake)
to physically drop kernel sources whose operators are not in the allow-list, so
their code is never compiled (real binary-size reduction).

A source file is excluded only when ALL of the following hold:
  * it registers at least one operator (via an ONNX_OPERATOR_*_KERNEL_EX macro),
  * every operator it registers can be resolved to a literal op type, and
  * none of those op types are in the allow-list.

Files whose registrations use a macro-parameter op name (wrapper macros) cannot
be resolved reliably and are kept, erring on the side of a correct build.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches e.g. ONNX_OPERATOR_TYPED_KERNEL_EX(RMSNormalization, ...) and captures
# the first argument (the operator type, or a macro parameter for wrapper macros).
_KERNEL_EX_RE = re.compile(
    r"ONNX_OPERATOR_(?:VERSIONED_)?(?:TWO_|THREE_)?(?:TYPED_)?KERNEL_EX\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)"
)


def parse_allowlist(path: Path) -> set[str]:
    ops: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ops.add(line)
    return ops


def file_registered_ops(path: Path) -> tuple[set[str], bool]:
    """Return (op_types, ambiguous).

    ambiguous is True when a registration uses a macro-parameter op name
    (lower-case first arg such as ``name``/``op_name``), meaning the file's real
    operators cannot be determined from a simple scan and it must be kept.
    """
    ops: set[str] = set()
    ambiguous = False
    text = path.read_text(errors="ignore")
    for match in _KERNEL_EX_RE.finditer(text):
        first_arg = match.group(1)
        # ONNX / contrib operator types are UpperCamelCase. A lower-case first
        # argument is a macro parameter from a wrapper macro -> unresolvable.
        if first_arg[0].islower():
            ambiguous = True
        else:
            ops.add(first_arg)
    return ops, ambiguous


def compute_excludable(sources: list[str], allowed: set[str]) -> list[str]:
    """Return the subset of `sources` (original path strings) that can be excluded."""
    excludable: list[str] = []
    for src in sources:
        path = Path(src)
        if path.suffix != ".cc" or not path.is_file():
            continue
        ops, ambiguous = file_registered_ops(path)
        if ambiguous or not ops:
            continue  # keep: helper file or unresolvable registration
        if ops.isdisjoint(allowed):
            excludable.append(src)
    return excludable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", required=True, type=Path, help="Operator allow-list file.")
    parser.add_argument(
        "--source-list-file",
        required=True,
        type=Path,
        help="File containing newline- or semicolon-separated source paths.",
    )
    args = parser.parse_args()

    allowed = parse_allowlist(args.allowlist)
    raw = args.source_list_file.read_text().replace(";", "\n")
    sources = [s.strip() for s in raw.splitlines() if s.strip()]

    excludable = compute_excludable(sources, allowed)

    total_cc = sum(1 for s in sources if s.endswith(".cc"))
    print(
        f"[cuda_op_allowlist_filter] allow-list ops={len(allowed)} "
        f".cc sources={total_cc} excluded={len(excludable)}",
        file=sys.stderr,
    )
    for src in excludable:
        print(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
