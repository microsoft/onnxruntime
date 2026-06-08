# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers for comparing generated WGSL output against goldens.

Some details of the generated file set are host-dependent and NOT part
of the equivalence contract (see DESIGN_wgsl_python_port.md,
"Determinism"): ``__str_N`` ids, the ordering of constants /
declarations / includes in the aggregate index/table files, and the
per-file ``// <sha256>`` markers. Only the shader string *contents* and
the C++ structure are contractual. :func:`canonicalize` normalizes the
non-contractual details away so goldens stay stable across tools and
hosts, while still comparing shader strings exactly (via their resolved
``__str_N`` values).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Aggregate files whose line *order* is not semantically meaningful;
# compared as unordered line sets. Per-template generated/*.h files keep
# their in-order comparison.
_ORDER_INSENSITIVE_FILES = {"index.h", "index_impl.h", "string_table.h"}

_STR_ID_RE = re.compile(r"__str_(\d+)")
_STR_TABLE_DEF_RE = re.compile(r"__str_(\d+)\s*=\s*(.*);")
_SHA256_COMMENT_RE = re.compile(r" *// [0-9a-f]{64}")


def read_tree(root: Path) -> dict[str, str]:
    """Read every file under ``root`` into a ``{posix_relpath: text}``
    map, normalizing CRLF that a Windows checkout may have introduced."""
    out: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            out[rel] = full.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    return out


def canonicalize(files: dict[str, str]) -> dict[str, str]:
    """Reduce a generated file set to a host-independent canonical form.

    Resolves ``__str_N`` references to their string literals, drops the
    derived sha256 markers, and sorts the order-insensitive aggregate
    files.
    """
    string_table = files.get("string_table.h", "")
    id_to_literal = {m.group(1): m.group(2) for m in _STR_TABLE_DEF_RE.finditer(string_table)}

    canonical: dict[str, str] = {}
    for rel, text in files.items():
        resolved = _STR_ID_RE.sub(lambda m: id_to_literal.get(m.group(1), m.group(0)), text)
        resolved = _SHA256_COMMENT_RE.sub("", resolved)
        if rel.rsplit("/", 1)[-1] in _ORDER_INSENSITIVE_FILES:
            resolved = "\n".join(sorted(resolved.split("\n")))
        canonical[rel] = resolved
    return canonical
