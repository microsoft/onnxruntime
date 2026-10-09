# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Golden tests against the real in-tree wgsl.template files.

These templates back shipped WebGPU operators (Pad, Transpose,
im2col-matmul, and others under core/providers/webgpu). The generated
C++ is compared against a committed golden snapshot so generator
regressions are caught automatically rather than relying on a manual
GPU run. The comparison uses ``canonicalize`` (see golden_compare).

When an in-tree template changes (or one is added), regenerate the
goldens and commit them::

    UPDATE_WGSL_GOLDEN=1 python wgsl_template/test/run_tests.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from golden_compare import canonicalize, read_tree  # noqa: E402
from wgsl_template import build  # noqa: E402

# Repo root: tools/python/wgsl_template/test/ -> ../../../..
_REPO_ROOT = _THIS_DIR.parent.parent.parent.parent
# WebGPU EP source root, where the real templates live.
_WEBGPU_ROOT = _REPO_ROOT / "onnxruntime" / "core" / "providers" / "webgpu"
# Committed golden snapshots of the generated output.
_GOLDEN_DIR = _THIS_DIR / "in_tree_golden"

# Set UPDATE_WGSL_GOLDEN=1 to (re)write the golden snapshots instead of
# comparing against them.
_UPDATE_GOLDEN = os.environ.get("UPDATE_WGSL_GOLDEN") == "1"

_REGEN_HINT = "UPDATE_WGSL_GOLDEN=1 python wgsl_template/test/run_tests.py"


def _generate(generator: str) -> Path:
    out = Path(tempfile.mkdtemp(prefix="wgsl_smoke_"))
    try:
        build(
            source_dirs=[_WEBGPU_ROOT],
            out_dir=out,
            generator=generator,
            include_path_prefix="wgsl_template_gen/",
            preserve_code_reference=True,
        )
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out


def _write_golden(generator: str, out_dir: Path) -> None:
    dest = _GOLDEN_DIR / generator
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(out_dir, dest)


class InTreeTemplatesGoldenTest(unittest.TestCase):
    def _check(self, generator: str) -> None:
        out = _generate(generator)
        try:
            if _UPDATE_GOLDEN:
                _write_golden(generator, out)
                self.skipTest(f"updated golden for {generator}")
                return

            golden = _GOLDEN_DIR / generator
            self.assertTrue(
                golden.is_dir(),
                f"missing golden for {generator}; run `{_REGEN_HINT}` to create it",
            )

            actual = canonicalize(read_tree(out))
            expected = canonicalize(read_tree(golden))

            self.assertEqual(
                sorted(actual),
                sorted(expected),
                f"{generator}: generated file set differs from golden. If an in-tree "
                f"template was added/removed, regenerate with `{_REGEN_HINT}`.",
            )
            for rel in sorted(expected):
                self.assertEqual(
                    actual[rel],
                    expected[rel],
                    f"{generator}: {rel} differs from golden. If the in-tree template "
                    f"changed intentionally, regenerate with `{_REGEN_HINT}`.",
                )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_static_cpp_matches_golden(self) -> None:
        self._check("static-cpp")

    def test_static_cpp_literal_matches_golden(self) -> None:
        self._check("static-cpp-literal")
