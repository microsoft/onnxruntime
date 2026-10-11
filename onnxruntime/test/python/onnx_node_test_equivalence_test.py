#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Retiring equivalence test for materialize_onnx_node_tests.py.

Proves the build-time-materialized node-test tree is byte-identical to ONNX's
still-present on-disk golden corpus (``.../onnx/backend/test/data/node``).

Two complementary equivalence guards exist -- this is one of them; they are NOT
redundant (different failure modes, different CI legs):

  * THIS unittest -- materializes FRESH into a tmpdir and byte-compares vs the
    oracle. Validates MATERIALIZER / GENERATOR-LOGIC correctness in isolation,
    with NO C++ build required, so it runs in the Python test legs and as a fast
    local dev check. It also carries the Python-side min-count floor
    (``MIN_NODE_CASES``) -- the Python twin of the C++ runtime tripwire.
  * The ``onnx_node_tests_equivalence`` CTEST (``compare_node_test_corpora.py``)
    -- byte-compares the oracle vs the ACTUAL build-tree artifact that
    ``onnx_test_runner`` consumes. Validates ARTIFACT INTEGRITY end-to-end
    (catches CMake wiring drift, stale/partial artifacts) in the C++ build legs.

Both auto-retire together when the oracle vanishes post-#7959.

This test is an ORACLE-DEPENDENT guard: it can only certify equivalence while
the on-disk corpus exists. Once ONNX PR onnx/onnx#7959 lands (deleting that
corpus), the oracle is gone and this test AUTO-SKIPS -- the materializer then
becomes the sole source of the node tree, and the permanent min-count tripwire
(in the C++ runner + the Python backend series) takes over as the cause-agnostic
backstop.

Version parity: the materialized model.onnx opset/IR is stamped from the
compiled onnx schema registry, so this test asserts the installed onnx equals
the ``cmake/deps.txt`` pin (compared on the release base, so a pre-release rcN/.dev
wheel of the pinned tag is accepted). numpy parity is NOT asserted -- the core
treats numpy as an informational soft-warning, and recomputed float outputs that
differ by within a small ULP band are accepted as Class A numpy-gen skew.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest

# Make the sibling materializer importable regardless of CWD.
_TOOLS_PYTHON = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "python"))
if _TOOLS_PYTHON not in sys.path:
    sys.path.insert(0, _TOOLS_PYTHON)

import materialize_onnx_node_tests as mat  # noqa: E402

# A conservative floor: today's corpus is ~1799 node cases. The floor exists so a
# silently-empty/undersized materialization fails loud rather than shipping green.
MIN_NODE_CASES = 1500


def _repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _expected_onnx_version() -> str | None:
    """Parse the onnx pin (archive tag) from cmake/deps.txt."""
    deps = os.path.join(_repo_root(), "cmake", "deps.txt")
    if not os.path.isfile(deps):
        return None
    with open(deps) as f:
        for line in f:
            if line.startswith("onnx;"):
                m = re.search(r"/tags/v([0-9]+\.[0-9]+\.[0-9]+)\.zip", line)
                if m:
                    return m.group(1)
    return None


def _find_oracle_node_dir() -> str | None:
    """Locate the on-disk ONNX node corpus (the oracle), if present.

    Search order: the FetchContent build tree (_deps/onnx-src), an env override,
    then the installed onnx wheel's bundled data. Returns None if the corpus is
    gone (post-#7959) -> the test auto-skips.
    """
    candidates = []
    env = os.environ.get("ORT_ONNX_NODE_ORACLE_DIR")
    if env:
        candidates.append(env)
    root = _repo_root()
    # Candidate build-tree oracle locations mirror the ORT convention
    # build/<Platform>/<Config> layout from tools/ci_build (Platform in
    # {Linux,Windows,MacOS}, Config in {Debug,Release,RelWithDebInfo,MinSizeRel};
    # "" covers a flat build/).
    for build in ("build", "build/Linux", "build/Windows", "build/MacOS"):
        for cfg in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel", ""):
            candidates.append(
                os.path.join(
                    root,
                    build,
                    cfg,
                    "_deps",
                    "onnx-src",
                    "onnx",
                    "backend",
                    "test",
                    "data",
                    "node",
                )
            )
    try:
        import onnx.backend.test  # noqa: PLC0415

        candidates.append(os.path.join(os.path.dirname(onnx.backend.test.__file__), "data", "node"))
    except ImportError:
        # onnx (or its backend.test data package) isn't importable in this env, so
        # the installed-wheel oracle location simply doesn't contribute a candidate.
        # Other candidate sources (env override, build tree) still apply.
        pass
    for c in candidates:
        if c and os.path.isdir(c) and any(n.startswith("test_") for n in os.listdir(c)):
            return c
    return None


class NodeTestEquivalence(unittest.TestCase):
    def test_materialized_equals_on_disk_oracle(self) -> None:
        oracle = _find_oracle_node_dir()
        if oracle is None:
            self.skipTest(
                "on-disk ONNX node corpus (oracle) not found -- either not built yet, "
                "or onnx#7959 landed and the corpus was removed. Equivalence oracle "
                "retired; the materializer + min-count tripwire are now the guard."
            )

        expected_onnx = _expected_onnx_version()
        try:
            import onnx  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"onnx not importable: {exc}")

        if expected_onnx and mat._release_base(onnx.__version__) != mat._release_base(expected_onnx):
            self.skipTest(
                f"installed onnx=={onnx.__version__} != cmake/deps.txt pin "
                f"{expected_onnx}; cannot certify equivalence under version skew. "
                f"Install the pinned onnx to run this check. (A pre-release rcN/.dev "
                f"of the same release base is accepted.)"
            )

        with tempfile.TemporaryDirectory(prefix="ort_node_mat_") as tmp:
            # numpy parity is intentionally NOT asserted here: the core treats it as
            # an informational soft-warning, and this test validates generator LOGIC
            # (not corpus-gen numpy identity), so pass expected_numpy=None.
            written = mat.materialize(
                tmp,
                expected_onnx=onnx.__version__,
                expected_numpy=None,
                min_cases=MIN_NODE_CASES,
            )
            self.assertGreaterEqual(
                len(written),
                MIN_NODE_CASES,
                f"materialized only {len(written)} node cases (< {MIN_NODE_CASES})",
            )
            # compare_to_oracle raises MaterializeError (Class B) on any structural
            # mismatch; Class A small-ULP float drift is accepted + warned.
            try:
                mat.compare_to_oracle(
                    tmp,
                    os.path.dirname(oracle),  # parent of .../node
                    min_cases=MIN_NODE_CASES,
                )
            except mat.MaterializeError as exc:
                self.fail(str(exc))


if __name__ == "__main__":
    unittest.main()
