#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Byte-equivalence comparator for ONNX node-test corpora.

This is the CANONICAL, build-wired equivalence check (the ``onnx_node_tests_equivalence``
ctest invokes it): it compares an ALREADY-materialized node-test tree against the
on-disk ONNX oracle corpus (``.../onnx/backend/test/data/node``, present only below
onnx/onnx#7959) WITHOUT regenerating anything, so it validates the exact artifact
``onnx_test_runner`` consumes and cannot race the build-time materialize step.

It is a thin wrapper over the byte/classifier core in
``materialize_onnx_node_tests.py`` (``compare_to_oracle``), so there is ZERO logic
duplication. Two sibling entry points share that same core and are documented as
aliases of THIS canonical check, not independent implementations:
  * ``materialize_onnx_node_tests.py --oracle-dir`` -- a dev-convenience one-shot
    (materialize + compare) for local use.
  * ``onnxruntime/test/python/onnx_node_test_equivalence_test.py`` -- the Python
    unittest that materializes fresh into a tmpdir to validate generator LOGIC in
    isolation (Python CI legs); this script validates ARTIFACT INTEGRITY (C++ legs).

Equivalence relation (see ``compare_to_oracle``):
  * dir-set equality (both directions),
  * ``model.onnx`` + ``input_*.pb`` byte-identical (hard-fail on diff), EXCEPT
    ``input_*.pb`` of the ``test_image_decoder_*`` family -- those are encoded-image
    blobs whose bytes are codec/env-dependent (and the family is globally excluded
    downstream), so input-byte divergence there is an accepted, logged, non-fatal class,
  * ``output_*.pb`` byte-identical OR within the Class-A float-ULP band (numpy-gen
    skew -> logged warning), else hard-fail,
  * a min-count floor so an empty/undersized tree fails loud.

Exit 0 == equivalent (modulo Class A warnings); exit 1 == Class B structural
mismatch, onnx version skew, dir-set mismatch, or under the floor.
"""

from __future__ import annotations

import argparse
import os
import sys

# Import the byte/classifier core from the sibling materializer.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import materialize_onnx_node_tests as mat


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Byte-compare a materialized node-test tree against the on-disk ONNX oracle.",
    )
    p.add_argument(
        "--oracle",
        required=True,
        help="On-disk ONNX oracle: the data/node dir (or a parent containing node/).",
    )
    p.add_argument(
        "--materialized",
        required=True,
        help="Materialized tree: the onnx_node_tests/node dir (or a parent containing node/).",
    )
    p.add_argument(
        "--expected-onnx-version",
        required=True,
        help="Assert onnx.__version__ equals this (the cmake/deps.txt pin) before comparing.",
    )
    p.add_argument(
        "--expected-numpy-version",
        default=None,
        help="Informational/forensic only: logs a WARNING (never fails) on numpy skew.",
    )
    p.add_argument(
        "--min-cases",
        type=int,
        default=1,
        help="Fail loud if fewer than this many cases are present in the materialized tree.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        # Version parity first: a skewed wheel produces confusing Class B noise.
        mat._check_versions(args.expected_onnx_version, args.expected_numpy_version)
        mat.compare_to_oracle(
            args.materialized,
            args.oracle,
            min_cases=args.min_cases,
        )
    except mat.MaterializeError as exc:
        print(f"[compare_node_test_corpora] {exc}", file=sys.stderr)
        return 1
    print(
        f"[compare_node_test_corpora] EQUIVALENCE OK: materialized tree "
        f"{args.materialized} is byte-identical to oracle {args.oracle} "
        f"(modulo <= {mat._CLASS_A_MAX_ULP} ULP Class A float drift)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
