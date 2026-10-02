#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Materialize ONNX node-test artifacts from ONNX's Python generators.

ONNX PR onnx/onnx#7959 deletes the on-disk node-test corpus
(``onnx/backend/test/data/node/**``) that ONNX Runtime's C++ ``onnx_test_runner``
consumes as a golden oracle for operator kernels. This script detaches ORT from
those shipped ``.pb`` artifacts by regenerating an identical tree at build time
from ONNX's *surviving* Python generators (``collect_testcases``), so ORT keeps
its per-operator guard regardless of what ONNX ships on disk.

Entry points:

* ``materialize`` (default) -- write ``<out>/node/<case>/model.onnx`` +
  ``test_data_set_*/{input,output}_k.pb`` mirroring ONNX's exact on-disk layout.
* ``--oracle-dir`` -- a DEV-CONVENIENCE alias that materializes and then
  byte-compares in one shot. The CANONICAL equivalence check wired into the
  build is the sibling ``compare_node_test_corpora.py`` (which compares the
  ALREADY-materialized build artifact against the oracle without regenerating
  it); this ``--oracle-dir`` path exists only for a quick one-command local
  check and shares the same ``compare_to_oracle`` core, so the two never drift.
  Both classify any mismatch (Class A = <=4-ULP numpy-recompute drift;
  Class B = structural onnx-version skew) and both retire when the oracle is gone.

MANDATORY correctness guards (run before any generation):
  * ``onnx.__version__ == --expected-onnx-version`` -- HARD FAIL. The model
    opset/IR is stamped from the *compiled* schema registry
    (``get_schema().since_version``), so a mismatched wheel silently bakes the
    wrong opset into ``model.onnx``. Compared on the release base so a pre-release
    (``rcN`` / ``.dev``) wheel of the pinned tag does not false-FATAL.
  * ``numpy.__version__`` vs ``--expected-numpy-version`` -- SOFT WARNING only,
    never a hard gate. ONNX's own corpus-gen numpy is uncontrollable/unknowable,
    so a fixed pin cannot guarantee byte-identity and a FATAL would false-red on
    a legitimate numpy. numpy correctness rests on requirements.txt + the Class-A
    ULP band pre-#7959, and on requirements.txt alone post-retirement. The flag
    is an informational/forensic value, not an assert.
  * ``len(cases) >= --min-cases`` -- a silently-empty materialization
    (e.g. an API change that makes ``collect_testcases`` return nothing) must
    fail loud instead of shipping an un-guarded, still-green test suite.

The writer body (the 4-branch container dispatch on the graph value type:
map -> from_dict, sequence -> from_list, optional -> from_optional,
tensor -> from_array / SerializeToString) is vendored **verbatim** from ONNX's
``onnx/backend/test/cmd_tools.py::generate_data`` -- the exact routine #7959
removes. It must NOT be simplified to a ``from_array``-only path: doing so would
silently mis-serialize Sequence / Map / Optional fixtures.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class MaterializeError(RuntimeError):
    """Raised on a fail-loud condition (version skew, empty corpus, arity mismatch)."""


# Class A (accepted numpy-gen skew) band for recomputed FLOAT OUTPUT tensors.
# When model.onnx and all inputs are byte-identical, any output difference is by
# construction pure numpy-recompute drift of the reference output. Inverse-trig /
# transcendental references (acos/acosh/asin/asinh/atan/atanh) legitimately differ
# by a few ULP across numpy versions -- empirically 1-2 ULP for the 6 inverse-trig
# node cases in the v1.22.0 corpus. Band at 4 (2x the observed max) for margin:
# tight enough that a real structural/opset regression (which changes model.onnx,
# shape/dtype, or produces large value drift) blows well past it and stays Class B.
_CLASS_A_MAX_ULP = 4


# Accepted-diff class for INPUT artifacts of the image_decoder test family.
# image_decoder inputs are ENCODED-IMAGE blobs (jpeg/jpeg2k/webp/png/bmp/pnm/tiff),
# whose exact bytes are codec-/environment-dependent, NOT opset/IR drift. The
# materializer is deterministic run-to-run (proven), so a byte-divergence here is
# only ever against the historical on-disk oracle -- a known-non-reproducible class,
# analogous to the Class-A float-ULP band. All 9 of these cases are ALSO globally
# excluded in the Python backend series (onnx_backend_test_series filters.jsonc), so
# their exact input bytes never affect a real test outcome. We therefore ACCEPT
# input-artifact byte-diffs for this family (dir-set must still match; the case is
# still materialized), while keeping every other input strictly byte-identical.
_IMAGE_DECODER_CASE_PREFIX = "test_image_decoder_"


# --------------------------------------------------------------------------- #
# Version / count guards
# --------------------------------------------------------------------------- #
def _release_base(version: str) -> str:
    """Return the release base (major.minor.micro) of a possibly-prerelease version.

    The ONNX opset-bump workflow ships release-candidate / dev wheels like
    ``1.23.0rc1`` or ``1.23.0.dev20240101`` whose COMPILED opset registry already
    matches the formal ``1.23.0`` tag. Comparing on the release base lets an RC
    wheel of the pinned tag pass, while a genuine major/minor/micro mismatch
    (e.g. ``1.22.0`` vs ``1.23.0``) still fails loud.
    """
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", version)
    return ".".join(m.groups()) if m else version.strip()


def _check_versions(expected_onnx: str, expected_numpy: str | None) -> None:
    """Assert wheel parity BEFORE importing/collecting cases.

    Must run before ``collect_testcases`` because ``expect()`` stamps the model
    opset from the compiled schema registry at collect time.

    onnx is a HARD gate (a mismatched wheel bakes the wrong opset); numpy is a
    SOFT warning only (see below).

    CAVEAT: the onnx gate compares ``onnx.__version__`` and so assumes that string
    is truthful. An editable / ``-e`` / dev onnx install can report a STALE
    ``__version__`` that disagrees with the actually-compiled schema registry,
    silently defeating the assert. Normal pip-wheel CI installs (what the build
    uses) report a truthful version, so this is a local-dev caveat, not a CI gap.
    """
    import numpy  # noqa: PLC0415
    import onnx  # noqa: PLC0415

    # onnx: HARD FAIL, RC/pre-release aware (compare release base, not the raw
    # string) so an rcN/dev wheel of the pinned tag isn't a false FATAL.
    if expected_onnx and _release_base(onnx.__version__) != _release_base(expected_onnx):
        raise MaterializeError(
            f"onnx version mismatch: installed onnx=={onnx.__version__} but "
            f"--expected-onnx-version=={expected_onnx} (release base "
            f"{_release_base(onnx.__version__)} != {_release_base(expected_onnx)}). "
            f"The node-test corpus opset/IR is baked from the COMPILED onnx schema "
            f"registry, so a mismatched wheel produces a silently-drifted corpus. "
            f"Install onnx=={expected_onnx} (the cmake/deps.txt pin) before materializing. "
            f"A pre-release (rcN/.dev) of the SAME release base is accepted."
        )
    # numpy: SOFT WARNING, never a hard gate. ONNX's OWN corpus-gen numpy is
    # uncontrollable and unknowable, so a fixed pin can neither guarantee
    # byte-identity nor justify a FATAL (it would false-red on a legitimate
    # numpy). numpy correctness rests on requirements.txt + the Class-A ULP band
    # pre-#7959, and on requirements.txt alone post-retirement. The value is
    # informational/forensic -- treat --expected-numpy-version as advisory.
    if expected_numpy and numpy.__version__ != expected_numpy:
        print(
            f"[materialize_onnx_node_tests] WARNING: numpy version skew: installed "
            f"numpy=={numpy.__version__} but --expected-numpy-version=={expected_numpy}. "
            f"Informational only -- recomputed float outputs are numpy-sensitive, but the "
            f"Class-A ULP band absorbs small drift and the corpus-gen numpy is set by ONNX, "
            f"not ORT. Not failing.",
            file=sys.stderr,
        )


def _collect_node_cases() -> list:
    """Return all node test cases (op_type=None) via ONNX's public collector.

    ``collect_testcases`` self-invokes ``import_recursive`` so this is
    self-contained and era-agnostic (works pre- and post-#7959).

    Post-#7959 (ONNX >= 1.23), the blessed public accessor is
    ``onnx.backend.test.loader.load_model_tests(kind="node")``, which returns the
    same in-memory ``TestCase`` list -- so only this collection call swaps, the
    writer below is unaffected. It does NOT exist on the pinned onnx 1.22, so do
    not import it here yet; swap the import if ``collect_testcases`` is removed.
    """
    from onnx.backend.test.case.node import collect_testcases  # noqa: PLC0415

    return list(collect_testcases(None))


# --------------------------------------------------------------------------- #
# Writer -- vendored VERBATIM from onnx/backend/test/cmd_tools.py::generate_data
# (the 4-branch container dispatch, inputs + outputs). Do NOT simplify.
# --------------------------------------------------------------------------- #
def _prepare_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def _write_case(case, output_root: str) -> None:
    from onnx import TensorProto, numpy_helper  # noqa: PLC0415

    # case.name already carries the "test_" prefix; case.kind == "node".
    output_dir = os.path.join(output_root, case.kind, case.name)
    _prepare_dir(output_dir)

    if not case.model:
        raise MaterializeError(f"node case {case.name!r} has no model")
    with open(os.path.join(output_dir, "model.onnx"), "wb") as f:
        f.write(case.model.SerializeToString())

    if not case.data_sets:
        raise MaterializeError(f"node case {case.name!r} has no data_sets")
    n_graph_in = len(case.model.graph.input)
    n_graph_out = len(case.model.graph.output)
    for i, (inputs, outputs) in enumerate(case.data_sets):
        # Fail loud BEFORE writing any .pb for this dataset: the writer indexes
        # graph.input[j]/graph.output[j] per collected value, so more collected
        # values than graph arity would raise a bare IndexError mid-write (not
        # caught by main's MaterializeError handler) and leave a partial tree.
        if len(inputs) > n_graph_in or len(outputs) > n_graph_out:
            raise MaterializeError(
                f"node case {case.name!r} data_set {i}: arity mismatch -- collected "
                f"{len(inputs)} input(s)/{len(outputs)} output(s) but the model graph "
                f"declares {n_graph_in} input(s)/{n_graph_out} output(s). The ONNX "
                f"generator produced a fixture the model shape cannot address; refusing "
                f"to write a partial/misaligned tree."
            )
        data_set_dir = os.path.join(output_dir, f"test_data_set_{i}")
        _prepare_dir(data_set_dir)
        # `input` shadows a builtin but is VERBATIM from ONNX cmd_tools.py -- do not rename.
        for j, input in enumerate(inputs):
            with open(os.path.join(data_set_dir, f"input_{j}.pb"), "wb") as f:
                if case.model.graph.input[j].type.HasField("map_type"):
                    f.write(numpy_helper.from_dict(input, case.model.graph.input[j].name).SerializeToString())
                elif case.model.graph.input[j].type.HasField("sequence_type"):
                    f.write(numpy_helper.from_list(input, case.model.graph.input[j].name).SerializeToString())
                elif case.model.graph.input[j].type.HasField("optional_type"):
                    f.write(numpy_helper.from_optional(input, case.model.graph.input[j].name).SerializeToString())
                else:
                    assert case.model.graph.input[j].type.HasField("tensor_type")
                    if isinstance(input, TensorProto):
                        f.write(input.SerializeToString())
                    else:
                        f.write(numpy_helper.from_array(input, case.model.graph.input[j].name).SerializeToString())
        # `output` shadows a builtin but is VERBATIM from ONNX cmd_tools.py -- do not rename.
        for j, output in enumerate(outputs):
            with open(os.path.join(data_set_dir, f"output_{j}.pb"), "wb") as f:
                if case.model.graph.output[j].type.HasField("map_type"):
                    f.write(numpy_helper.from_dict(output, case.model.graph.output[j].name).SerializeToString())
                elif case.model.graph.output[j].type.HasField("sequence_type"):
                    f.write(numpy_helper.from_list(output, case.model.graph.output[j].name).SerializeToString())
                elif case.model.graph.output[j].type.HasField("optional_type"):
                    f.write(numpy_helper.from_optional(output, case.model.graph.output[j].name).SerializeToString())
                else:
                    assert case.model.graph.output[j].type.HasField("tensor_type")
                    if isinstance(output, TensorProto):
                        f.write(output.SerializeToString())
                    else:
                        f.write(numpy_helper.from_array(output, case.model.graph.output[j].name).SerializeToString())


def materialize(
    output_root: str,
    expected_onnx: str,
    expected_numpy: str | None = None,
    min_cases: int = 1,
) -> list[str]:
    """Materialize the node-test tree under ``<output_root>/node/``.

    Returns the sorted list of case names written. Raises ``MaterializeError``
    on version skew, empty corpus, or an arity mismatch.
    """
    _check_versions(expected_onnx, expected_numpy)
    cases = _collect_node_cases()
    if not cases:
        # Total generator breakage (e.g. an ONNX API change): fail loud BEFORE
        # touching disk so we never destroy an existing tree over an empty collect.
        raise MaterializeError(
            "collect_testcases returned no cases at all -- the ONNX node-test "
            "generator API likely changed. Refusing to touch the output tree."
        )
    node_root = os.path.join(output_root, "node")
    if os.path.exists(node_root):
        shutil.rmtree(node_root)
    os.makedirs(node_root, exist_ok=True)
    written = []
    for case in cases:
        if case.kind != "node":
            continue
        _write_case(case, output_root)
        written.append(case.name)
    written.sort()
    # Authoritative floor: assert on the WRITTEN node count (post kind-filter), not
    # the pre-filter collected count. A silently empty/undersized materialization
    # would ship an un-guarded, still-green test suite.
    if len(written) < min_cases:
        raise MaterializeError(
            f"only {len(written)} node case(s) written (< floor {min_cases}); "
            f"{len(cases)} case(s) were collected before the kind filter. Refusing to "
            f"ship an un-guarded, still-green node-test tree."
        )
    return written


# --------------------------------------------------------------------------- #
# Equivalence check -- byte compare materialized tree vs on-disk oracle
# --------------------------------------------------------------------------- #
def _list_case_dirs(node_root: str) -> set[str]:
    if not os.path.isdir(node_root):
        return set()
    return {name for name in os.listdir(node_root) if os.path.isdir(os.path.join(node_root, name))}


def _resolve_node_dir(path: str) -> str:
    """Normalize a corpus path to the ``node/`` directory.

    Accepts either the ``node/`` dir itself (contains ``test_*`` case dirs) or a
    parent that contains a ``node/`` subdir. Lets callers pass either form.
    """
    if os.path.isdir(path) and any(name.startswith("test_") for name in os.listdir(path)):
        return path
    return os.path.join(path, "node")


def _iter_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            yield os.path.relpath(os.path.join(dirpath, fn), root)


def _classify_tensor_diff(a_path: str, b_path: str) -> tuple[str, str]:
    """Classify a mismatching .pb pair. Returns (class, human message).

    Class A: recomputed float tensor within the numpy-gen ULP band (accepted,
    pass-with-warning). Class B: structural / out-of-band / non-float /
    unparseable (hard-fail).
    """
    try:
        import numpy as np  # noqa: PLC0415
        from onnx import TensorProto, numpy_helper  # noqa: PLC0415

        def _load(p):
            with open(p, "rb") as f:
                t = TensorProto()
                t.ParseFromString(f.read())
            return numpy_helper.to_array(t)

        a = _load(a_path)
        b = _load(b_path)
        if a.shape != b.shape or a.dtype != b.dtype:
            return "B", f"shape/dtype differ: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        if np.array_equal(a, b):
            # equal arrays w/ differing bytes => nondeterministic serialization;
            # fail to preserve byte-reproducibility.
            # Bytes differ yet decoded arrays are equal -> the PROTO ENCODING
            # drifted (field order, raw_data vs typed field, default-value
            # elision). Byte-identity is the contract, so this is a real (Class B)
            # failure, not tolerable numpy float skew. NOTE: a pure +0.0/-0.0
            # flip lands here by design -- ``np.array_equal`` treats the zeros as
            # equal while the bytes differ, so it is (correctly) a byte/proto
            # drift, NOT tolerable skew. The signed-zero merging in
            # ``_max_ulp_diff`` is for MIXED tensors (a co-occurring signed zero
            # must not inflate the ULP of a genuine sub-band skew); it is not a
            # claim that a pure sign-of-zero flip passes.
            return "B", "bytes differ but arrays equal (proto encoding drift or signed-zero flip)"
        if not np.issubdtype(a.dtype, np.floating):
            return "B", "non-float tensor values differ (real regression)"
        with np.errstate(over="ignore", invalid="ignore"):
            ulp = _max_ulp_diff(a, b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # inf/NaN -> All-NaN slice is expected here
                max_abs = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        # A raw-bit ULP metric treats max-finite<->inf and inf<->NaN as adjacent
        # (both are 1 ULP), so a STRUCTURAL finite->inf or inf->NaN output drift
        # (max|d|=inf/nan) would otherwise be mis-blessed as a tolerable <=4-ULP
        # numpy skew. A genuine numpy-gen skew never changes the finite/inf/NaN
        # structure, so require the two arrays to share it -- and the max abs
        # diff to be finite -- before Class A can apply.
        if np.any(np.isfinite(a) != np.isfinite(b)) or np.any(np.isnan(a) != np.isnan(b)) or not np.isfinite(max_abs):
            return "B", f"finite/inf/NaN structure differs (max|d|={max_abs:.3e}); real drift"
        if ulp <= _CLASS_A_MAX_ULP:
            return (
                "A",
                f"float outputs differ by {ulp} ULP (<= {_CLASS_A_MAX_ULP}, max|d|={max_abs:.3e}); numpy-gen skew",
            )
        return "B", f"float outputs differ by {ulp} ULP (max|d|={max_abs:.3e}); real drift"
    except Exception as exc:  # diagnostic path, never the pass criterion
        return "B", f"could not classify (treated as structural): {exc}"


def _max_ulp_diff(a, b) -> int:
    """Max ULP distance between two same-shape float arrays.

    Maps each float to a monotonic "sortable" magnitude key computed in the
    UNSIGNED bit domain, then measures the max absolute key distance. Using a
    sign-magnitude fold (``signmask +/- magnitude``) merges signed zeros (so
    ``+0.0`` and ``-0.0`` are 0 ULP apart) and never overflows -- the previous
    signed ``sign_bit - ia`` fold wrapped int64 for negative operands (e.g. fp32
    ``-0.0`` vs ``+0.0`` reported 2**32 instead of 0). Keys are promoted to
    Python ints (object dtype) before subtraction so even float64 (64-bit) key
    distances cannot overflow.

    The signed-zero merge matters for MIXED tensors: a co-occurring ``+0.0``/
    ``-0.0`` element must not inflate the max ULP of an otherwise genuine
    sub-band skew. It is NOT a claim that a pure signed-zero-only difference is
    tolerated -- that case has no other diff, so ``_classify_tensor_diff``
    catches it upstream via the byte-identity (``np.array_equal``) Class-B gate
    before this metric is ever consulted.

    NOTE: this raw-bit metric is intentionally NOT finite-aware -- max-finite
    and ``inf`` (and ``inf`` and the adjacent ``NaN`` bit pattern) are 1 ULP
    apart here. The caller (``_classify_tensor_diff``) rejects any finite/inf/NaN
    structural mismatch BEFORE applying the Class-A ULP band, so a finite->inf or
    inf->NaN drift can never be mis-accepted despite its small raw-bit distance.

    float16/float32/float64 are handled exactly. Other float kinds (e.g.
    ml_dtypes bfloat16) have no matching fixed-width unsigned view here; the
    caller classifies them as Class B before calling, and the fall-through
    returns an out-of-band sentinel so an accidental call can never look Class A.
    """
    import numpy as np  # noqa: PLC0415

    if a.dtype == np.float16:
        uint, width = np.uint16, 16
    elif a.dtype == np.float32:
        uint, width = np.uint32, 32
    elif a.dtype == np.float64:
        uint, width = np.uint64, 64
    else:
        return _CLASS_A_MAX_ULP + 1

    signmask = 1 << (width - 1)
    magmask = signmask - 1

    def _key(arr):
        # Promote raw bits to Python ints (object dtype) => overflow-proof.
        u = np.ascontiguousarray(arr).view(uint).astype(object)
        mag = u & magmask
        neg = (u & signmask) != 0
        # Sign-magnitude fold: negatives descend below signmask, non-negatives
        # ascend above it; both zeros collapse to exactly signmask.
        return np.where(neg, signmask - mag, signmask + mag)

    d = np.abs(_key(a) - _key(b))
    return int(np.max(d)) if d.size else 0


def compare_to_oracle(materialized_root: str, oracle_root: str, min_cases: int = 1) -> None:
    """Byte-compare the materialized node tree against the on-disk oracle.

    Pass criterion:
      * dir-set equality (both directions),
      * per-case byte-identity of model.onnx + input_*.pb (hard),
      * per-case byte-identity of output_*.pb, OR <=4-ULP float drift (Class A warn),
      * >= ``min_cases`` cases present.

    Raises ``MaterializeError`` with an actionable, classified message on failure.
    """
    mat_node = _resolve_node_dir(materialized_root)
    ora_node = _resolve_node_dir(oracle_root)

    mat_cases = _list_case_dirs(mat_node)
    ora_cases = _list_case_dirs(ora_node)

    if len(mat_cases) < min_cases:
        raise MaterializeError(f"materialized corpus has {len(mat_cases)} cases (< floor {min_cases})")

    only_oracle = sorted(ora_cases - mat_cases)
    only_mat = sorted(mat_cases - ora_cases)
    if only_oracle or only_mat:
        raise MaterializeError(
            "dir-set mismatch between materialized tree and oracle:\n"
            f"  {len(only_oracle)} oracle-only (missing from materialization): "
            f"{only_oracle[:10]}{' ...' if len(only_oracle) > 10 else ''}\n"
            f"  {len(only_mat)} materialized-only (not in oracle): "
            f"{only_mat[:10]}{' ...' if len(only_mat) > 10 else ''}"
        )

    class_b: list[str] = []
    class_a: list[str] = []
    class_img: list[str] = []
    for case in sorted(mat_cases):
        mat_dir = os.path.join(mat_node, case)
        ora_dir = os.path.join(ora_node, case)
        mat_files = set(_iter_files(mat_dir))
        ora_files = set(_iter_files(ora_dir))
        if mat_files != ora_files:
            class_b.append(
                f"{case}: file-set differs (+{sorted(mat_files - ora_files)} / -{sorted(ora_files - mat_files)})"
            )
            continue
        for rel in sorted(mat_files):
            mp = os.path.join(mat_dir, rel)
            op = os.path.join(ora_dir, rel)
            with open(mp, "rb") as f:
                mb = f.read()
            with open(op, "rb") as f:
                ob = f.read()
            if mb == ob:
                continue
            base = os.path.basename(rel)
            is_output_tensor = base.startswith("output_") and base.endswith(".pb")
            if is_output_tensor:
                cls, msg = _classify_tensor_diff(mp, op)
                if cls == "A":
                    class_a.append(f"{case}/{rel}: {msg}")
                    continue
                class_b.append(f"{case}/{rel}: {msg}")
            elif base == "model.onnx":
                class_b.append(
                    f"{case}/{rel}: model.onnx bytes differ -- "
                    f"{_classify_model_diff(mp, op)} (Class B: onnx-version skew)"
                )
            else:
                # input_*.pb or any other artifact: strict byte-identity required,
                # EXCEPT encoded-image inputs of the image_decoder family, whose bytes
                # are codec/env-dependent and globally excluded downstream (see
                # _IMAGE_DECODER_CASE_PREFIX). Those are an accepted, non-load-bearing
                # class; every other input must still match byte-for-byte.
                is_input_tensor = base.startswith("input_") and base.endswith(".pb")
                if is_input_tensor and case.startswith(_IMAGE_DECODER_CASE_PREFIX):
                    class_img.append(f"{case}/{rel}: encoded-image input bytes differ (codec/env-dependent)")
                    continue
                class_b.append(f"{case}/{rel}: input/artifact bytes differ (Class B)")

    if class_a:
        # Non-fatal: log to stderr so a maintainer sees numpy-gen skew explicitly.
        print(
            f"[materialize_onnx_node_tests] WARNING: {len(class_a)} case(s) differ by "
            f"<= {_CLASS_A_MAX_ULP} float ULP (Class A numpy-gen skew, accepted):",
            file=sys.stderr,
        )
        for line in class_a[:20]:
            print(f"    {line}", file=sys.stderr)

    if class_img:
        # Non-fatal: encoded-image inputs are codec/env-dependent and globally
        # excluded downstream, so byte-divergence from the historical oracle is
        # expected and non-load-bearing (see _IMAGE_DECODER_CASE_PREFIX).
        print(
            f"[materialize_onnx_node_tests] WARNING: {len(class_img)} image_decoder input "
            f"artifact(s) differ from the oracle (codec/env-dependent encoded bytes, accepted):",
            file=sys.stderr,
        )
        for line in class_img[:20]:
            print(f"    {line}", file=sys.stderr)

    if class_b:
        detail = "\n".join(f"    {line}" for line in class_b[:30])
        more = "" if len(class_b) <= 30 else f"\n    ... and {len(class_b) - 30} more"
        raise MaterializeError(
            f"EQUIVALENCE FAILED: {len(class_b)} Class B (structural) mismatch(es) between "
            f"the materialized tree and the on-disk oracle. Class B means a real corpus "
            f"drift -- most likely an onnx-version skew (installed wheel != cmake/deps.txt "
            f"pin) baking a different opset/IR into model.onnx. Fix the onnx pin parity.\n"
            f"{detail}{more}"
        )


def _classify_model_diff(mat_path: str, ora_path: str) -> str:
    try:
        import onnx  # noqa: PLC0415

        m = onnx.load(mat_path)
        o = onnx.load(ora_path)
        mo = {i.domain or "ai.onnx": i.version for i in m.opset_import}
        oo = {i.domain or "ai.onnx": i.version for i in o.opset_import}
        return (
            f"opset {mo} vs {oo}; IR {m.ir_version} vs {o.ir_version}; nodes {len(m.graph.node)} vs {len(o.graph.node)}"
        )
    except Exception as exc:
        return f"(unparseable: {exc})"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize ONNX node-test artifacts from ONNX Python generators.",
    )
    p.add_argument(
        "--out",
        "--output",
        dest="out",
        required=True,
        help="Output root; writes <out>/node/<case>/... mirroring ONNX's layout.",
    )
    p.add_argument(
        "--expected-onnx-version",
        required=True,
        help="Assert onnx.__version__ equals this (the cmake/deps.txt pin). MANDATORY.",
    )
    p.add_argument(
        "--expected-numpy-version",
        default=None,
        help="Informational/forensic only: logs a WARNING (never fails) if "
        "numpy.__version__ differs. numpy is not a hard gate -- ONNX controls the "
        "corpus-gen numpy and the Class-A ULP band absorbs small drift.",
    )
    p.add_argument(
        "--min-cases",
        type=int,
        default=1,
        help="Fail loud if fewer than this many node cases are collected/compared.",
    )
    p.add_argument(
        "--oracle-dir",
        default=None,
        help="DEV-CONVENIENCE alias: if set, materialize then byte-compare against "
        "this on-disk oracle in one command. The build wires the CANONICAL check via "
        "compare_node_test_corpora.py (compares the existing artifact, no regeneration); "
        "this flag shares the same compare_to_oracle core for quick local use.",
    )
    p.add_argument(
        "--stamp",
        default=None,
        help="Touch this file LAST, only after a full successful write, so a partial "
        "tree never satisfies a build stamp.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.expected_numpy_version:
        print(
            "[materialize_onnx_node_tests] WARNING: --expected-numpy-version not given; "
            "recomputed float outputs are numpy-sensitive and reproducibility is not guarded.",
            file=sys.stderr,
        )
    # Unlink any pre-existing stamp FIRST, before regeneration. If generation
    # crashes midway, a stale stamp must not survive to mask a partial/absent tree
    # as an up-to-date build output; the stamp is re-created LAST only on success.
    if args.stamp and os.path.exists(args.stamp):
        os.unlink(args.stamp)
    try:
        written = materialize(
            args.out,
            expected_onnx=args.expected_onnx_version,
            expected_numpy=args.expected_numpy_version,
            min_cases=args.min_cases,
        )
        print(f"[materialize_onnx_node_tests] wrote {len(written)} node cases to {os.path.join(args.out, 'node')}")
        if args.oracle_dir:
            compare_to_oracle(args.out, args.oracle_dir, min_cases=args.min_cases)
            print(
                f"[materialize_onnx_node_tests] EQUIVALENCE OK vs oracle {args.oracle_dir} "
                f"({len(written)} cases byte-identical modulo <= {_CLASS_A_MAX_ULP}-ULP float drift)"
            )
    except MaterializeError as exc:
        print(f"[materialize_onnx_node_tests] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.stamp:
        # Touch LAST: a partial/aborted tree must never satisfy the build stamp.
        os.makedirs(os.path.dirname(os.path.abspath(args.stamp)), exist_ok=True)
        with open(args.stamp, "w") as f:
            f.write(f"{len(written)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
