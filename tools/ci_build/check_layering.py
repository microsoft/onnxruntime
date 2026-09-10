#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Ratchet on cross-layer ``#include`` edges in ``onnxruntime/core``.

The core libraries are a layered stack with a documented, reversed-topological
link order in ``cmake/onnxruntime.cmake`` ("Earlier entries may depend on later
ones. Later ones should not depend on earlier ones."):

    session > providers / optimizer > framework > graph > util > common

That rule is only enforced by the static-library link graph today; nothing stops
a source file in a *lower* layer from ``#include``-ing a *higher* layer's internal
header, which is how layering erodes. This script flags every such upward include
and fails when a new one appears that is not in the frozen baseline, so the set of
violations can only shrink over time.

Public API headers are **not** violations: anything that resolves under the public
``include/onnxruntime/`` tree (the C API surface, config-key headers, ``environment.h``,
...) is header-only contract and is always allowed, even when it physically lives
under ``core/session/``.

Usage::

    python tools/ci_build/check_layering.py              # check against baseline
    python tools/ci_build/check_layering.py --list       # list current upward includes
    python tools/ci_build/check_layering.py --update-baseline   # print a fresh _BASELINE

Exit code is non-zero when a new upward include appears (CI gate), or when a
baselined entry disappears (a reminder to tighten the baseline).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Reversed-topological rank: a lower number is a lower layer. A file may include
# headers from its own or any lower-ranked layer, never a higher-ranked one.
_LAYER_RANKS = {
    "common": 0,
    "platform": 0,
    "mlas": 0,
    "flatbuffers": 0,
    "util": 1,
    "graph": 2,
    "framework": 3,
    "optimizer": 4,
    "providers": 4,
    "session": 5,
}

# Only the foundational tiers are policed as *sources*. The top tiers
# (optimizer/providers/session) legitimately reach into the session ABI surface
# (e.g. every EP factory includes core/session/ort_apis.h), so flagging them as
# sources would penalize accepted glue. This mirrors the audit's intent, whose
# grep only scanned graph + framework.
_MAX_SOURCE_RANK = _LAYER_RANKS["framework"]

_SOURCE_SUFFIXES = (".h", ".hpp", ".cc", ".cpp", ".cxx", ".cu", ".cuh")

# Matches  #include "core/<layer>/<rest>"  (quoted includes only).
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"(core/([A-Za-z0-9_]+)/[^"]+)"')

# Frozen set of (source_file, included_header) upward edges that exist today and
# are accepted (legacy / intentional ABI glue). Regenerate with --update-baseline.
# This set must only ever shrink.
_BASELINE: frozenset[tuple[str, str]] = frozenset(
    {
        ("include/onnxruntime/core/framework/allocator.h", "core/session/abi_key_value_pairs.h"),
        ("include/onnxruntime/core/graph/graph.h", "core/framework/prepacked_weights_container.h"),
        ("include/onnxruntime/core/graph/graph_viewer.h", "core/framework/session_options.h"),
        ("onnxruntime/core/framework/allocator.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/compute_capability.h", "core/optimizer/graph_optimizer_registry.h"),
        ("onnxruntime/core/framework/error_code.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/graph_partitioner.h", "core/optimizer/graph_optimizer_registry.h"),
        ("onnxruntime/core/framework/layering_annotations.cc", "core/session/abi_devices.h"),
        ("onnxruntime/core/framework/onnxruntime_map_type_info.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/onnxruntime_optional_type_info.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/onnxruntime_sequence_type_info.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/onnxruntime_typeinfo.cc", "core/session/model_editor_api.h"),
        ("onnxruntime/core/framework/onnxruntime_typeinfo.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/plugin_data_transfer.h", "core/session/abi_devices.h"),
        ("onnxruntime/core/framework/plugin_ep_stream.cc", "core/session/abi_logger.h"),
        ("onnxruntime/core/framework/plugin_ep_stream.h", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/run_options.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/sequential_executor.cc", "core/providers/cuda/nvtx_profile.h"),
        ("onnxruntime/core/framework/sequential_executor.cc", "core/providers/cuda/nvtx_profile_context.h"),
        ("onnxruntime/core/framework/session_state.cc", "core/providers/cpu/controlflow/utils.h"),
        ("onnxruntime/core/framework/tensor_type_and_shape.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/tensorprotoutils.cc", "core/session/ort_apis.h"),
        ("onnxruntime/core/framework/transpose_helper.cc", "core/providers/cpu/tensor/utils.h"),
        ("onnxruntime/core/graph/abi_graph_types.h", "core/framework/onnxruntime_typeinfo.h"),
        ("onnxruntime/core/graph/abi_graph_types.h", "core/framework/tensor_external_data_info.h"),
        ("onnxruntime/core/graph/abi_graph_types.h", "core/session/inference_session.h"),
        ("onnxruntime/core/graph/contrib_ops/nchwc_schema_defs.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/contrib_ops/range_schema_defs.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/data_propagation/add_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/data_propagation/div_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/data_propagation/gather_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/data_propagation/mul_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/data_propagation/squeeze_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/data_propagation/sub_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/data_propagation/unsqueeze_op_data_propagation.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/dml_ops/dml_defs.cc", "core/providers/dml/OperatorAuthorHelper/Attributes.h"),
        ("onnxruntime/core/graph/ep_api_types.cc", "core/framework/onnxruntime_typeinfo.h"),
        ("onnxruntime/core/graph/ep_api_types.cc", "core/framework/tensor_external_data_info.h"),
        ("onnxruntime/core/graph/ep_api_types.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/function.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/function_utils.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/graph.cc", "core/framework/error_code_helper.h"),
        ("onnxruntime/core/graph/graph.cc", "core/framework/tensor_external_data_info.h"),
        ("onnxruntime/core/graph/graph.cc", "core/framework/tensor_type_and_shape.h"),
        ("onnxruntime/core/graph/graph.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/graph.cc", "core/framework/utils.h"),
        ("onnxruntime/core/graph/graph.cc", "core/providers/common.h"),
        ("onnxruntime/core/graph/graph_flatbuffers_utils.cc", "core/framework/tensor_external_data_info.h"),
        ("onnxruntime/core/graph/graph_flatbuffers_utils.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/graph_proto_serializer.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/graph_utils.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/model.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/graph/model.h", "core/framework/session_options.h"),
        ("onnxruntime/core/graph/model_editor_api_types.h", "core/session/inference_session.h"),
        ("onnxruntime/core/graph/model_editor_api_types.h", "core/session/ort_apis.h"),
        ("onnxruntime/core/graph/node_attr_utils.cc", "core/framework/tensorprotoutils.h"),
        ("onnxruntime/core/platform/device_discovery.h", "core/session/abi_devices.h"),
        ("onnxruntime/core/platform/linux/npu_device_discovery.h", "core/session/abi_devices.h"),
        ("onnxruntime/core/platform/linux/pci_device_discovery.h", "core/session/abi_devices.h"),
        ("onnxruntime/core/platform/windows/device_discovery.cc", "core/session/abi_devices.h"),
        ("onnxruntime/core/util/qmath.h", "core/framework/element_type_lists.h"),
        ("onnxruntime/core/util/thread_utils.cc", "core/session/ort_apis.h"),
    }
)


def _layer_of_path(path: Path, repo_root: Path) -> str | None:
    """Return the layer name a source file belongs to, or None if unclassified."""
    parts = path.relative_to(repo_root).parts
    for i, part in enumerate(parts[:-1]):
        if part == "core":
            candidate = parts[i + 1]
            if candidate in _LAYER_RANKS:
                return candidate
    return None


def _is_public_api(target: str, repo_root: Path) -> bool:
    """True if the included header resolves under the public include/ tree."""
    return (repo_root / "include" / "onnxruntime" / target).is_file()


def find_upward_includes(repo_root: Path) -> set[tuple[str, str]]:
    """Return (source_rel_posix, included_target) for every upward cross-layer include."""
    scan_roots = [
        repo_root / "onnxruntime" / "core",
        repo_root / "include" / "onnxruntime" / "core",
    ]
    hits: set[tuple[str, str]] = set()
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for path in scan_root.rglob("*"):
            if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
                continue
            source_layer = _layer_of_path(path, repo_root)
            if source_layer is None:
                continue
            source_rank = _LAYER_RANKS[source_layer]
            if source_rank > _MAX_SOURCE_RANK:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:  # pragma: no cover - defensive I/O guard
                print(f"warning: could not read {path}: {exc}", file=sys.stderr)
                continue
            for line in text.splitlines():
                match = _INCLUDE_RE.match(line)
                if not match:
                    continue
                target, target_layer = match.group(1), match.group(2)
                if target_layer not in _LAYER_RANKS:
                    continue
                if _LAYER_RANKS[target_layer] <= source_rank:
                    continue
                if _is_public_api(target, repo_root):
                    continue
                hits.add((path.relative_to(repo_root).as_posix(), target))
    return hits


def _print_baseline(hits: set[tuple[str, str]]) -> None:
    print("_BASELINE: frozenset[tuple[str, str]] = frozenset(")
    print("    {")
    for source, target in sorted(hits):
        print(f"        ({source!r}, {target!r}),")
    print("    }")
    print(")")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Print every current upward include.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Print a fresh _BASELINE literal for the current tree and exit 0.",
    )
    args = parser.parse_args()

    hits = find_upward_includes(repo_root)

    if args.list:
        for source, target in sorted(hits):
            print(f"{source}: {target}")

    if args.update_baseline:
        _print_baseline(hits)
        return 0

    new_violations = sorted(hits - _BASELINE)
    resolved = sorted(_BASELINE - hits)

    print(f"Cross-layer upward includes: {len(hits)} (baseline {len(_BASELINE)}).")

    if new_violations:
        print(
            f"error: {len(new_violations)} new cross-layer include(s) introduced:",
            file=sys.stderr,
        )
        for source, target in new_violations:
            print(f"  {source} -> {target}", file=sys.stderr)
        print(
            "A lower layer must not include a higher layer's internal header. "
            "Route through an existing abstraction (e.g. ProviderHost / a public API header) instead.",
            file=sys.stderr,
        )
        return 1

    if resolved:
        print(
            f"note: {len(resolved)} baselined include(s) are gone; remove them from _BASELINE "
            "in tools/ci_build/check_layering.py to keep the ratchet tight:",
            file=sys.stderr,
        )
        for source, target in resolved:
            print(f"  {source} -> {target}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
