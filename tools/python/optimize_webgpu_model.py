#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Save an ONNX model with ONNX Runtime's WebGPU graph fusions applied."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import onnx


def _default_perf_test() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "build" / "WGPU" / "RelWithDebInfo" / "RelWithDebInfo" / "onnxruntime_perf_test.exe"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hardware-specific ONNX model containing the graph fusions selected by "
            "ONNX Runtime's WebGPU execution provider."
        )
    )
    parser.add_argument("input_model", type=Path, help="Input ONNX model.")
    parser.add_argument("output_model", type=Path, help="Output fused ONNX model.")
    parser.add_argument(
        "--perf-test",
        type=Path,
        default=_default_perf_test(),
        help="Path to the WebGPU-enabled onnxruntime_perf_test executable.",
    )
    parser.add_argument(
        "--external-data-name",
        default=None,
        help="External initializer filename. Defaults to <output filename>.data.",
    )
    parser.add_argument(
        "--external-data-min-size",
        type=int,
        default=1024,
        help=(
            "Minimum initializer size, in bytes, to place in external data. "
            "Small shape constants must remain inline. Default: 1024."
        ),
    )
    parser.add_argument(
        "--direct-storage",
        action="store_true",
        help="Use DirectStorage while verifying the saved model.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not load the saved model with runtime graph optimization disabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output model and external-data files.",
    )
    return parser.parse_args()


def _run(command: list[str], description: str) -> None:
    print(f"{description}:\n  {subprocess.list2cmdline(command)}", flush=True)
    result = subprocess.run(command, text=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}.")


def _external_data_value(initializer: onnx.TensorProto, key: str) -> str | None:
    return next((entry.value for entry in initializer.external_data if entry.key == key), None)


def _restore_shared_gather_weights(
    model: onnx.ModelProto, external_data_location: str
) -> tuple[int, int]:
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    restored_count = 0

    for folded_initializer in list(graph.initializer):
        suffix = "_reshaped_for_gbq"
        if not folded_initializer.name.endswith(suffix):
            continue

        source_name = folded_initializer.name[: -len(suffix)]
        source_initializer = initializers.get(source_name)
        if source_initializer is None:
            continue

        consumers = [node for node in graph.node if folded_initializer.name in node.input]
        if not consumers or any(node.op_type != "GatherBlockQuantized" for node in consumers):
            continue
        if source_initializer.data_type != folded_initializer.data_type:
            raise RuntimeError(f"Cannot share differently typed initializer {folded_initializer.name!r}.")

        if math.prod(source_initializer.dims) != math.prod(folded_initializer.dims):
            raise RuntimeError(f"Cannot share differently sized initializer {folded_initializer.name!r}.")

        shape_name = f"{folded_initializer.name}_shape"
        if shape_name in initializers:
            raise RuntimeError(f"Generated shape initializer already exists: {shape_name!r}.")
        shape_initializer = onnx.helper.make_tensor(
            shape_name,
            onnx.TensorProto.INT64,
            [len(folded_initializer.dims)],
            folded_initializer.dims,
        )
        reshape_node = onnx.helper.make_node(
            "Reshape",
            [source_name, shape_name],
            [folded_initializer.name],
            name=f"RestoreShared_{folded_initializer.name}",
        )
        first_consumer_index = min(
            index for index, node in enumerate(graph.node) if node in consumers
        )
        graph.node.insert(first_consumer_index, reshape_node)
        graph.initializer.append(shape_initializer)
        graph.initializer.remove(folded_initializer)
        restored_count += 1

    max_referenced_end = 0
    if restored_count:
        for initializer in graph.initializer:
            location = _external_data_value(initializer, "location")
            if location != external_data_location:
                continue
            offset = int(_external_data_value(initializer, "offset") or 0)
            length = int(_external_data_value(initializer, "length") or 0)
            max_referenced_end = max(max_referenced_end, offset + length)
        if max_referenced_end == 0:
            raise RuntimeError(f"No initializers reference external data file {external_data_location!r}.")

    return restored_count, max_referenced_end


def _postprocess_saved_model(model_path: Path, external_data_path: Path) -> tuple[int, int]:
    model = onnx.load_model(model_path, load_external_data=False)
    graph = model.graph
    memcpy_nodes = [
        node for node in graph.node if node.op_type in {"MemcpyFromHost", "MemcpyToHost"} and not node.domain
    ]

    for memcpy_node in memcpy_nodes:
        if len(memcpy_node.input) != 1 or len(memcpy_node.output) != 1:
            raise RuntimeError(f"Unexpected {memcpy_node.op_type} input/output count.")

        source_name = memcpy_node.input[0]
        destination_name = memcpy_node.output[0]
        if any(output.name == destination_name for output in graph.output):
            raise RuntimeError(
                f"Cannot remove {memcpy_node.op_type} whose output {destination_name!r} is a model output."
            )

        for node in graph.node:
            if node is memcpy_node:
                continue
            for index, input_name in enumerate(node.input):
                if input_name == destination_name:
                    node.input[index] = source_name

    retained_nodes = [node for node in graph.node if node not in memcpy_nodes]
    del graph.node[:]
    graph.node.extend(retained_nodes)

    removed_value_info_names = {node.output[0] for node in memcpy_nodes}
    retained_value_info = [value_info for value_info in graph.value_info if value_info.name not in removed_value_info_names]
    del graph.value_info[:]
    graph.value_info.extend(retained_value_info)

    external_data_location = external_data_path.relative_to(model_path.parent).as_posix()
    restored_shared_weights, max_referenced_end = _restore_shared_gather_weights(model, external_data_location)
    onnx.save_model(model, model_path)
    if max_referenced_end:
        with external_data_path.open("r+b") as external_data:
            external_data.truncate(max_referenced_end)
    return len(memcpy_nodes), restored_shared_weights


def main() -> int:
    args = _parse_args()
    input_model = args.input_model.resolve(strict=True)
    output_model = args.output_model.resolve()
    perf_test = args.perf_test.resolve(strict=True)

    if input_model == output_model:
        raise ValueError("The output model must be different from the input model.")
    if output_model.suffix.lower() != ".onnx":
        raise ValueError("The output model must have an .onnx extension.")
    if args.external_data_min_size < 1:
        raise ValueError("--external-data-min-size must be at least 1 byte.")

    external_data_name = args.external_data_name or f"{output_model.name}.data"
    external_data_relative_path = Path(external_data_name)
    if external_data_relative_path.is_absolute() or ".." in external_data_relative_path.parts:
        raise ValueError("--external-data-name must be a path relative to the output model.")
    external_data_path = output_model.parent / external_data_relative_path

    existing_outputs = [path for path in (output_model, external_data_path) if path.exists()]
    if existing_outputs and not args.overwrite:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Output already exists: {paths}. Use --overwrite to replace it.")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    external_data_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()

    optimize_command = [
        str(perf_test),
        "-e",
        "webgpu",
        "-o",
        "99",
        "-n",
        "-u",
        str(output_model),
        "--opt_data",
        external_data_relative_path.as_posix(),
        "--opt_weight_min_size",
        str(args.external_data_min_size),
        str(input_model),
    ]
    _run(optimize_command, "Optimizing WebGPU graph")

    if not output_model.is_file() or not external_data_path.is_file():
        raise RuntimeError("ONNX Runtime completed without creating both expected output files.")

    removed_memcpy_nodes, restored_shared_weights = _postprocess_saved_model(output_model, external_data_path)
    print(f"Removed {removed_memcpy_nodes} serialized device-copy nodes; ORT will recreate them after partitioning.")
    print(f"Restored {restored_shared_weights} folded gather weight(s) as zero-copy Reshape views.")

    if not args.skip_verify:
        verify_command = [str(perf_test), "-e", "webgpu"]
        if args.direct_storage:
            verify_command += ["-i", "directStorageExternalWeights|1"]
        verify_command += ["-o", "0", "-n", str(output_model)]
        _run(verify_command, "Verifying fused model with runtime graph optimization disabled")

    print(f"Fused model: {output_model}")
    print(f"External data: {external_data_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
