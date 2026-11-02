# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Prepare a separate WebGPU decoder copy, removing provably empty LoRA branches.

Requires onnx. Weights remain byte-for-byte unchanged. This is an offline model
preparation tool, not an ORT core graph transformer. Compare logits/KV and measure
the whole workload before using its output with a different model or backend.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import onnx


def prune_graph(graph):
    """Keep reachable nodes/initializers and put nodes in dependency order."""
    producers = {value: index for index, node in enumerate(graph.node) for value in node.output if value}
    needed = set()
    visited = set()
    active = set()
    ordered = []

    def visit(value):
        needed.add(value)
        index = producers.get(value)
        if index is None or index in visited:
            return
        if index in active:
            raise ValueError("The decoder graph contains a cycle.")
        active.add(index)
        node = graph.node[index]
        for name in node.input:
            if name:
                visit(name)
        active.remove(index)
        visited.add(index)
        ordered.append(node)

    for output in graph.output:
        visit(output.name)
    del graph.node[:]
    graph.node.extend(ordered)
    initializers = [value for value in graph.initializer if value.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    values = [value for value in graph.value_info if value.name in needed]
    del graph.value_info[:]
    graph.value_info.extend(values)


def optimize(model, head_size=None, freeze_initializers=False):
    """Remove Add(base, MatMul(MatMul(X, [K,0]), [0,N])) with matching X/K/N."""
    graph = model.graph
    if graph.sparse_initializer or any(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in graph.node
        for attr in node.attribute
    ):
        raise ValueError("Only dense decoder graphs without subgraphs are supported.")
    initializers = {value.name: value for value in graph.initializer}
    frozen = 0
    if freeze_initializers:
        inputs = [value for value in graph.input if value.name not in initializers]
        frozen = len(graph.input) - len(inputs)
        del graph.input[:]
        graph.input.extend(inputs)
    overridable = {value.name for value in graph.input}
    producers = {value: node for node in graph.node for value in node.output if value}

    def weight_shape(value):
        seen = set()
        while value not in seen:
            seen.add(value)
            if value in overridable:
                return None
            if value in initializers:
                return tuple(initializers[value].dims)
            node = producers.get(value)
            if node is None or node.domain or node.op_type not in ("Cast", "DequantizeLinear"):
                return None
            value = node.input[0]
        return None

    aliases = {}
    for node in graph.node:
        if node.domain or node.op_type != "Add" or len(node.input) != 2:
            continue
        for index in (0, 1):
            branch = producers.get(node.input[index])
            base = producers.get(node.input[1 - index])
            if (
                branch is None
                or branch.domain
                or branch.op_type != "MatMul"
                or base is None
                or base.domain != "com.microsoft"
                or base.op_type != "MatMulNBits"
            ):
                continue
            first = producers.get(branch.input[0])
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in base.attribute}
            if (
                first is not None
                and not first.domain
                and first.op_type == "MatMul"
                and first.input[0] == base.input[0]
                and attrs.get("K", 0) > 0
                and attrs.get("N", 0) > 0
                and weight_shape(first.input[1]) == (attrs["K"], 0)
                and weight_shape(branch.input[1]) == (0, attrs["N"])
            ):
                aliases[node.output[0]] = base.output[0]

    for node in graph.node:
        for index, value in enumerate(node.input):
            node.input[index] = aliases.get(value, value)
    # Preserve public output names even when the Add itself is an output.
    for output in graph.output:
        if output.name in aliases:
            old = producers[output.name]
            old.CopyFrom(onnx.helper.make_node("Identity", [aliases[output.name]], [output.name], name=old.name))
    prune_graph(graph)
    specialized = 0
    if head_size is not None:
        if head_size <= 0:
            raise ValueError("head_size must be positive.")
        for value in [*graph.input, *graph.output, *graph.value_info]:
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param == "kv_cache_head_size":
                    dim.dim_value = head_size
                    specialized += 1
    return {
        "empty_lora_branches_removed": len(aliases),
        "kv_head_dimensions_specialized": specialized,
        "initializer_inputs_frozen": frozen,
    }


def prepare(source, destination, specialize_head=False, freeze_initializers=False):
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Output directory must not exist; the source model is never overwritten.")
    config = json.loads((source / "genai_config.json").read_text(encoding="utf-8"))
    decoder = config["model"]["decoder"]
    model_path = source / decoder["filename"]
    if model_path.parent != source:
        raise ValueError("The decoder filename must be directly inside the model directory.")
    model = onnx.load(model_path, load_external_data=False)
    before = Counter(node.op_type for node in model.graph.node)
    result = optimize(model, decoder["head_size"] if specialize_head else None, freeze_initializers)
    # Copy external files instead of sharing writable model weights or modifying their references.
    locations = {
        item.value for tensor in model.graph.initializer for item in tensor.external_data if item.key == "location"
    }
    for location in locations:
        if Path(location).is_absolute() or ".." in Path(location).parts:
            raise ValueError("External weight locations must stay inside the model directory.")
    destination.mkdir(parents=True)
    for name in sorted(locations | {"genai_config.json", "tokenizer.json", "tokenizer_config.json"}):
        src = source / name
        if src.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        elif name in locations:
            raise FileNotFoundError(src)
    onnx.save(model, destination / decoder["filename"])
    result.update(
        source_model=str(model_path),
        source_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        model=str(destination / decoder["filename"]),
        model_sha256=hashlib.sha256((destination / decoder["filename"]).read_bytes()).hexdigest(),
        nodes_before=dict(before),
        nodes_after=dict(Counter(node.op_type for node in model.graph.node)),
        note="Empty LoRA removal can change signed zero. Validate numerical outputs; weights are unchanged.",
        validation="Load with ORT to validate its experimental/contrib schemas before benchmarking.",
    )
    (destination / "optimization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--freeze-initializers",
        action="store_true",
        help="Make initializer inputs constant; prepared models cannot override those weights/adapters.",
    )
    parser.add_argument(
        "--specialize-head",
        action="store_true",
        help="Opt into config head-width specialization and additional FP16 fusions; revalidate numerics.",
    )
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.model_dir, args.output_dir, args.specialize_head, args.freeze_initializers), indent=2)
    )


if __name__ == "__main__":
    main()
