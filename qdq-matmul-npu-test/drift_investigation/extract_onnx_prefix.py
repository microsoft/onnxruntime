#!/usr/bin/env python3
"""Extract the ancestors of one ONNX value to minimize an EP mismatch."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnx_ir as ir

from split_gemma_vision_pooler import (
    ancestor_nodes,
    find_value,
    infer_qdq_value_metadata,
    prune_component,
    required_graph_inputs,
    save_component,
)


def make_prefix(model: ir.Model, value_name: str, output_name: str = "probe_output") -> ir.Model:
    graph = model.graph
    boundary = find_value(graph, value_name)
    infer_qdq_value_metadata(boundary)
    selected = ancestor_nodes([boundary], set())
    indices = {index for index, node in enumerate(graph) if node in selected}
    inputs = required_graph_inputs(selected, graph.inputs)
    if not indices or not inputs:
        raise ValueError(f"no runnable ancestors and graph inputs for {value_name!r}")
    return prune_component(
        model, indices, inputs, [value_name], value_name, output_name, "diagnostic_prefix"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Source ONNX model.")
    parser.add_argument("--value", required=True, help="Graph value at which to cut the model.")
    parser.add_argument("--output-name", default="probe_output", help="Output name for the cut value.")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX model.")
    parser.add_argument(
        "--self-contained", action="store_true",
        help="Write only the prefix's used external weights beside the output, instead of reusing source weights.",
    )
    args = parser.parse_args()
    source = args.model.resolve()
    output = args.output.resolve()
    if not source.is_file() or output.exists() or source == output:
        raise ValueError("source must exist and output must be a new model")
    if not args.self_contained and output.parent != source.parent:
        raise ValueError("without --self-contained, output must be beside the source's external data")
    prefix = make_prefix(ir.load(source), args.value, args.output_name)
    if args.self_contained:
        save_component(prefix, output, False)
    else:
        onnx.save(ir.serde.serialize_model(prefix), output)
    print(
        f"Saved {output}: {prefix.graph.num_nodes()} nodes, "
        f"inputs={[value.name for value in prefix.graph.inputs]}, "
        f"output={[value.name for value in prefix.graph.outputs]}"
    )


if __name__ == "__main__":
    main()
