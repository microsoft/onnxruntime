#!/usr/bin/env python3
"""Expose selected ONNX graph values while preserving model inputs and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnx_ir as ir

from split_gemma_vision_pooler import find_value, infer_qdq_value_metadata


def add_taps(model: ir.Model, names: list[str]) -> None:
    graph = model.graph
    if len(names) != len(set(names)):
        raise ValueError("tap names must be unique")
    existing_outputs = {value.name for value in graph.outputs}
    for name in names:
        if name in existing_outputs:
            raise ValueError(f"{name!r} is already a graph output")
        value = find_value(graph, name)
        if value.shape is None or value.type is None:
            infer_qdq_value_metadata(value)
        if value.shape is None or value.type is None:
            raise ValueError(f"tap {name!r} has no static shape or data type")
        graph.outputs.append(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Source model with its external data beside it.")
    parser.add_argument("--tap", action="append", required=True, help="Graph value name to expose (repeatable).")
    parser.add_argument("--output", type=Path, required=True, help="New model beside the source.")
    args = parser.parse_args()
    source = args.model.resolve()
    output = args.output.resolve()
    if not source.is_file() or source == output or output.exists() or output.parent != source.parent:
        raise ValueError("source must exist; output must be a new model beside it")
    model = ir.load(source)
    add_taps(model, args.tap)
    onnx.save(ir.serde.serialize_model(model), output)
    print(f"Exposed {len(args.tap)} graph values: {output}")


if __name__ == "__main__":
    main()
