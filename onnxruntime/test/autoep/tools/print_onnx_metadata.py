#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import onnx


def model_metadata_to_dict(model: onnx.ModelProto) -> dict[str, Any]:
    custom_metadata = {entry.key: entry.value for entry in model.metadata_props}

    graph = model.graph
    info = {
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": model.model_version,
        "doc_string": model.doc_string,
        "graph_name": graph.name,
        "opset_imports": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "counts": {
            "inputs": len(graph.input),
            "outputs": len(graph.output),
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
        },
        "custom_metadata": custom_metadata,
    }

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Print metadata from an ONNX model file.")
    parser.add_argument("model", type=Path, help="Path to .onnx file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = onnx.load(str(args.model))
    info = model_metadata_to_dict(model)

    if args.pretty:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
