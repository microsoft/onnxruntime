#!/usr/bin/env python3
"""Compile an ONNX model for a Windows ML NPU execution provider."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import onnx

from run_winml_ep import (
    PROVIDER_NAMES,
    find_npu_device,
    parse_provider_options,
    register_provider,
)


COMPILE_PROVIDERS = ("qnn", "openvino")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Input ONNX model.")
    parser.add_argument(
        "--provider",
        choices=COMPILE_PROVIDERS,
        required=True,
        help="Windows ML NPU provider used to compile the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Compiled EPContext ONNX path; defaults beside the input model.",
    )
    parser.add_argument(
        "--embed-context",
        action="store_true",
        help="Embed compiled context data in the output ONNX model.",
    )
    parser.add_argument(
        "--require-full-npu",
        action="store_true",
        help="Fail if compilation requires CPU fallback.",
    )
    parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Provider-specific compilation option; may be supplied repeatedly.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise ValueError(f"input model does not exist: {args.model}")
    if args.output is not None and args.output.exists():
        raise ValueError(f"output model already exists: {args.output}")


def printable_attribute(attribute: onnx.AttributeProto) -> str:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<{len(value)} binary bytes>"
    return str(value)


def print_compiled_model_summary(model_path: Path, provider_name: str) -> None:
    model = onnx.load(model_path, load_external_data=False)
    ep_context_nodes = [
        node for node in model.graph.node if node.op_type == "EPContext"
    ]
    print(f"Compiled model:  {model_path.resolve()}")
    print(f"Provider:        {provider_name}")
    print(f"EPContext nodes: {len(ep_context_nodes)}")
    for index, node in enumerate(ep_context_nodes):
        attributes = {
            attribute.name: printable_attribute(attribute)
            for attribute in node.attribute
            if attribute.name != "ep_cache_context"
        }
        print(
            f"  [{index}] {node.name or '<unnamed>'} "
            f"(domain={node.domain or '<default>'}): {attributes}"
        )


def compile_model(
    ort: Any,
    input_model: Path,
    output_model: Path,
    provider_name: str,
    provider_options: dict[str, str],
    embed_context: bool,
    require_full_npu: bool,
) -> None:
    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices(
        [find_npu_device(ort, provider_name)],
        provider_options,
    )
    if require_full_npu:
        session_options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", "1"
        )

    compiler = ort.ModelCompiler(
        session_options,
        input_model,
        embed_compiled_data_into_model=embed_context,
        flags=ort.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    )
    compiler.compile_to_file(output_model)


def main() -> None:
    args = parse_args()
    validate_args(args)

    provider_name = PROVIDER_NAMES[args.provider]
    provider_options = parse_provider_options(args.provider_option)
    input_model = args.model.resolve()
    output_model = (
        args.output
        if args.output is not None
        else input_model.with_name(f"{input_model.stem}_{args.provider}_ctx.onnx")
    ).resolve()
    if output_model.exists():
        raise ValueError(f"output model already exists: {output_model}")
    output_model.parent.mkdir(parents=True, exist_ok=True)

    ort = register_provider(provider_name)
    try:
        compile_model(
            ort,
            input_model,
            output_model,
            provider_name,
            provider_options,
            args.embed_context,
            args.require_full_npu,
        )
    finally:
        ort.unregister_execution_provider_library(provider_name)

    print_compiled_model_summary(output_model, provider_name)


if __name__ == "__main__":
    main()
