#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Generate CUDA plugin kernel registration entries from the internal CUDA EP
function table in `cuda_execution_provider.cc`.

The output is a C++ include file with entries in this form:
  {"Add", 7, 12, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

TYPE_TO_ORT_ENUM = {
    "bool": "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL",
    "float": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
    "double": "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
    "double_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
    "MLFloat16": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16",
    "BFloat16": "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16",
    "int8_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8",
    "int16_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16",
    "int32_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32",
    "int64_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64",
    "uint8_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8",
    "uint16_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16",
    "uint32_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32",
    "uint64_t": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64",
    "Float8E4M3FN": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN",
    "Float8E5M2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2",
    "Float8E4M3FNUZ": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ",
    "Float8E5M2FNUZ": "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ",
    "Float4E2M1x2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED",
    "UInt4x2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4",
    "Int4x2": "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4",
}

# Mapping of Op -> (input_mem_type_indices, output_mem_type_indices)
# Only OrtMemTypeCPUInput/CPUOutput are supported for now.
# Indices are based on ORT's internal CUDA EP registration.
# This ensures that when the plugin registers These kernels, it correctly identifies
# which inputs (e.g., shape tensor for Reshape, axes for Concat) must reside in CPU memory.
# Format: { op_type: ( [input_indices], [output_indices] ) }
MEMORY_TYPE_OVERRIDES: dict[str, tuple[list[int], list[int]]] = {
    "Reshape": ([1], []),
    "Slice": ([1, 2, 3, 4], []),
    "Tile": ([1], []),
    "Expand": ([1], []),
    "Pad": ([1, 2], []),
    "Resize": ([1, 2, 3], []),
    "TopK": ([1], []),
    "CumSum": ([1], []),
    "Range": ([0, 1, 2], []),
    "ConstantOfShape": ([0], []),
    "Dropout": ([1, 2], []),
    "NonMaxSuppression": ([2, 3, 4], []),
    "MatMulInteger": ([2, 3], []),
    "OneHot": ([1], []),
    "Squeeze": ([1], []),
    "Unsqueeze": ([1], []),
    "Split": ([1], []),
    "SplitToSequence": ([1], []),
    "ConcatFromSequence": ([1], []),
    "MemcpyFromHost": ([0], []),
    "MemcpyToHost": ([], [0]),
    "IF": ([0], []),
    "Loop": ([0, 1], []),
    "Scan": ([0], []),
}

MACRO_PATTERN = re.compile(r"BuildKernelCreateInfo\s*<\s*([^>]+)\s*>")


@dataclass(frozen=True, order=True)
class Entry:
    op_type: str
    since_version_start: int
    since_version_end: int
    domain: str
    registration_id: int
    constraint_name: str
    type_enum: str
    input_mem_types: list[int] = field(default_factory=list, compare=False)
    output_mem_types: list[int] = field(default_factory=list, compare=False)


@dataclass(frozen=True)
class MacroParseResult:
    macro_name: str
    op_type: str
    since_version_start: int
    since_version_end: int
    domain_token: str
    type_tokens: list[str]


DOMAIN_TOKEN_TO_LITERAL = {
    "kOnnxDomain": "",
    "kMSDomain": "com.microsoft",
}

# Specializations for contrib kernels that encode multiple template types into one token.
# Example: GroupQueryAttention uses type alias "BFloat16_int8_t", which maps to:
#   T=BFloat16, T_CACHE=int8_t.
COMPOSITE_TYPE_CONSTRAINTS: dict[str, list[str]] = {
    "GroupQueryAttention": ["T", "T_CACHE"],
    "MultiHeadAttention": ["T", "QK"],
    "DecoderMaskedMultiHeadAttention": ["T", "QK"],
    "LayerNormalization": ["T", "U", "V"],
    "SimplifiedLayerNormalization": ["T", "U", "V"],
    "QuantizeLinear": ["T2", "T1"],
    "DequantizeLinear": ["T1", "T2"],
}

# Additional fixed constraints for specific ops that are not encoded in the typed class token.
FIXED_CONSTRAINTS: dict[str, list[tuple[str, str]]] = {
    "GroupQueryAttention": [("T_KV_SCALE", "float"), ("M", "int32_t")],
    "GatherBlockQuantized": [("Tind", "int32_t")],
}

CRITICAL_CONTRIB_OP_CONSTRAINTS: dict[str, set[str]] = {
    "GroupQueryAttention": {"T", "T_CACHE", "T_KV_SCALE", "M"},
}

SORTED_TYPE_TOKENS = sorted(TYPE_TO_ORT_ENUM.keys(), key=len, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("onnxruntime/core/providers/cuda/cuda_execution_provider.cc"),
        help="Path to cuda_execution_provider.cc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .inc file path. Defaults depend on --contrib.",
    )
    parser.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="Optional op-type allowlist. Empty means no op filter.",
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Optional type-token allowlist. Empty means no type filter.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Only include registrations with this domain token. Defaults depend on --contrib.",
    )
    parser.add_argument(
        "--contrib",
        action="store_true",
        help="Generate contrib registrations (defaults: domain=kMSDomain, include all ops/types).",
    )
    parser.add_argument(
        "--max-opset",
        type=int,
        default=23,
        help="Upper opset bound used for non-versioned registrations.",
    )
    parser.add_argument(
        "--check-critical-contrib",
        action="store_true",
        help="Validate critical contrib ops emit required type constraints.",
    )
    return parser.parse_args()


def split_args(arg_blob: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in arg_blob:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf.clear()
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def parse_macro(macro_expr: str, max_opset: int) -> MacroParseResult | None:
    m = re.match(r"([A-Z0-9_]+)\s*\((.*)\)$", macro_expr.strip())
    if not m:
        return None

    macro_name = m.group(1)
    args = split_args(m.group(2))

    if macro_name == "ONNX_OPERATOR_KERNEL_CLASS_NAME" and len(args) == 4:
        _, domain, since, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, domain, [])

    if macro_name == "ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME" and len(args) == 5:
        _, domain, start, end, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), domain, [])

    if macro_name == "ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME" and len(args) == 5:
        _, domain, since, t, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, domain, [t])

    if macro_name == "ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME" and len(args) == 6:
        _, domain, start, end, t, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), domain, [t])

    if macro_name == "ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME" and len(args) == 6:
        _, domain, since, t1, t2, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, domain, [t1, t2])

    if macro_name == "ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME" and len(args) == 7:
        _, domain, start, end, t1, t2, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), domain, [t1, t2])

    if macro_name == "ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME" and len(args) == 7:
        _, domain, since, t1, t2, t3, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, domain, [t1, t2, t3])

    if macro_name == "ONNX_OPERATOR_VERSIONED_THREE_TYPED_KERNEL_CLASS_NAME" and len(args) == 8:
        _, domain, start, end, t1, t2, t3, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), domain, [t1, t2, t3])

    # Contrib aliases in onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc
    if macro_name == "CUDA_MS_OP_CLASS_NAME" and len(args) == 2:
        since, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, "kMSDomain", [])

    if macro_name == "CUDA_MS_OP_VERSIONED_CLASS_NAME" and len(args) == 3:
        start, end, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), "kMSDomain", [])

    if macro_name == "CUDA_MS_OP_TYPED_CLASS_NAME" and len(args) == 3:
        since, t, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, "kMSDomain", [t])

    if macro_name == "CUDA_MS_OP_VERSIONED_TYPED_CLASS_NAME" and len(args) == 4:
        start, end, t, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), "kMSDomain", [t])

    if macro_name == "CUDA_MS_OP_TWO_TYPED_CLASS_NAME" and len(args) == 4:
        since, t1, t2, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, "kMSDomain", [t1, t2])

    if macro_name == "CUDA_MS_OP_THREE_TYPED_CLASS_NAME" and len(args) == 5:
        since, t1, t2, t3, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, "kMSDomain", [t1, t2, t3])

    # Legacy contrib operators in ONNX domain
    if macro_name == "CUDA_ONNX_OP_TYPED_CLASS_NAME" and len(args) == 3:
        since, t, op = args
        return MacroParseResult(macro_name, op, int(since), max_opset, "kOnnxDomain", [t])

    if macro_name == "CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME" and len(args) == 4:
        start, end, t, op = args
        return MacroParseResult(macro_name, op, int(start), int(end), "kOnnxDomain", [t])

    return None


def get_constraint_pairs(parsed: MacroParseResult) -> list[tuple[str, str]]:
    macro_name = parsed.macro_name
    op = parsed.op_type
    type_tokens = parsed.type_tokens

    if not type_tokens:
        return []

    if len(type_tokens) == 1 and "_" in type_tokens[0]:
        parts = split_composite_type_token(type_tokens[0])
        names = COMPOSITE_TYPE_CONSTRAINTS.get(op)
        if names and len(names) == len(parts):
            return list(zip(names, parts, strict=False))
        return [(f"T{i + 1}", token) for i, token in enumerate(parts)]

    if macro_name in {
        "ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME",
        "ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME",
        "CUDA_MS_OP_TYPED_CLASS_NAME",
        "CUDA_MS_OP_VERSIONED_TYPED_CLASS_NAME",
        "CUDA_ONNX_OP_TYPED_CLASS_NAME",
        "CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME",
    }:
        return [("T", type_tokens[0])]

    if macro_name in {
        "ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME",
        "ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME",
        "CUDA_MS_OP_TWO_TYPED_CLASS_NAME",
    }:
        return [("T1", type_tokens[0]), ("T2", type_tokens[1])]

    if macro_name in {
        "ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME",
        "ONNX_OPERATOR_VERSIONED_THREE_TYPED_KERNEL_CLASS_NAME",
        "CUDA_MS_OP_THREE_TYPED_CLASS_NAME",
    }:
        if op == "GatherBlockQuantized":
            return [("T1", type_tokens[0]), ("T2", type_tokens[1]), ("Tind", type_tokens[2])]
        return [("T1", type_tokens[0]), ("T2", type_tokens[1]), ("T3", type_tokens[2])]

    return [("T", type_tokens[0])]


def split_composite_type_token(token: str) -> list[str]:
    parts: list[str] = []
    i = 0
    n = len(token)
    while i < n:
        if token[i] == "_":
            i += 1
            continue

        matched = None
        for t in SORTED_TYPE_TOKENS:
            if token.startswith(t, i):
                end = i + len(t)
                if end == n or token[end] == "_":
                    matched = t
                    break

        if matched is None:
            return token.split("_")

        parts.append(matched)
        i += len(matched)
        if i < n and token[i] == "_":
            i += 1

    return parts


def extract_function_table(source_text: str) -> str:
    start = source_text.find("static const BuildKernelCreateInfoFn function_table[] = {")
    if start < 0:
        raise RuntimeError("Failed to locate CUDA function_table start")

    end = source_text.find("};", start)
    if end < 0:
        raise RuntimeError("Failed to locate CUDA function_table end")

    return source_text[start:end]


def iter_entries(
    source_text: str,
    domain_filter: str,
    max_opset: int,
    op_filter: set[str],
    type_filter: set[str],
) -> Iterable[Entry]:
    table_text = extract_function_table(source_text)

    registration_id = 0
    for line in table_text.splitlines():
        if "BuildKernelCreateInfo<" not in line:
            continue

        macro_match = MACRO_PATTERN.search(line)
        if not macro_match:
            continue

        macro_expr = macro_match.group(1).strip()
        parsed = parse_macro(macro_expr, max_opset)
        if parsed is None:
            continue

        if parsed.domain_token != domain_filter:
            continue

        if op_filter and parsed.op_type not in op_filter:
            continue

        if parsed.since_version_start > parsed.since_version_end:
            continue

        domain_literal = DOMAIN_TOKEN_TO_LITERAL.get(parsed.domain_token)
        if domain_literal is None:
            registration_id += 1
            continue

        mem_in, mem_out = MEMORY_TYPE_OVERRIDES.get(parsed.op_type, ([], []))
        constraint_pairs = get_constraint_pairs(parsed)
        if not constraint_pairs:
            if not type_filter:
                yield Entry(
                    parsed.op_type,
                    parsed.since_version_start,
                    parsed.since_version_end,
                    domain_literal,
                    registration_id,
                    "",
                    "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED",
                    mem_in,
                    mem_out,
                )
            registration_id += 1
            continue

        emitted_any = False
        for constraint_name, token in constraint_pairs + FIXED_CONSTRAINTS.get(parsed.op_type, []):
            if type_filter and token not in type_filter:
                continue
            type_enum = TYPE_TO_ORT_ENUM.get(token)
            if type_enum is None:
                continue
            emitted_any = True
            yield Entry(
                parsed.op_type,
                parsed.since_version_start,
                parsed.since_version_end,
                domain_literal,
                registration_id,
                constraint_name,
                type_enum,
                mem_in,
                mem_out,
            )

        if not emitted_any and not type_filter:
            yield Entry(
                parsed.op_type,
                parsed.since_version_start,
                parsed.since_version_end,
                domain_literal,
                registration_id,
                "",
                "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED",
                mem_in,
                mem_out,
            )

        registration_id += 1


def write_output(output_path: Path, entries: list[Entry], argv: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "// This file is generated by tools/python/migrate_cuda_registrations.py",
        "// Do not edit by hand.",
        f"// Command: {' '.join(argv)}",
        "",
    ]

    for e in entries:
        in_mem = "{" + ", ".join(map(str, e.input_mem_types)) + "}" if e.input_mem_types else "{}"
        out_mem = "{" + ", ".join(map(str, e.output_mem_types)) + "}" if e.output_mem_types else "{}"
        n_in = len(e.input_mem_types)
        n_out = len(e.output_mem_types)
        lines.append(
            f'{{"{e.op_type}", {e.since_version_start}, {e.since_version_end}, "{e.domain}", '
            f"{e.registration_id}, "
            f'"{e.constraint_name}", {e.type_enum}, '
            f"{in_mem}, {n_in}, {out_mem}, {n_out}}},"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_critical_contrib_constraints(entries: list[Entry]) -> None:
    grouped: dict[tuple[str, int, int, str, int], set[str]] = {}
    for entry in entries:
        key = (entry.op_type, entry.since_version_start, entry.since_version_end, entry.domain, entry.registration_id)
        grouped.setdefault(key, set())
        if entry.constraint_name:
            grouped[key].add(entry.constraint_name)

    errors: list[str] = []
    for (op_type, start, end, domain, reg_id), emitted_constraints in sorted(grouped.items()):
        required = CRITICAL_CONTRIB_OP_CONSTRAINTS.get(op_type)
        if required is None:
            continue
        missing = required - emitted_constraints
        if missing:
            errors.append(
                f"{op_type} registration_id={reg_id} opset={start}-{end} domain='{domain}' "
                f"missing constraints: {sorted(missing)} (emitted: {sorted(emitted_constraints)})"
            )

    missing_ops = set(CRITICAL_CONTRIB_OP_CONSTRAINTS) - {k[0] for k in grouped}
    for op in sorted(missing_ops):
        errors.append(f"{op} has no generated contrib registrations.")

    if errors:
        raise RuntimeError("Critical contrib constraint validation failed:\n  - " + "\n  - ".join(errors))


def main() -> None:
    args = parse_args()
    if args.domain is None:
        args.domain = "kMSDomain" if args.contrib else "kOnnxDomain"

    if args.input == Path("onnxruntime/core/providers/cuda/cuda_execution_provider.cc") and args.contrib:
        args.input = Path("onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc")

    if args.output is None:
        args.output = (
            Path("onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_contrib_registrations.inc")
            if args.contrib
            else Path("onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc")
        )

    if args.ops is None:
        args.ops = []

    if args.types is None:
        args.types = []

    source_text = args.input.read_text(encoding="utf-8")
    op_filter = set(args.ops) if args.ops else set()
    type_filter = set(args.types) if args.types else set()

    entries = sorted(
        set(
            iter_entries(
                source_text,
                domain_filter=args.domain,
                max_opset=args.max_opset,
                op_filter=op_filter,
                type_filter=type_filter,
            )
        )
    )

    if args.check_critical_contrib:
        if not args.contrib:
            raise RuntimeError("--check-critical-contrib requires --contrib.")
        validate_critical_contrib_constraints(entries)

    write_output(
        args.output,
        entries,
        argv=["python", "tools/python/migrate_cuda_registrations.py", *sys.argv[1:]],
    )

    print(
        f"Generated {len(entries)} registrations to {args.output} "
        f"(ops={','.join(args.ops) if args.ops else 'ALL'}, "
        f"types={','.join(args.types) if args.types else 'ALL'})"
    )


if __name__ == "__main__":
    main()
