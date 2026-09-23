#!/usr/bin/env python3
"""Inventory or bypass activation QDQ pairs in an ONNX model for EP experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import onnx
import onnx_ir as ir


@dataclass(frozen=True)
class QdqPair:
    quantize: ir.Node
    dequantize: ir.Node

    @property
    def name(self) -> str:
        name = self.quantize.name or self.quantize.outputs[0].name
        if name is None:
            raise ValueError("an activation QuantizeLinear must have a node or output name")
        return name


def activation_pairs(graph: ir.Graph) -> list[QdqPair]:
    pairs = []
    for quantize in graph:
        if quantize.op_type != "QuantizeLinear" or quantize.domain not in ("", "ai.onnx", "com.microsoft"):
            continue
        source = quantize.inputs[0]
        if source is None or source.is_initializer():
            continue
        for output in quantize.outputs:
            for use in output.uses():
                if use.idx == 0 and use.node.op_type == "DequantizeLinear":
                    pairs.append(QdqPair(quantize, use.node))
    return pairs


def attention_core_groups(graph: ir.Graph) -> list[tuple[ir.Node, QdqPair, QdqPair, QdqPair]]:
    pairs = activation_pairs(graph)
    by_dequantized_output = {pair.dequantize.outputs[0]: pair for pair in pairs}
    groups = []
    for softmax in graph:
        if softmax.op_type != "Softmax":
            continue
        softmax_input = softmax.inputs[0]
        if softmax_input is None:
            continue
        add_pair = by_dequantized_output.get(softmax_input)
        add = add_pair.quantize.inputs[0].producer() if add_pair and add_pair.quantize.inputs[0] else None
        if add is None or add.op_type != "Add":
            continue
        score_pairs = [
            pair for value in add.inputs
            if (pair := by_dequantized_output.get(value)) is not None
            and pair.quantize.inputs[0] is not None
            and pair.quantize.inputs[0].producer() is not None
            and pair.quantize.inputs[0].producer().op_type == "MatMul"
        ]
        probability_pairs = [pair for pair in pairs if pair.quantize.inputs[0] is softmax.outputs[0]]
        if len(score_pairs) != 1 or len(probability_pairs) != 1:
            raise ValueError(f"{softmax.name}: expected one QDQ-wrapped score MatMul and one Softmax output QDQ")
        groups.append((softmax, score_pairs[0], add_pair, probability_pairs[0]))
    if not groups:
        raise ValueError("no QDQ-wrapped MatMul -> Add -> Softmax attention pattern was found")
    return groups


def attention_core_pairs(graph: ir.Graph) -> list[QdqPair]:
    return [pair for _, *group in attention_core_groups(graph) for pair in group]


def _validate_pair(pair: QdqPair, graph: ir.Graph) -> None:
    q, dq = pair.quantize, pair.dequantize
    if q.domain != dq.domain or len(q.inputs) != len(dq.inputs) or len(q.inputs) not in (2, 3):
        raise ValueError(f"{pair.name}: incompatible Q/DQ operators")
    if any(q.inputs[index] is not dq.inputs[index] for index in range(1, len(q.inputs))):
        raise ValueError(f"{pair.name}: Q/DQ scale or zero point differs")
    if {key: attr.value for key, attr in q.attributes.items()} != {
        key: attr.value for key, attr in dq.attributes.items()
    }:
        raise ValueError(f"{pair.name}: Q/DQ attributes differ")
    if len(q.outputs) != 1 or len(dq.outputs) != 1:
        raise ValueError(f"{pair.name}: expected single-output Q/DQ operators")
    if q.inputs[0].type is not None and dq.outputs[0].type is not None:
        if q.inputs[0].type != dq.outputs[0].type:
            raise ValueError(f"{pair.name}: bypass would change the downstream data type")
    if any(use.node is not dq for use in q.outputs[0].uses()):
        raise ValueError(f"{pair.name}: quantized output also feeds other nodes")
    if q.inputs[0] is None:
        raise ValueError(f"{pair.name}: QuantizeLinear has no data input")
    if dq.outputs[0] in graph.outputs and dq.outputs[0].type is None and q.inputs[0].type is None:
        raise ValueError(f"{pair.name}: graph output has no type metadata")


def bypass_pairs(model: ir.Model, selected: list[QdqPair]) -> None:
    graph = model.graph
    if not selected:
        raise ValueError("no activation QDQ pairs matched")
    if len({pair.quantize for pair in selected}) != len(selected):
        raise ValueError("selection includes a QuantizeLinear with multiple DequantizeLinear consumers")
    for pair in selected:
        _validate_pair(pair, graph)
    for pair in selected:
        source = pair.quantize.inputs[0]
        if source is None:
            raise ValueError(f"{pair.name}: QuantizeLinear has no data input")
        output = pair.dequantize.outputs[0]
        if output in graph.outputs:
            name = output.name
            output.name = f"{name}_removed_by_qdq_ablation"
            replacement = ir.Value(name=name, type=output.type or source.type, shape=output.shape or source.shape)
            for index, graph_output in enumerate(graph.outputs):
                if graph_output is output:
                    graph.outputs[index] = replacement
            graph.append(ir.Node("", "Identity", [source], outputs=[replacement], name=f"qdq_ablation_output_{name}"))
        output.replace_all_uses_with(source)
    for pair in selected:
        graph.remove(pair.dequantize, safe=True)
        graph.remove(pair.quantize, safe=True)
    used_initializers = {
        value.name
        for node in graph
        for value in node.inputs
        if value is not None and value.is_initializer()
    }
    used_initializers.update(value.name for value in graph.outputs if value.is_initializer())
    for name in list(graph.initializers):
        if name not in used_initializers:
            graph.initializers.pop(name)
    graph.sort()


def compile_regex(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ValueError(f"invalid regular expression {pattern!r}: {error}") from error


def select_pairs(
    pairs: list[QdqPair],
    bypass: re.Pattern[str] | None,
    keep: re.Pattern[str] | None,
    protected: list[re.Pattern[str]],
    names: set[str] | None = None,
) -> list[QdqPair]:
    if names:
        missing = names - {pair.name for pair in pairs}
        if missing:
            raise ValueError(f"unknown activation QDQ pairs: {sorted(missing)}")
        excluded = {
            name for name in names
            if any(pattern.search(name) for pattern in protected)
        }
        if excluded:
            raise ValueError(f"explicitly selected pairs are also protected: {sorted(excluded)}")
    selected = []
    for pair in pairs:
        if any(pattern.search(pair.name) for pattern in protected):
            continue
        if names is not None and pair.name in names:
            selected.append(pair)
        elif bypass is not None and bypass.search(pair.name):
            selected.append(pair)
        elif keep is not None and not keep.search(pair.name):
            selected.append(pair)
    return selected


def list_pairs(pairs: list[QdqPair], group_regex: re.Pattern[str] | None, show_pairs: bool) -> None:
    print(f"Activation QDQ pairs: {len(pairs)}")
    if group_regex is not None:
        groups = Counter(
            (match.group(1) if match.lastindex else match.group(0)) if (match := group_regex.search(pair.name))
            else "<ungrouped>"
            for pair in pairs
        )
        ordered = sorted(groups.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]))
        for name, count in ordered:
            print(f"  {name}: {count}")
    if show_pairs:
        for pair in pairs:
            print(pair.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Source ONNX model.")
    parser.add_argument("--list", action="store_true", help="List pairs without writing a model.")
    parser.add_argument("--show-pairs", action="store_true", help="Also print each pair's QuantizeLinear name.")
    parser.add_argument("--group-regex", help="Group list by first capture (or the full match).")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--bypass", metavar="REGEX", help="Bypass matching activation QDQ pairs.")
    selection.add_argument("--keep", metavar="REGEX", help="Bypass all activation QDQ pairs except matching ones.")
    selection.add_argument(
        "--bypass-name", action="append", metavar="NAME",
        help="Bypass an exact QuantizeLinear name (repeatable). Use --list --show-pairs to discover names.",
    )
    selection.add_argument(
        "--attention-core", action="store_true",
        help="Bypass the QK MatMul, mask Add, and Softmax QDQ pairs in every matching attention pattern.",
    )
    parser.add_argument(
        "--protect", action="append", default=[], metavar="REGEX",
        help="Never bypass matching pairs (repeatable; e.g. a repaired attention mask).",
    )
    parser.add_argument("--output", type=Path, help="Output ONNX model, beside the source to reuse external weights.")
    return parser.parse_args()


def write_variant(
    model: ir.Model,
    source: Path,
    output: Path,
    count: int,
    selection: dict[str, object] | None = None,
) -> None:
    if source == output or output.exists():
        raise ValueError(f"output must be a new file, distinct from the source: {output}")
    if output.suffix.lower() != ".onnx":
        raise ValueError("output must be an .onnx file")
    if source.parent != output.parent:
        raise ValueError("output must be beside the source model so external weights remain accessible")
    manifest = output.with_suffix(".ablation.json")
    if selection is not None and manifest.exists():
        raise ValueError(f"selection manifest already exists: {manifest}")
    model.metadata_props["qdq_ablation_pairs_removed"] = str(count)
    if selection is not None:
        names = selection["selected_pairs"]
        model.metadata_props["qdq_ablation_selection_sha256"] = hashlib.sha256(
            json.dumps(names, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    proto = ir.serde.serialize_model(model)
    for initializer in proto.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            location = next((entry.value for entry in initializer.external_data if entry.key == "location"), "")
            if not location or not (output.parent / location).is_file():
                raise ValueError(f"external data for {initializer.name!r} was not found beside the output: {location}")
    onnx.save(proto, output)
    if selection is not None:
        manifest.write_text(
            json.dumps({"source": str(source), "model": str(output), "pairs_removed": count, **selection}, indent=2)
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    source = args.model.resolve()
    if not source.is_file():
        raise ValueError(f"model does not exist: {source}")
    if args.list == (args.output is not None):
        raise ValueError("use --list to inspect or --output with --bypass/--keep to write a variant")
    if args.list and (args.bypass is not None or args.keep is not None or args.bypass_name or args.attention_core):
        raise ValueError("selection options require --output")
    if not args.list and args.bypass is None and args.keep is None and not args.bypass_name and not args.attention_core:
        raise ValueError("a bypass selection is required when writing a variant")
    if args.bypass_name and len(args.bypass_name) != len(set(args.bypass_name)):
        raise ValueError("--bypass-name must not repeat a node name")

    model = ir.load(source)
    pairs = activation_pairs(model.graph)
    if args.list:
        list_pairs(pairs, compile_regex(args.group_regex) if args.group_regex else None, args.show_pairs)
        return

    protected = [compile_regex(pattern) for pattern in args.protect]
    if args.attention_core:
        selected = attention_core_pairs(model.graph)
        excluded = [pair.name for pair in selected if any(pattern.search(pair.name) for pattern in protected)]
        if excluded:
            raise ValueError(f"attention-core pairs are also protected: {excluded}")
    else:
        selected = select_pairs(
            pairs,
            compile_regex(args.bypass) if args.bypass is not None else None,
            compile_regex(args.keep) if args.keep is not None else None,
            protected,
            set(args.bypass_name) if args.bypass_name else None,
        )
    bypass_pairs(model, selected)
    output = args.output.resolve()
    write_variant(
        model,
        source,
        output,
        len(selected),
        {
            "mode": (
                "attention-core" if args.attention_core else
                "exact" if args.bypass_name else
                "bypass" if args.bypass is not None else "keep"
            ),
            "pattern": args.bypass if args.bypass is not None else args.keep,
            "protected": args.protect,
            "selected_pairs": [pair.name for pair in selected],
        },
    )
    print(f"Bypassed {len(selected)} / {len(pairs)} activation QDQ pairs: {output}")


if __name__ == "__main__":
    main()
