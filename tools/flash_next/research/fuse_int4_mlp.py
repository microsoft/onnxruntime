import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def fuse_int4_mlp(source):
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in source.graph.node for a in node.attribute):
        raise ValueError("Subgraphs require implicit-input analysis and are not supported")
    model = onnx.ModelProto()
    model.CopyFrom(source)
    nodes = list(model.graph.node)
    producers = {name: i for i, node in enumerate(nodes) for name in node.output}
    consumers = defaultdict(list)
    for i, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(i)
    graph_outputs = {value.name for value in model.graph.output}
    initializers = {value.name: value for value in model.graph.initializer}
    names = {node.name for node in nodes}
    removed, replacements, matches = set(), {}, []

    def producer(name, op_type, domain=""):
        index = producers.get(name)
        if index is None:
            return None
        node = nodes[index]
        if node.op_type != op_type or node.domain != domain or len(node.output) != 1:
            return None
        return index

    def attributes(node):
        values = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if not set(values) <= {"K", "N", "bits", "block_size", "accuracy_level"}:
            return None
        values.setdefault("bits", 4)
        values.setdefault("accuracy_level", 0)
        k, n = values.get("K", 0), values.get("N", 0)
        if (k <= 0 or n <= 0 or k % 32 or values["bits"] != 4 or
                values.get("block_size") != 32 or len(node.input) != 3):
            return None
        weight, scale = (initializers.get(name) for name in node.input[1:])
        if (weight is None or scale is None or
                weight.data_type != TensorProto.UINT8 or
                scale.data_type != TensorProto.FLOAT16 or
                list(weight.dims) != [n, k // 32, 16] or
                list(scale.dims) != [n, k // 32]):
            return None
        return values

    for final_index, final in enumerate(nodes):
        if final.domain or final.op_type != "Mul" or len(final.input) != 2 or len(final.output) != 1:
            continue
        for side in (0, 1):
            act_index = producer(final.input[side], "Mul")
            up_index = producer(final.input[1 - side], "MatMulNBits", "com.microsoft")
            if act_index is None or up_index is None or len(nodes[act_index].input) != 2:
                continue
            act, up = nodes[act_index], nodes[up_index]
            for slot in (0, 1):
                gate_index = producer(act.input[slot], "MatMulNBits", "com.microsoft")
                sig_index = producer(act.input[1 - slot], "Sigmoid")
                if gate_index is None or sig_index is None:
                    continue
                gate, sig = nodes[gate_index], nodes[sig_index]
                if list(sig.input) != [gate.output[0]]:
                    continue
                attrs = attributes(gate)
                if attrs is None or attrs != attributes(up) or gate.input[0] != up.input[0]:
                    continue
                indices = {final_index, act_index, up_index, gate_index, sig_index}
                intermediates = [gate.output[0], up.output[0], sig.output[0], act.output[0]]
                if (len(indices) != 5 or removed.intersection(indices) or
                        graph_outputs.intersection(intermediates) or
                        sorted(consumers[gate.output[0]]) != sorted([sig_index, act_index]) or
                        consumers[up.output[0]] != [final_index] or
                        consumers[sig.output[0]] != [act_index] or
                        consumers[act.output[0]] != [final_index]):
                    continue
                name = f"WebGPUInt4Mlp_{final_index}"
                if name in names:
                    raise ValueError(f"Fused node name already exists: {name}")
                replacements[final_index] = helper.make_node(
                    "MatMulNBitsMlp",
                    [gate.input[0], "", "", gate.input[1], gate.input[2], "",
                     up.input[1], up.input[2], ""],
                    list(final.output), name=name, domain="com.microsoft",
                    activation="silu", **attrs)
                removed.update(indices)
                names.add(name)
                matches.append({"output": final.output[0], "K": attrs["K"], "N": attrs["N"]})
    rewritten = []
    for index, node in enumerate(nodes):
        if index in replacements:
            rewritten.append(replacements[index])
        elif index not in removed:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return model, {"fusions": len(matches), "removed_nodes": len(removed),
                   "nodes_before": len(nodes), "nodes_after": len(rewritten), "matches": matches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model, report = fuse_int4_mlp(onnx.load(args.input, load_external_data=False))
    if not report["fusions"]:
        raise ValueError("No supported bias-free INT4 SiLU projection pairs found")
    if args.output:
        with tempfile.NamedTemporaryFile(prefix=".int4-mlp-", suffix=".onnx",
                                         dir=args.output.parent) as temporary:
            onnx.save(model, temporary.name)
            onnx.checker.check_model(temporary.name)
            os.link(temporary.name, args.output)
    print(json.dumps(report, indent=2))
