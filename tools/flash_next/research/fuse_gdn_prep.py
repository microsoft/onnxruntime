import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = "com.microsoft"
OP = "GdnProjPrepF16"


def definition(opset):
    constants = [
        helper.make_node("Constant", [], ["QkvZSizes"],
                         value=numpy_helper.from_array(np.array([10240, 6144], np.int64))),
        helper.make_node("Constant", [], ["AbSizes"],
                         value=numpy_helper.from_array(np.array([48, 48], np.int64))),
    ]
    return helper.make_function(DOMAIN, OP, ["Projection", "AB", "W", "Bias", "State"],
        ["Conv", "Z", "A32", "B32", "Present"], [
            *constants,
            helper.make_node("Split", ["Projection", "QkvZSizes"], ["Qkv", "Z"], axis=-1),
            helper.make_node("Split", ["AB", "AbSizes"], ["A", "B"], axis=-1),
            helper.make_node("Transpose", ["Qkv"], ["CF"], perm=[0, 2, 1]),
            helper.make_node("CausalConvWithState", ["CF", "W", "Bias", "State"],
                             ["ConvCF", "Present"], domain=DOMAIN, ndim=1,
                             activation="silu", state_window=0),
            helper.make_node("Transpose", ["ConvCF"], ["Conv"], perm=[0, 2, 1]),
            helper.make_node("Cast", ["A"], ["A32"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["B"], ["B32"], to=TensorProto.FLOAT),
        ], [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)])


def fuse(model):
    result = deepcopy(model)
    if any(f.domain == DOMAIN and f.name == OP for f in model.functions):
        raise ValueError("GdnProjPrepF16 already defined")
    imports = {v.domain: v.version for v in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires com.microsoft opset1 and ONNX opset>=13")
    nodes = list(result.graph.node)
    producers = {o: n for n in nodes for o in n.output}
    consumers = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    uses = Counter(i for n in nodes for i in n.input)
    exposed = {v.name for v in result.graph.output}
    infos = {v.name: v.type.tensor_type for v in
             [*result.graph.input, *result.graph.output, *result.graph.value_info]}
    tensors = {v.name: v for v in result.graph.initializer}
    for node in nodes:
        if (node.domain == "" and node.op_type == "Constant" and len(node.attribute) == 1 and
                node.attribute[0].name == "value" and node.attribute[0].type == 4):
            tensors[node.output[0]] = node.attribute[0].t
    types = {k: v.elem_type for k, v in infos.items()}
    types.update({k: v.data_type for k, v in tensors.items()})

    def attrs(node):
        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    def is_transpose(node):
        return (node is not None and node.domain == "" and node.op_type == "Transpose" and
                len(node.input) == 1 and len(node.output) == 1 and attrs(node) == {"perm": [0, 2, 1]})

    def split_sizes(node, expected):
        if (node is None or node.domain != "" or node.op_type != "Split" or len(node.input) != 2 or
                len(node.output) != 2 or attrs(node) != {"axis": -1}):
            return False
        tensor = tensors.get(node.input[1])
        return (tensor is not None and tensor.data_type == TensorProto.INT64 and
                list(tensor.dims) == [2] and numpy_helper.to_array(tensor).tolist() == expected)

    replaced = {}
    removed = set()
    dead = set()
    for gdn in nodes:
        if (gdn.domain != DOMAIN or gdn.op_type != "GatedDeltaNet" or len(gdn.input) < 6 or
                gdn.input[1] or gdn.input[2]):
            continue
        after = producers.get(gdn.input[0])
        if not is_transpose(after):
            continue
        conv = producers.get(after.input[0])
        if (conv is None or conv.domain != DOMAIN or conv.op_type != "CausalConvWithState" or
                len(conv.input) != 4 or len(conv.output) != 2 or
                attrs(conv) != {"ndim": 1, "activation": b"silu", "state_window": 0}):
            continue
        before = producers.get(conv.input[0])
        if not is_transpose(before):
            continue
        qkvz = producers.get(before.input[0])
        if not split_sizes(qkvz, [10240, 6144]) or before.input[0] != qkvz.output[0]:
            continue
        a = producers.get(gdn.input[4])
        b = producers.get(gdn.input[5])
        if any(n is None or n.domain != "" or n.op_type != "Cast" or len(n.input) != 1 or
               len(n.output) != 1 or attrs(n) != {"to": TensorProto.FLOAT} for n in (a, b)):
            continue
        ab = producers.get(a.input[0])
        if not split_sizes(ab, [48, 48]) or [a.input[0], b.input[0]] != list(ab.output):
            continue
        intermediate = [qkvz.output[0], before.output[0], conv.output[0], *ab.output]
        if any(uses[v] != 1 or v in exposed for v in intermediate):
            continue
        inputs = [qkvz.input[0], ab.input[0], conv.input[1], conv.input[2], conv.input[3]]
        outputs = [after.output[0], qkvz.output[1], a.output[0], b.output[0], conv.output[1]]
        if (not all(types.get(v) == TensorProto.FLOAT16 for v in [*inputs, outputs[0], outputs[1], outputs[4]]) or
                not all(types.get(v) == TensorProto.FLOAT for v in outputs[2:4])):
            continue
        projection_type = infos.get(inputs[0])
        ab_type = infos.get(inputs[1])
        if (projection_type is None or ab_type is None or
                len(projection_type.shape.dim) != 3 or projection_type.shape.dim[-1].dim_value != 16384 or
                len(ab_type.shape.dim) != 3 or ab_type.shape.dim[-1].dim_value != 96):
            continue
        chain = [qkvz, ab, before, conv, after, a, b]
        # Both projection inputs must already be produced at the insertion point.
        index = max(nodes.index(n) for n in chain)
        if any(v in producers and nodes.index(producers[v]) >= index for v in inputs):
            continue
        if any(nodes.index(n) < index and n not in chain for v in outputs for n in consumers.get(v, [])):
            continue
        replaced[id(nodes[index])] = helper.make_node(OP, inputs, outputs, name=conv.name + "/ProjPrep",
                                                       domain=DOMAIN)
        removed.update(id(n) for n in chain if n is not nodes[index])
        dead.update(intermediate)
    if not replaced:
        raise ValueError("No eligible exact GDN projection preparation chains")
    del result.graph.node[:]
    result.graph.node.extend(replaced.get(id(n), n) for n in nodes if id(n) not in removed)
    retained = [v for v in result.graph.value_info if v.name not in dead]
    del result.graph.value_info[:]
    result.graph.value_info.extend(retained)
    result.functions.append(definition(imports[""]))
    return result, len(replaced)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent != args.source.parent:
        raise ValueError("New output must share source directory")
    original = onnx.load(args.source, load_external_data=False)
    result, count = fuse(original)
    if count != args.expect_count:
        raise ValueError(f"Expected {args.expect_count} chains, found {count}")
    if [v.SerializeToString() for v in original.graph.initializer] != [
        v.SerializeToString() for v in result.graph.initializer
    ]:
        raise ValueError("Initializer descriptors changed")
    onnx.save_model(result, args.output)
    onnx.checker.check_model(str(args.output))
    print(f"PASS {count} GDN projection-preparation fusions; not GPU-qualified")


if __name__ == "__main__":
    main()
