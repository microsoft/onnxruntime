import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

DOMAIN = "com.microsoft"
OP = "SplitScaledSiLUF16"


def definition(opset):
    silu = helper.make_node("ScaledSiLU", ["Prefix"], ["Y"], domain=DOMAIN)
    silu.attribute.append(AttributeProto(name="alpha", ref_attr_name="alpha", type=AttributeProto.FLOAT))
    return helper.make_function(
        DOMAIN, OP, ["X", "Sizes"], ["Y", "Tail"],
        [helper.make_node("Split", ["X", "Sizes"], ["Prefix", "Tail"], axis=-1), silu],
        [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)],
        attributes=["alpha"],
    )


def fuse(model):
    result = deepcopy(model)
    if any(f.domain == DOMAIN and f.name == OP for f in model.functions):
        raise ValueError("SplitScaledSiLUF16 already defined")
    imports = {v.domain: v.version for v in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires com.microsoft opset1 and ONNX opset>=13")
    nodes = list(result.graph.node)
    uses = Counter(i for n in nodes for i in n.input)
    outputs = {v.name for v in result.graph.output}
    consumers = {i: n for n in nodes for i in n.input if uses[i] == 1}
    types = {v.name: v.type.tensor_type.elem_type
             for v in [*result.graph.input, *result.graph.output, *result.graph.value_info]}
    tensors = {v.name: v for v in result.graph.initializer}
    types.update({v.name: v.data_type for v in result.graph.initializer})
    for n in nodes:
        if (n.domain == "" and n.op_type == "Constant" and len(n.attribute) == 1 and
                n.attribute[0].name == "value" and n.attribute[0].type == AttributeProto.TENSOR and
                not n.input and len(n.output) == 1):
            tensors[n.output[0]] = n.attribute[0].t
    replaced = {}
    removed = set()
    dead_values = set()
    for split in nodes:
        if (split.domain != "" or split.op_type != "Split" or len(split.input) != 2 or
                len(split.output) != 2 or split.output[0] in outputs or uses[split.output[0]] != 1):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in split.attribute}
        if attrs != {"axis": -1}:
            continue
        silu = consumers.get(split.output[0])
        if (silu is None or silu.domain != DOMAIN or silu.op_type != "ScaledSiLU" or
                len(silu.input) != 1 or len(silu.output) != 1 or
                any(a.name != "alpha" or a.type != AttributeProto.FLOAT for a in silu.attribute)):
            continue
        sizes = tensors.get(split.input[1])
        if sizes is None or sizes.data_type != TensorProto.INT64 or list(sizes.dims) != [2]:
            continue
        if any(value <= 0 for value in numpy_helper.to_array(sizes)):
            continue
        if not all(types.get(v) == TensorProto.FLOAT16
                   for v in [split.input[0], *split.output, silu.output[0]]):
            continue
        alpha = next((a.f for a in silu.attribute if a.name == "alpha"), 1.0)
        fused = helper.make_node(OP, list(split.input), [silu.output[0], split.output[1]],
                                 name=split.name, domain=DOMAIN, alpha=alpha)
        replaced[id(split)] = fused
        removed.add(id(silu))
        dead_values.add(split.output[0])
    if not replaced:
        raise ValueError("No eligible FP16 Split/ScaledSiLU pairs")
    del result.graph.node[:]
    result.graph.node.extend(replaced.get(id(n), n) for n in nodes if id(n) not in removed)
    retained = [v for v in result.graph.value_info if v.name not in dead_values]
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
    if args.output.exists():
        raise FileExistsError(args.output)
    original = onnx.load(args.source, load_external_data=False)
    result, count = fuse(original)
    if count != args.expect_count:
        raise ValueError(f"Expected {args.expect_count} fusions, got {count}")
    if [v.SerializeToString() for v in original.graph.initializer] != [
        v.SerializeToString() for v in result.graph.initializer
    ]:
        raise ValueError("Initializer descriptors changed")
    if args.output.parent != args.source.parent:
        raise ValueError("Output must share the source directory to preserve external data resolution")
    onnx.save_model(result, args.output)
    onnx.checker.check_model(str(args.output))
    print(f"PASS {count} FP16 Split/ScaledSiLU fusions; not GPU-qualified")


if __name__ == "__main__":
    main()
