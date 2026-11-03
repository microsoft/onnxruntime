import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto, helper

from fuse_hyper_connections import DOMAIN, POST, function_definitions

OP = "HyperConnectionSharedPostMixF16"


def definition(opset):
    post = helper.make_node("HyperConnectionPostMix", ["X", "Branch", "Gate"], ["Y"], domain=DOMAIN)
    post.attribute.append(AttributeProto(name="num_branches", ref_attr_name="num_branches",
                                        type=AttributeProto.INT))
    return helper.make_function(DOMAIN, OP, ["X", "Expert", "G", "D", "S", "Shared", "SharedGate"], ["Y"], [
        helper.make_node("Sigmoid", ["SharedGate"], ["SharedSigmoid"]),
        helper.make_node("GatedAdd", ["Expert", "Shared", "SharedSigmoid"], ["Branch"], domain=DOMAIN),
        helper.make_node("Div", ["G", "D"], ["Divided"]),
        helper.make_node("Sigmoid", ["Divided"], ["SigmoidGate"]),
        helper.make_node("Mul", ["SigmoidGate", "S"], ["Gate"]),
        post,
    ], [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)],
        attributes=["num_branches"])


def fuse(original):
    model = deepcopy(original)
    if any(f.domain == DOMAIN and f.name == OP for f in model.functions):
        raise ValueError("Shared-post fusion is already defined")
    imports = {v.domain: v.version for v in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires com.microsoft opset1 and ONNX opset>=13")
    post_definitions = [f for f in model.functions if f.domain == DOMAIN and f.name == POST]
    if (len(post_definitions) != 1 or post_definitions[0].SerializeToString() !=
            function_definitions(imports[""])[POST].SerializeToString()):
        raise ValueError("Requires the exact qualified post-mix reference function")
    nodes = list(model.graph.node)
    producers = {v: n for n in nodes for v in n.output}
    uses = Counter(v for n in nodes for v in n.input)
    exposed = {v.name for v in model.graph.output}
    types = {v.name: v.type.tensor_type.elem_type for v in [
        *model.graph.input, *model.graph.output, *model.graph.value_info]}
    types.update({v.name: v.data_type for v in model.graph.initializer})
    removed = set()
    replacements = {}
    for node in nodes:
        if (node.domain != DOMAIN or node.op_type != POST or len(node.input) != 5 or
                len(node.output) != 1 or {a.name for a in node.attribute} != {"num_branches"}):
            continue
        add = producers.get(node.input[1])
        if (add is None or add.domain != DOMAIN or add.op_type != "GatedAdd" or add.attribute or
                len(add.input) != 3 or len(add.output) != 1 or
                uses[add.output[0]] != 1 or add.output[0] in exposed):
            continue
        sigmoid = producers.get(add.input[2])
        if (sigmoid is None or sigmoid.domain != "" or sigmoid.op_type != "Sigmoid" or
                sigmoid.attribute or len(sigmoid.input) != 1 or len(sigmoid.output) != 1 or
                uses[sigmoid.output[0]] != 1 or sigmoid.output[0] in exposed):
            continue
        inputs = [node.input[0], add.input[0], *node.input[2:], add.input[1], sigmoid.input[0]]
        if any(types.get(v) != TensorProto.FLOAT16 for v in [
            *inputs, *node.output, *add.output, *sigmoid.output]):
            continue
        replacement = helper.make_node(OP, inputs, list(node.output), name=node.name, domain=DOMAIN)
        replacement.attribute.extend(node.attribute)
        replacements[id(node)] = replacement
        removed.update([*sigmoid.output, *add.output])
    if not replacements:
        raise ValueError("No eligible single-consumer FP16 shared-post chains")
    del model.graph.node[:]
    model.graph.node.extend(replacements.get(id(n), n) for n in nodes if not any(v in removed for v in n.output))
    infos = [v for v in model.graph.value_info if v.name not in removed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(infos)
    model.functions.append(definition(imports[""]))
    return model, len(replacements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.source.parent != args.output.parent:
        raise ValueError("New output must share source directory")
    original = onnx.load(args.source, load_external_data=False)
    model, count = fuse(original)
    if count != args.expect_count:
        raise ValueError(f"Expected {args.expect_count} chains, found {count}")
    if [v.SerializeToString() for v in model.graph.initializer] != [
        v.SerializeToString() for v in original.graph.initializer
    ]:
        raise ValueError("Initializer descriptors changed")
    onnx.save_model(model, args.output)
    onnx.checker.check_model(str(args.output))
    print(f"PASS {count} exact-reference shared-post rewrites; not GPU-qualified")


if __name__ == "__main__":
    main()
