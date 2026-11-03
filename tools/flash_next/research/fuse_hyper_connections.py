import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import onnx
from onnx import AttributeProto, TensorProto, helper


DOMAIN = "com.microsoft"
PRE = "HyperConnectionPreMixSigmoidF16"
POST = "HyperConnectionPostMixGateF16"


def function_definitions(opset):
    def reference(node, name, kind):
        node.attribute.append(AttributeProto(name=name, ref_attr_name=name, type=kind))
        return node

    pre = reference(
        helper.make_node("HyperConnectionPreMix", ["X", "Gate"], ["Y"], domain=DOMAIN),
        "num_branches", AttributeProto.INT,
    )
    reference(pre, "reduction_scale", AttributeProto.FLOAT)
    post = reference(
        helper.make_node("HyperConnectionPostMix", ["X", "Z", "Gate"], ["Y"], domain=DOMAIN),
        "num_branches", AttributeProto.INT,
    )
    imports = [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)]
    return {
        PRE: helper.make_function(
            DOMAIN, PRE, ["X", "G"], ["Y"],
            [helper.make_node("Sigmoid", ["G"], ["Gate"]), pre], imports,
            attributes=["num_branches", "reduction_scale"],
        ),
        POST: helper.make_function(
            DOMAIN, POST, ["X", "Z", "G", "D", "S"], ["Y"],
            [
                helper.make_node("Div", ["G", "D"], ["Divided"]),
                helper.make_node("Sigmoid", ["Divided"], ["SigmoidGate"]),
                helper.make_node("Mul", ["SigmoidGate", "S"], ["Gate"]),
                post,
            ], imports, attributes=["num_branches"],
        ),
    }


def fuse(model):
    model = deepcopy(model)
    if any(f.domain == DOMAIN and f.name in (PRE, POST) for f in model.functions):
        raise ValueError("Model already defines a fusion function")
    if any(n.domain == DOMAIN and n.op_type in (PRE, POST) for n in model.graph.node):
        raise ValueError("Model already calls a fusion function")
    imports = {x.domain: x.version for x in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Fusion requires com.microsoft opset1 and ONNX opset>=13")
    nodes = list(model.graph.node)
    producers = {o: n for n in nodes for o in n.output}
    uses = Counter(i for n in nodes for i in n.input)
    exposed = {o.name for o in model.graph.output}
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]
    }
    initializers = {v.name: v for v in model.graph.initializer}
    types.update({v.name: v.data_type for v in model.graph.initializer})
    removed = set()
    replacements = {}
    counts = {PRE: 0, POST: 0}

    def producer(value, op, owner):
        node = producers.get(value)
        if (node is None or node.domain != "" or node.op_type != op or
                node.attribute or len(node.output) != 1 or value in exposed or
                uses[value] != 1 or value not in owner.input):
            return None
        return node

    def fp16(values):
        return all(types.get(v) == TensorProto.FLOAT16 for v in values)

    def scalar(value):
        tensor = initializers.get(value)
        if tensor is None:
            constant = producers.get(value)
            if (constant is not None and constant.domain == "" and
                    constant.op_type == "Constant" and not constant.input and
                    len(constant.output) == 1 and len(constant.attribute) == 1 and
                    constant.attribute[0].name == "value" and
                    constant.attribute[0].type == AttributeProto.TENSOR):
                tensor = constant.attribute[0].t
        return tensor is not None and not tensor.dims and tensor.data_type == TensorProto.FLOAT16

    for node in nodes:
        if node.domain != DOMAIN or len(node.output) != 1:
            continue
        attrs = {a.name for a in node.attribute}
        chain = []
        if node.op_type == "HyperConnectionPreMix" and len(node.input) == 2:
            sigmoid = producer(node.input[1], "Sigmoid", node)
            if (sigmoid is None or len(sigmoid.input) != 1 or
                    attrs != {"num_branches", "reduction_scale"}):
                continue
            inputs = [node.input[0], sigmoid.input[0]]
            kind = PRE
            chain = [sigmoid]
        elif node.op_type == "HyperConnectionPostMix" and len(node.input) == 3:
            mul = producer(node.input[2], "Mul", node)
            if mul is None or len(mul.input) != 2 or attrs != {"num_branches"}:
                continue
            sigmoid = producer(mul.input[0], "Sigmoid", mul)
            if sigmoid is None or len(sigmoid.input) != 1:
                continue
            div = producer(sigmoid.input[0], "Div", sigmoid)
            if (div is None or len(div.input) != 2 or
                    not scalar(div.input[1]) or not scalar(mul.input[1])):
                continue
            inputs = [node.input[0], node.input[1], div.input[0], div.input[1], mul.input[1]]
            kind = POST
            chain = [div, sigmoid, mul]
        else:
            continue
        values = [*inputs, *node.output, *(o for n in chain for o in n.output)]
        if not fp16(values):
            continue
        replacement = helper.make_node(kind, inputs, list(node.output), name=node.name, domain=DOMAIN)
        replacement.attribute.extend(node.attribute)
        replacements[id(node)] = replacement
        removed.update(o for n in chain for o in n.output)
        counts[kind] += 1

    if not any(counts.values()):
        raise ValueError("No eligible single-consumer FP16 HC chains found")
    del model.graph.node[:]
    model.graph.node.extend(
        replacements.get(id(n), n) for n in nodes if not any(o in removed for o in n.output)
    )
    retained_info = [v for v in model.graph.value_info if v.name not in removed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    definitions = function_definitions(imports[""])
    model.functions.extend(definitions[k] for k, count in counts.items() if count)
    return model, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-pre", type=int)
    parser.add_argument("--expect-post", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = onnx.load(args.source, load_external_data=False)
    result, counts = fuse(source)
    for key, expected in ((PRE, args.expect_pre), (POST, args.expect_post)):
        if expected is not None and counts[key] != expected:
            raise ValueError(f"Expected {expected} {key} chains, got {counts[key]}")
    if [v.SerializeToString() for v in source.graph.initializer] != [
        v.SerializeToString() for v in result.graph.initializer
    ]:
        raise ValueError("Initializer descriptors changed")
    locations = {
        item.value for v in result.graph.initializer for item in v.external_data if item.key == "location"
    }
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unconfined external location: {location}")
        if not (args.source.parent / relative).is_file():
            raise FileNotFoundError(args.source.parent / relative)
        if (args.output.parent / relative).exists():
            if not os.path.samefile(args.source.parent / relative, args.output.parent / relative):
                raise ValueError(f"External data collision: {relative}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for location in locations:
        target = args.output.parent / location
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(args.source.parent / location, target)
    onnx.save_model(result, args.output)
    onnx.checker.check_model(str(args.output))
    config = args.source.parent / "genai_config.json"
    if config.is_file():
        target = args.output.parent / config.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(config, target)
    print(json.dumps({
        "status": "graph_rewritten_not_GPU_qualified",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "counts": counts,
        "removed_dispatches_if_claimed": counts[PRE] + 3 * counts[POST],
        "external_weights": "unchanged descriptors and same-inode hardlinks",
        "quality_qualified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
