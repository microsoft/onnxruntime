from copy import deepcopy

import numpy as np
from onnx import AttributeProto, TensorProto, helper, numpy_helper

DOMAIN = "com.microsoft"
OPS = {"HyperConnectionPostMixGateF16": "HCPostNormF16",
       "HyperConnectionSharedPostMixF16": "HCSharedPostNormF16"}


def definition(original_op, opset):
    shared = original_op == "HyperConnectionSharedPostMixF16"
    inputs = ["Streams", "Branch", "Gate", "Norm"]
    if shared:
        inputs += ["Shared", "SharedGate"]
    post_inputs = ["Streams", "Branch", "Gate", "Divisor", "Multiplier"]
    if shared:
        post_inputs += ["Shared", "SharedGate"]
    norm = helper.make_node("BranchwiseRMSNorm", ["Y", "Norm"], ["Normalized"],
                            domain=DOMAIN, num_branches=4)
    norm.attribute.append(helper.make_attribute_ref("epsilon", AttributeProto.FLOAT))
    return helper.make_function(DOMAIN, OPS[original_op], inputs, ["Y", "Normalized"], [
        helper.make_node("Constant", [], ["Divisor"],
                         value=numpy_helper.from_array(np.array(4, dtype=np.float16))),
        helper.make_node("Constant", [], ["Multiplier"],
                         value=numpy_helper.from_array(np.array(2, dtype=np.float16))),
        helper.make_node(original_op, post_inputs, ["Y"], domain=DOMAIN, num_branches=4),
        norm,
    ], [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)], attributes=["epsilon"])


def fuse(original):
    model = deepcopy(original)
    if any(f.domain == DOMAIN and f.name in OPS.values() for f in model.functions):
        raise ValueError("Already fused")
    imports = {op.domain: op.version for op in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires ONNX opset>=13 and com.microsoft opset1")
    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    infos = {value.name: value.type.tensor_type for value in [
        *model.graph.input, *model.graph.output, *model.graph.value_info]}
    removed, replacements, used = set(), {}, set()
    for norm in nodes:
        if norm.domain != DOMAIN or norm.op_type != "BranchwiseRMSNorm" or len(norm.input) != 2:
            continue
        post = producers.get(norm.input[0])
        if post is None or post.domain != DOMAIN or post.op_type not in OPS or post.output[0] in replacements:
            continue
        norm_attrs = {a.name: helper.get_attribute_value(a) for a in norm.attribute}
        post_attrs = {a.name: helper.get_attribute_value(a) for a in post.attribute}
        if (set(norm_attrs) - {"num_branches", "epsilon"} or
                post_attrs != {"num_branches": 4} or norm_attrs.get("num_branches") != 4):
            continue
        if len(post.input) != (7 if post.op_type == "HyperConnectionSharedPostMixF16" else 5):
            continue
        constants = []
        for name in post.input[3:5]:
            tensor = tensors.get(name)
            producer = producers.get(name)
            if tensor is None and producer is not None and not producer.domain and producer.op_type == "Constant":
                tensor = next((a.t for a in producer.attribute if a.name == "value"), None)
            constants.append(tensor)
        if any(t is None or t.data_type != TensorProto.FLOAT16 or t.external_data for t in constants):
            continue
        arrays = [numpy_helper.to_array(t) for t in constants]
        if any(a.size != 1 or float(a.reshape(-1)[0]) != v for a, v in zip(arrays, (4, 2))):
            continue
        inputs = list(post.input[:3]) + [norm.input[1]] + list(post.input[5:])
        if any(name not in tensors and (name not in infos or infos[name].elem_type != TensorProto.FLOAT16)
               or name in tensors and tensors[name].data_type != TensorProto.FLOAT16
               for name in [*inputs, *post.output, *norm.output]):
            continue
        replacement = helper.make_node(
            OPS[post.op_type], inputs, [post.output[0], norm.output[0]], domain=DOMAIN,
            name=post.name + "_norm", epsilon=norm_attrs.get("epsilon", 1e-5))
        replacements[post.output[0]] = replacement
        removed.add(norm.output[0])
        used.add(post.op_type)
    if not replacements:
        raise ValueError("No compatible FP16 post-mix/normalization chains")
    del model.graph.node[:]
    model.graph.node.extend(replacements.get(n.output[0], n) for n in nodes
                            if n.output[0] not in removed)
    for op in sorted(used):
        model.functions.append(definition(op, imports[""]))
    return model, len(replacements)
