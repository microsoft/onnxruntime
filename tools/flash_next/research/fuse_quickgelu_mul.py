"""Opt-in fusion matching ORT's existing production SiLU -> QuickGelu lowering."""
from collections import Counter
from copy import deepcopy

from onnx import TensorProto, helper

DOMAIN = "com.microsoft"
OP = "QuickGeluMulF16"


def definition(opset):
    return helper.make_function(DOMAIN, OP, ["X", "Up"], ["Y"], [
        helper.make_node("QuickGelu", ["X"], ["Activated"], domain=DOMAIN, alpha=1.0),
        helper.make_node("Mul", ["Activated", "Up"], ["Y"]),
    ], [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)])


def fuse(original):
    model = deepcopy(original)
    if any(f.domain == DOMAIN and f.name == OP for f in model.functions):
        raise ValueError("Already fused")
    imports = {v.domain: v.version for v in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires ONNX opset>=13 and com.microsoft opset1")
    nodes = list(model.graph.node)
    producers = {v: n for n in nodes for v in n.output}
    uses = Counter(v for n in nodes for v in n.input)
    exposed = {v.name for v in model.graph.output}
    infos = {v.name: v.type.tensor_type for v in [
        *model.graph.input, *model.graph.output, *model.graph.value_info]}
    removed, replacements = set(), {}
    for node in nodes:
        if node.domain or node.op_type != "Mul" or node.attribute or len(node.input) != 2:
            continue
        for activated_index in (0, 1):
            activation = producers.get(node.input[activated_index])
            if (activation is None or activation.domain or activation.op_type != "Mul" or
                    activation.attribute or len(activation.input) != 2):
                continue
            for sigmoid_index in (0, 1):
                sigmoid = producers.get(activation.input[sigmoid_index])
                x = activation.input[1 - sigmoid_index]
                if (sigmoid is None or sigmoid.domain or sigmoid.op_type != "Sigmoid" or
                        sigmoid.attribute or list(sigmoid.input) != [x]):
                    continue
                intermediates = [sigmoid.output[0], activation.output[0]]
                if any(uses[v] != 1 or v in exposed for v in intermediates):
                    continue
                up = node.input[1 - activated_index]
                values = [x, up, *node.output, *intermediates]
                if any(v not in infos or infos[v].elem_type != TensorProto.FLOAT16 for v in values):
                    continue
                if any(infos[v].shape.SerializeToString() != infos[x].shape.SerializeToString()
                       for v in values):
                    continue
                replacements[node.output[0]] = helper.make_node(
                    OP, [x, up], list(node.output), name=node.name, domain=DOMAIN)
                removed.update(intermediates)
    if not replacements:
        raise ValueError("No matching single-consumer same-shape FP16 SiLU/Mul chains")
    del model.graph.node[:]
    model.graph.node.extend(replacements.get(n.output[0], n) for n in nodes
                            if not any(v in removed for v in n.output))
    retained = [v for v in model.graph.value_info if v.name not in removed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    model.functions.append(definition(imports[""]))
    return model, len(replacements)
