import onnx
from onnx import TensorProto, helper


DOMAIN = "com.microsoft"
OP = "MlpInputProjectionsF16"
WIDTHS = (512, 640, 640, 1)


def definition(opset):
    names = ["x"]
    nodes = []
    for role, width in zip(("router", "gate", "up", "shared"), WIDTHS):
        names.extend((role + "_weight", role + "_scale"))
        nodes.append(helper.make_node("MatMulNBits",
            ["x", role + "_weight", role + "_scale"], [role], domain=DOMAIN,
            K=2560, N=width, bits=4, block_size=32, accuracy_level=4))
    nodes.append(helper.make_node("QuickGeluMulF16", ["gate", "up"], ["activated"], domain=DOMAIN))
    return helper.make_function(DOMAIN, OP, names, ["router", "activated", "shared"], nodes,
        [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)])


def fuse(original):
    model = onnx.ModelProto()
    model.CopyFrom(original)
    if any(f.domain == DOMAIN and f.name == OP for f in model.functions):
        raise ValueError("Already fused MLP input projections")
    imports = {o.domain: o.version for o in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires ONNX opset>=13 and com.microsoft opset1")
    nodes = list(model.graph.node)
    producers = {o: i for i, n in enumerate(nodes) for o in n.output}
    initializers = {t.name: t for t in model.graph.initializer}
    types = {v.name: v.type.tensor_type.elem_type
             for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    graph_outputs = {v.name for v in model.graph.output}
    consumers = {}
    for i, n in enumerate(nodes):
        for value in n.input:
            consumers.setdefault(value, []).append(i)

    def projection(i, width):
        n = nodes[i]
        if (n.domain != DOMAIN or n.op_type != "MatMulNBits" or
                len(n.input) != 3 or len(n.output) != 1):
            return False
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if attrs != dict(K=2560, N=width, bits=4, block_size=32, accuracy_level=4):
            return False
        weight, scale = (initializers.get(name) for name in n.input[1:])
        return (types.get(n.input[0]) == types.get(n.output[0]) == TensorProto.FLOAT16 and
                weight is not None and weight.data_type == TensorProto.UINT8 and
                list(weight.dims) == [width, 80, 16] and
                scale is not None and scale.data_type == TensorProto.FLOAT16 and
                list(scale.dims) == [width, 80])

    replacements, removed = {}, set()
    for i, activation in enumerate(nodes):
        if (activation.domain != DOMAIN or activation.op_type != "QuickGeluMulF16" or
                len(activation.input) != 2 or len(activation.output) != 1 or activation.attribute or
                types.get(activation.output[0]) != TensorProto.FLOAT16):
            continue
        pair = [producers.get(name) for name in activation.input]
        if any(index is None or not projection(index, 640) for index in pair):
            continue
        gate, up = (nodes[index] for index in pair)
        if (gate.input[0] != up.input[0] or
                any(consumers[name] != [i] or name in graph_outputs for name in activation.input)):
            continue
        choices = [[j for j, n in enumerate(nodes)
                    if n.input and n.input[0] == gate.input[0] and projection(j, width)]
                   for width in (512, 1)]
        if any(len(choice) != 1 for choice in choices):
            continue
        indices = [choices[0][0], *pair, choices[1][0]]
        first = min(indices)
        if (len(set(indices)) != 4 or any(j in removed or j in replacements for j in indices) or
                (gate.input[0] in producers and producers[gate.input[0]] >= first)):
            continue
        projections = [nodes[j] for j in indices]
        replacements[first] = helper.make_node(OP,
            [gate.input[0], *[name for p in projections for name in p.input[1:]]],
            [projections[0].output[0], activation.output[0], projections[3].output[0]],
            name=activation.name + "/InputProjections", domain=DOMAIN)
        removed.update([*indices, i])
        removed.remove(first)
    del model.graph.node[:]
    model.graph.node.extend(replacements.get(i, n) for i, n in enumerate(nodes) if i not in removed)
    if replacements:
        model.functions.append(definition(imports[""]))
    return model, len(replacements)
