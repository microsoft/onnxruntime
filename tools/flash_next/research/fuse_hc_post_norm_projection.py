import onnx
from onnx import AttributeProto, TensorProto, helper


DOMAIN = "com.microsoft"
OPS = {"HCPostNormF16": "HCPostNormProjectionF16",
       "HCSharedPostNormF16": "HCSharedPostNormProjectionF16"}
PROJECTION_ATTRS = {"K", "N", "block_size", "bits", "accuracy_level", "alpha"}


def definition(original_op, opset):
    shared = original_op == "HCSharedPostNormF16"
    norm_inputs = ["streams", "branch", "gate", "norm_scale"]
    if shared:
        norm_inputs += ["shared", "shared_gate"]
    norm = helper.make_node(original_op, norm_inputs, ["y", "normalized"], domain=DOMAIN)
    norm.attribute.add(name="epsilon", ref_attr_name="epsilon", type=AttributeProto.FLOAT)
    projection = helper.make_node("HCProjectionSiLUF16",
        ["normalized", "sizes", "weights", "scales"], ["activated", "tail"], domain=DOMAIN)
    for name in sorted(PROJECTION_ATTRS):
        projection.attribute.add(name=name, ref_attr_name=name,
            type=AttributeProto.FLOAT if name == "alpha" else AttributeProto.INT)
    return helper.make_function(DOMAIN, OPS[original_op],
        norm_inputs + ["sizes", "weights", "scales"], ["y", "normalized", "activated", "tail"],
        [norm, projection], [helper.make_opsetid("", opset), helper.make_opsetid(DOMAIN, 1)],
        attributes=["epsilon", *sorted(PROJECTION_ATTRS)])


def fuse(original):
    model = onnx.ModelProto()
    model.CopyFrom(original)
    if any(f.domain == DOMAIN and f.name in OPS.values() for f in model.functions):
        raise ValueError("Already fused post-normalization/projection functions")
    imports = {op.domain: op.version for op in model.opset_import}
    if imports.get(DOMAIN) != 1 or imports.get("", 0) < 13:
        raise ValueError("Requires ONNX opset>=13 and com.microsoft opset1")
    nodes = list(model.graph.node)
    producers = {output: index for index, node in enumerate(nodes) for output in node.output}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    infos = {value.name: value.type.tensor_type.elem_type
             for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)}

    def element_type(name):
        return initializers[name].data_type if name in initializers else infos.get(name)

    replacements, removed, used = {}, set(), set()
    for index, projection in enumerate(nodes):
        if (projection.domain != DOMAIN or projection.op_type != "HCProjectionSiLUF16" or
                len(projection.input) != 4 or len(projection.output) != 2):
            continue
        parent_index = producers.get(projection.input[0])
        if parent_index is None or parent_index in replacements:
            continue
        parent = nodes[parent_index]
        if (parent.domain != DOMAIN or parent.op_type not in OPS or len(parent.output) != 2 or
                parent.output[1] != projection.input[0] or
                len(parent.input) != (6 if parent.op_type == "HCSharedPostNormF16" else 4)):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in projection.attribute}
        norm_attrs = {a.name: helper.get_attribute_value(a) for a in parent.attribute}
        if (set(attrs) - PROJECTION_ATTRS or set(norm_attrs) - {"epsilon"} or
                any(attrs.get(key) != value for key, value in
                    (("K", 10240), ("N", 324), ("block_size", 32), ("bits", 4)))):
            continue
        if any(element_type(name) != TensorProto.FLOAT16 for name in
               (*parent.input, *parent.output, *projection.output, projection.input[3])):
            continue
        if (element_type(projection.input[1]) != TensorProto.INT64 or
                element_type(projection.input[2]) != TensorProto.UINT8):
            continue
        if any(name in producers and producers[name] >= parent_index
               for name in projection.input[1:]):
            continue
        attrs.setdefault("accuracy_level", 4)
        attrs.setdefault("alpha", 1.0)
        attrs["epsilon"] = norm_attrs.get("epsilon", 1e-6)
        replacements[parent_index] = helper.make_node(OPS[parent.op_type],
            [*parent.input, *projection.input[1:]], [*parent.output, *projection.output],
            name=parent.name + "/Projection", domain=DOMAIN, **attrs)
        removed.add(index)
        used.add(parent.op_type)
    del model.graph.node[:]
    model.graph.node.extend(replacements.get(index, node)
                            for index, node in enumerate(nodes) if index not in removed)
    for op in sorted(used):
        model.functions.append(definition(op, imports[""]))
    return model, len(replacements)
