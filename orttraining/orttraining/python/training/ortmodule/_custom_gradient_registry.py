# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Support registration of ATen op's symbolic (PyTorch exporter overriding) and gradient definition.

# Each gradient definition of ATen op is a list of node definitions: [node_def1, node_def2, ...].
#     Each node definition is a tuple: (op_type, inputs, outputs, attributes), while attributes is optional.
#         'op_type' is a string or a tuple of two strings with op_name and domain respectively.
#             If it's string type, then the domain is the default ONNX domain: ''.
#         'inputs' is a list of strings. Each input string can be GO(i), I(i), O(i),
#             which means ith gradient output, ith input, ith output of forward node respectively,
#             or any string for intermediate output of one of other nodes.
#         'outputs' is a list of strings. Each input string can be GI(i),
#             which means ith gradient input of forward node,
#             or any string for intermediate output for some of other nodes.
#         'attributes' (if present) is a dictionary. Each entry's key is the attribute name,
#             the value is also a dictionary: {'value': v, 'dtype': t, 'is_tensor': b}.
#                 'v' can be string, number, or list of numbers, which is the value of the attribute.
#                 't' is a string to describe the element type. It can be 'int', 'float', 'bool', etc,
#                     or 'IElemType(i)', 'OElemType(i)', which means the same type of ith input or
#                     output of forward node.
#                 'b' is True or False, indicating if this attribute is tensor_proto ot not.
#                     'is_tensor' is optional, if not present, the default is False.

import json

from onnxruntime.capi import _pybind_state as C


def _to_gradient_definition(gradient):
    node_defs = []
    for node in gradient:
        node_def = C.GradientNodeDefinition()
        if isinstance(node[0], str):
            node_def.op_type = node[0]
            node_def.domain = ""
        else:
            node_def.op_type = node[0][0]
            node_def.domain = node[0][1]
        node_def.inputs = node[1]
        node_def.outputs = node[2]
        attributes = []
        if len(node) >= 4:
            for key, value in node[3].items():
                attr_def = C.GradientNodeAttributeDefinition()
                attr_def.name = key
                attr_def.value_json = json.dumps(value["value"])
                attr_def.dtype = value["dtype"]
                attr_def.is_tensor = value["is_tensor"] if "is_tensor" in value else False
                attributes.append(attr_def)
        node_def.attributes = attributes
        node_defs.append(node_def)
    return node_defs


class CustomGradientRegistry:
    _GRADIENTS = {}  # noqa: RUF012
    _STOP_GRADIENT_EDGES = {}  # noqa: RUF012

    @classmethod
    def register(cls, domain, name, attributes, fn):
        key = "::".join([domain, name, *list(attributes)])
        cls._GRADIENTS[key] = _to_gradient_definition(fn())

    @classmethod
    def register_custom_stop_gradient_edges(cls, edges, domain, name, *attributes):
        key = "::".join([domain, name, *list(attributes)])
        cls._STOP_GRADIENT_EDGES[key] = set(edges)

    @classmethod
    def register_all(cls):
        for key, value in cls._GRADIENTS.items():
            C.register_gradient_definition(key, value)
        for key, value in cls._STOP_GRADIENT_EDGES.items():
            C.register_custom_stop_gradient_edges(key, value)


def register_gradient(domain, name, *attributes):
    def gradient_wrapper(fn):
        CustomGradientRegistry.register(domain, name, attributes, fn)
        return fn

    return gradient_wrapper


# For ATen op, we need to provide op_name and overload name.
@register_gradient("org.pytorch.aten", "ATen", "embedding", "")
def embedding_gradient():
    return [
        ("Constant", [], ["Const_0"], {"value": {"value": 0, "dtype": "int", "is_tensor": True}}),
        ("Shape", ["I(0)"], ["Shape_X"]),
        ("Gather", ["Shape_X", "Const_0"], ["Gather_X_0"], {"axis": {"value": 0, "dtype": "int"}}),
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "I(1)", "Gather_X_0", "I(2)", "I(3)", "I(4)"],
            ["GI(0)"],
            {"operator": {"value": "embedding_backward", "dtype": "string"}},
        ),
    ]


@register_gradient("org.pytorch.aten", "ATen", "diagonal", "")
def diagonal_gradient():
    return [
        ("Shape", ["I(0)"], ["Shape_X"]),
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "Shape_X", "I(1)", "I(2)", "I(3)"],
            ["GI(0)"],
            {"operator": {"value": "diagonal_backward", "dtype": "string"}},
        ),
    ]


@register_gradient("org.pytorch.aten", "ATen", "max_pool2d_with_indices", "")
def max_pool2d_gradient():
    return [
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "I(0)", "I(1)", "I(2)", "I(3)", "I(4)", "I(5)", "O(1)"],
            ["GI(0)"],
            {"operator": {"value": "max_pool2d_with_indices_backward", "dtype": "string"}},
        ),
    ]


def minmax_gradient():
    # Gradient of torch.min(input) (and max)
    # In PyTorch, when there are multiple maxima/minima, the gradient is evenly distributed among them.
    # e.g.: x = torch.tensor([1., 2., 2., 1., 2.], requires_grad=True)
    #       y = x.max()       (tensor(2., grad_fn=<MaxBackward1>))
    #       y.backward()
    #       print(x.grad)     (tensor([0.0000, 0.3333, 0.3333, 0.0000, 0.3333]))
    return [
        ("Equal", ["I(0)", "O(0)"], ["Mask"]),
        ("Constant", [], ["Const_0"], {"value": {"value": 0.0, "dtype": "IElemType(0)", "is_tensor": True}}),
        ("Constant", [], ["Const_1"], {"value": {"value": 1.0, "dtype": "IElemType(0)", "is_tensor": True}}),
        ("Where", ["Mask", "Const_1", "Const_0"], ["MaskValue"]),
        ("ReduceSum", ["MaskValue"], ["MaskSum"], {"keepdims": {"value": 0, "dtype": "int"}}),
        ("Div", ["GO(0)", "MaskSum"], ["DistributedGrad"]),
        ("Mul", ["MaskValue", "DistributedGrad"], ["GI(0)"]),
    ]


min_gradient = register_gradient("org.pytorch.aten", "ATen", "min", "")(minmax_gradient)
max_gradient = register_gradient("org.pytorch.aten", "ATen", "max", "")(minmax_gradient)


def minmax_dim_gradient():
    # Gradient of torch.min(input, dim, keepdim) (and max)
    # In PyTorch, when there are multiple maxima/minima along the axis, the gradient is granted to the first.
    # e.g.: x = torch.tensor([1., 2., 2., 1., 2.], requires_grad=True)
    #       y = x.max(dim=0)     (torch.return_types.max(values=tensor(2., grad_fn=<MaxBackward0>),
    #                                                    indices=tensor(1)))
    #       y.values.backward()
    #       print(x.grad)        (tensor([0., 1., 0., 0., 0.]))
    return [
        ("Shape", ["I(0)"], ["Shape_X"]),
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "I(1)", "O(1)", "Shape_X", "I(2)"],
            ["GI(0)"],
            {"operator": {"value": "value_selecting_reduction_backward", "dtype": "string"}},
        ),
    ]


min_dim_gradient = register_gradient("org.pytorch.aten", "ATen", "min", "dim")(minmax_dim_gradient)
max_dim_gradient = register_gradient("org.pytorch.aten", "ATen", "max", "dim")(minmax_dim_gradient)


@register_gradient("org.pytorch.aten", "ATen", "unfold", "")
def unfold_gradient():
    return [
        ("Shape", ["I(0)"], ["Shape_X"]),
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "Shape_X", "I(1)", "I(2)", "I(3)"],
            ["GI(0)"],
            {"operator": {"value": "unfold_backward", "dtype": "string"}},
        ),
    ]


@register_gradient("org.pytorch.aten", "ATen", "avg_pool2d", "")
def avg_pool2d_gradient():
    return [
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "I(0)", "I(1)", "I(2)", "I(3)", "I(4)", "I(5)", "I(6)"],
            ["GI(0)"],
            {"operator": {"value": "avg_pool2d_backward", "dtype": "string"}},
        ),
    ]


@register_gradient("org.pytorch.aten", "ATen", "_adaptive_avg_pool2d", "")
def adaptive_avg_pool2d_gradient():
    return [
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "I(0)"],
            ["GI(0)"],
            {"operator": {"value": "_adaptive_avg_pool2d_backward", "dtype": "string"}},
        ),
    ]


CustomGradientRegistry.register_custom_stop_gradient_edges([0], "org.pytorch.aten", "ATen", "argmax", "")
CustomGradientRegistry.register_custom_stop_gradient_edges([0], "org.pytorch.aten", "ATen", "multinomial", "")


@register_gradient("org.pytorch.aten", "ATen", "numpy_T", "")
def numpy_T_gradient():  # noqa: N802
    return [
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)"],
            ["GI(0)"],
            {"operator": {"value": "numpy_T", "dtype": "string"}},
        ),
    ]


@register_gradient("org.pytorch.aten", "ATen", "native_group_norm", "")
def native_group_norm_gradient():
    return [
        ("Constant", [], ["Const_0"], {"value": {"value": [True, True, True], "dtype": "bool", "is_tensor": True}}),
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "I(0)", "O(1)", "O(2)", "I(1)", "I(3)", "I(4)", "I(5)", "I(6)", "Const_0"],
            ["GI(0)", "GI(1)", "GI(2)"],
            {"operator": {"value": "native_group_norm_backward", "dtype": "string"}},
        ),
    ]


# PyTorch removed related backward functions with "vec" overload name since 1.13. The functions with no overload name
# are available for all versions, though they are not that convienent to use.
def _upsample_gradient(backward_fn, dims):
    scales = ["" for _ in range(dims)]
    if "bilinear" in backward_fn:
        scales = ["I(2)", *scales]
    return [
        ("Shape", ["I(0)"], ["Shape_X"]),
        ("Shape", ["O(0)"], ["Shape_Y"]),
        ("Constant", [], ["Const_Start"], {"value": {"value": [2], "dtype": "int", "is_tensor": True}}),
        ("Constant", [], ["Const_End"], {"value": {"value": [2 + dims], "dtype": "int", "is_tensor": True}}),
        ("Slice", ["Shape_Y", "Const_Start", "Const_End"], ["Sliced_Shape_Y"]),
        (
            ("ATen", "org.pytorch.aten"),
            ["GO(0)", "Sliced_Shape_Y", "Shape_X", *scales],
            ["GI(0)"],
            {"operator": {"value": backward_fn, "dtype": "string"}},
        ),
    ]


@register_gradient("org.pytorch.aten", "ATen", "upsample_nearest1d", "vec")
def upsample_nearest1d_gradient():
    return _upsample_gradient("upsample_nearest1d_backward", 1)


@register_gradient("org.pytorch.aten", "ATen", "upsample_nearest2d", "vec")
def upsample_nearest2d_gradient():
    return _upsample_gradient("upsample_nearest2d_backward", 2)


@register_gradient("org.pytorch.aten", "ATen", "upsample_nearest3d", "vec")
def upsample_nearest3d_gradient():
    return _upsample_gradient("upsample_nearest3d_backward", 3)
