# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Support registration of ATenOp's symbolic (PyTorch exporter overriding) and gradient definition.

# Each gradient definition of ATenOp is a list of node definitions: [node_def1, node_def2, ...].
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
            node_def.domain = ''
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
                attr_def.value_json = json.dumps(value['value'])
                attr_def.dtype = value['dtype']
                attr_def.is_tensor = value['is_tensor'] if 'is_tensor' in value else False
                attributes.append(attr_def)
        node_def.attributes = attributes
        node_defs.append(node_def)
    return node_defs


class CustomGradientRegistry:
    _GRADIENTS = {}

    @classmethod
    def register(cls, domain, name, attributes, fn):
        key = '::'.join([domain, name] + list(attributes))
        cls._GRADIENTS[key] = _to_gradient_definition(fn())

    @classmethod
    def register_all(cls):
        for key, value in cls._GRADIENTS.items():
            C.register_gradient_definition(key, value)


def register_gradient(domain, name, *attributes):
    def gradient_wrapper(fn):
        CustomGradientRegistry.register(domain, name, attributes, fn)
        return fn
    return gradient_wrapper


# For ATenOp, need to provide op_name and overload name. For example:
#
# @register_gradient('com.microsoft', 'ATenOp', 'aten::unfold', '')
# def unfold_gradient():
#     return [
#         ('Shape', ['I(0)'], ['Shape_X']),
#         (('ATenOp', 'com.microsoft'), ['GO(0)', 'Shape_X', 'I(1)', 'I(2)', 'I(3)'], [
#          'GI(0)'], {'name': {'value': 'aten::unfold_backward', 'dtype': 'string'}}),
#     ]
