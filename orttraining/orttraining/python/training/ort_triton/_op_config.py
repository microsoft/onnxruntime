# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
This file controls the Ops supported by Triton codegen.
The Triton fusion on backend will try to extract subgraphs from the ONNX model which contain connected supported ops.
For each supported op, it has the following attributes:
  - domain: The domain of the op. If not specified, the op is from ONNX domain, which is empty string.
  - versions: The supported opset versions.
  - is_no_op: If the op is no-op. If not specified, the op is not no-op.
  - conditions: define some conditions to control the fusion. For example, for Ops that has axis attribute,
                a condition "axis": "-1" means only Ops that reduce on last dimension will be fused.
                For Reduce* Ops, the "axes" condition can be list of ints, such as "axes": "[-1]",
                or "axes": "single", means only support Ops reduce on single constant dimension,
                or "axes": "constant", means the axes attribute or input must be constant.
  - ignore_min_nodes: by default is False. If set to True, the fusion will ignore the min_nodes check for this Op.
                      For example, if the min_nodes is 2, the fusion will only fuse the subgraphs with 2 or more
                      non-no-op nodes. But if the ignore_min_nodes is True for ReduceSum, it's OK to fuse a single
                      ReduceSum node to the subgraph.
"""

from onnx import NodeProto

_ELEMENTWISE_OPS = {
    "Add": {"versions": [13, 14]},
    "Sub": {"versions": [13, 14]},
    "Mul": {"versions": [13, 14]},
    "Div": {"versions": [13, 14]},
    "Pow": {"versions": [13, 15]},
    "Sqrt": {"versions": [13]},
    "Exp": {"versions": [13]},
    "Where": {"versions": [9, 16]},
    "Cast": {"versions": [13]},
    "Dropout": {"versions": [13]},
    "DropoutGrad": {"domain": "com.microsoft", "versions": [1]},
    "Identity": {"versions": [13], "is_no_op": True},
    "Sum": {"versions": [13]},
    "Gelu": {"domain": "com.microsoft", "versions": [1]},
    "QuickGelu": {"domain": "com.microsoft", "versions": [1]},
    "GeluGrad": {"domain": "com.microsoft", "versions": [1]},
    "QuickGeluGrad": {"domain": "com.microsoft", "versions": [1]},
}

_REDUCTION_OPS = {
    "ReduceMean": {"versions": [13], "conditions": {"axes": "[-1]"}},
    "ReduceSum": {"versions": [13], "conditions": {"axes": "[-1]"}},
    "ReduceMax": {"versions": [13], "conditions": {"axes": "[-1]"}},
    "ReduceMin": {"versions": [13], "conditions": {"axes": "[-1]"}},
    "Softmax": {"versions": [13]},
    "SoftmaxGrad_13": {"domain": "com.microsoft", "versions": [1]},
    # Disable LayerNormalization fusion for now as it's generated Triton code is inefficient compared to C++ kernel.
    # "LayerNormalization": {"versions": [1]},
    # "LayerNormalizationGrad": {"domain": "com.microsoft", "versions": [1]},
}


def is_elementwise_node(node: NodeProto) -> bool:
    return node.op_type in _ELEMENTWISE_OPS


def is_reduction_node(node: NodeProto) -> bool:
    return node.op_type in _REDUCTION_OPS


def get_supported_ops():
    return {**_ELEMENTWISE_OPS, **_REDUCTION_OPS}
