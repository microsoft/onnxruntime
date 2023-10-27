# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
PyTorch's _efficient_attention_forward/_efficient_attention_backward APIs is keep changing. Current implementation
is tested well on version 2.2.0.dev20231010+cu121, and should be run well since official version 2.2.0. If may fail to
run is you are using PyTorch with older versions.

PyTorch also has API for flash attention (currently doesn't support random attention mask or Dropout), we can add
support if we want to try in the future.
"""

from typing import List, Tuple

from onnx import GraphProto, NodeProto, TensorProto, helper

from ..graph_optimizer_registry import register_graph_optimizer
from .utils import GraphMatcher, check_attribute_value, make_constant_node, update_graph


def _make_efficient_attention_nodes(
    idx: int,
    q: str,
    k: str,
    v: str,
    y: str,
    dy: str,
    dq: str,
    dk: str,
    dv: str,
    bias: str,
    expand_bias: bool,
    scale: float,
    dropout_ratio: float,
    causal: bool,
):
    nodes_to_add = []
    scale_node = make_constant_node("scale_" + str(idx), TensorProto.FLOAT, [], [scale])
    dropout_ratio_node = make_constant_node("dropout_ratio_" + str(idx), TensorProto.FLOAT, [], [dropout_ratio])
    causal_node = make_constant_node("causal_" + str(idx), TensorProto.INT64, [], [1 if causal else 0])
    int_zero_node = make_constant_node("int_zero_" + str(idx), TensorProto.INT64, [], [0])
    true_node = make_constant_node("true_" + str(idx), TensorProto.BOOL, [], [True])
    false_node = make_constant_node("false_" + str(idx), TensorProto.BOOL, [], [False])
    logsumexp = helper.make_tensor_value_info("logsumexp" + str(idx), TensorProto.FLOAT, [])
    seed = helper.make_tensor_value_info("seed" + str(idx), TensorProto.INT64, [])
    offset = helper.make_tensor_value_info("offset" + str(idx), TensorProto.INT64, [])
    new_value_infos = [logsumexp, seed, offset]
    if expand_bias:
        shape_0 = helper.make_node("Shape", [q], ["shape_0_" + str(idx)], start=0, end=1)
        shape_1 = helper.make_node("Shape", [q], ["shape_1_" + str(idx)], start=2, end=3)
        shape_2 = helper.make_node("Shape", [q], ["shape_2_" + str(idx)], start=1, end=2)
        shape_3 = helper.make_node("Shape", [k], ["shape_3_" + str(idx)], start=1, end=2)
        concat = helper.make_node(
            "Concat",
            ["shape_0_" + str(idx), "shape_1_" + str(idx), "shape_2_" + str(idx), "shape_3_" + str(idx)],
            ["concated_shape_" + str(idx)],
            axis=0,
        )
        expand = helper.make_node("Expand", [bias, "concated_shape_" + str(idx)], ["expanded_bias_" + str(idx)])
        nodes_to_add.extend([shape_0, shape_1, shape_2, shape_3, concat, expand])
        bias = "expanded_bias_" + str(idx)
    fwd_node = helper.make_node(
        "ATen",
        [
            q,
            k,
            v,
            bias,
            "",
            "",
            "",
            dropout_ratio_node.output[0],
            causal_node.output[0],
            true_node.output[0],
            scale_node.output[0],
            "",
            "",
        ],
        [y, logsumexp.name, seed.name, offset.name],
        "efficient_attention_forward_" + str(idx),
        None,
        "org.pytorch.aten",
        operator="_efficient_attention_forward",
    )
    bwd_node = helper.make_node(
        "ATen",
        [
            dy,
            q,
            k,
            v,
            bias,
            y,
            "",
            "",
            int_zero_node.output[0],
            int_zero_node.output[0],
            logsumexp.name,
            dropout_ratio_node.output[0],
            seed.name,
            offset.name,
            causal_node.output[0],
            false_node.output[0],
            scale_node.output[0],
            "",
        ],
        [dq, dk, dv, ""],
        "efficient_attention_backward_" + str(idx),
        None,
        "org.pytorch.aten",
        operator="_efficient_attention_backward",
    )
    nodes_to_add.extend(
        [scale_node, dropout_ratio_node, causal_node, int_zero_node, true_node, false_node, fwd_node, bwd_node]
    )
    return nodes_to_add, new_value_infos


# Without causal mask, with Dropout. For example, BERT model in HuggingFace.
_PATTERN_0: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Transpose", True, [(0, 0, 1)]),  # 2
    ("Div", False, [(0, 0, 0)]),  # 3
    ("Add", False, [(3, 0, 0)]),  # 4
    ("Softmax", False, [(4, 0, 0)]),  # 5
    ("Dropout", False, [(5, 0, 0)]),  # 6
    ("MatMul", False, [(6, 0, 0)]),  # 7
    ("Transpose", True, [(7, 0, 1)]),  # 8
    ("Transpose", False, [(7, 0, 0)]),  # 9
    ("FusedMatMul", False, [(8, 0, 1)]),  # 10
    ("DropoutGrad", False, [(10, 0, 0), (6, 1, 1)]),  # 11
    ("SoftmaxGrad_13", False, [(11, 0, 0), (5, 0, 1)]),  # 12
    ("Identity", False, [(12, 0, 0)]),  # 13
    ("Div", False, [(13, 0, 0)]),  # 14
    ("Identity", False, [(14, 0, 0)]),  # 15
    ("FusedMatMul", False, [(2, 0, 1), (15, 0, 0)]),  # 16
    ("FusedMatMul", False, [(1, 0, 0), (15, 0, 1)]),  # 17
    ("FusedMatMul", False, [(6, 0, 0)]),  # 18
    ("Transpose", True, [(18, 0, 1)]),  # 19
    ("Transpose", False, [(16, 0, 0)]),  # 20
    ("Transpose", False, [(17, 0, 0)]),  # 21
    ("Transpose", False, [(18, 0, 0)]),  # 22
]


def _optimize_for_pattern_0(matcher: GraphMatcher, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[3].input[1])
    ratio_value = matcher.get_constant_value(nodes[6].input[1])
    if not (
        check_attribute_value(nodes[1], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[2], "perm", [0, 2, 3, 1])
        and scale_value is not None
        and ratio_value is not None
        and check_attribute_value(nodes[8], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[9], "perm", [0, 2, 1, 3])
    ):
        return [], [], []

    _, add_input_shape_0 = matcher.get_type_and_shape(nodes[4].input[0])
    _, add_input_shape_1 = matcher.get_type_and_shape(nodes[4].input[1])
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[1].input[0],
        nodes[2].input[0],
        nodes[8].input[0],
        nodes[9].output[0],
        nodes[19].input[0],
        nodes[20].output[0],
        nodes[21].output[0],
        nodes[22].output[0],
        nodes[4].input[1],
        add_input_shape_0 != add_input_shape_1,
        1 / float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        ratio_value,
        False,
    )
    return nodes, nodes_to_add, new_value_infos


# Without causal mask, without Dropout. For example, BERT model and disabling attention dropout in HuggingFace.
_PATTERN_1: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Transpose", True, [(0, 0, 1)]),  # 2
    ("Div", False, [(0, 0, 0)]),  # 3
    ("Add", False, [(3, 0, 0)]),  # 4
    ("Softmax", False, [(4, 0, 0)]),  # 5
    ("MatMul", False, [(5, 0, 0)]),  # 6
    ("Transpose", True, [(6, 0, 1)]),  # 7
    ("Transpose", False, [(6, 0, 0)]),  # 8
    ("FusedMatMul", False, [(7, 0, 1)]),  # 9
    ("SoftmaxGrad_13", False, [(9, 0, 0), (5, 0, 1)]),  # 10
    ("Identity", False, [(10, 0, 0)]),  # 11
    ("Div", False, [(11, 0, 0)]),  # 12
    ("Identity", False, [(12, 0, 0)]),  # 13
    ("FusedMatMul", False, [(2, 0, 1), (13, 0, 0)]),  # 14
    ("FusedMatMul", False, [(1, 0, 0), (13, 0, 1)]),  # 15
    ("FusedMatMul", False, [(5, 0, 0)]),  # 16
    ("Transpose", True, [(16, 0, 1)]),  # 17
    ("Transpose", False, [(14, 0, 0)]),  # 18
    ("Transpose", False, [(15, 0, 0)]),  # 19
    ("Transpose", False, [(16, 0, 0)]),  # 20
]


def _optimize_for_pattern_1(matcher: GraphMatcher, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[3].input[1])
    if not (
        check_attribute_value(nodes[1], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[2], "perm", [0, 2, 3, 1])
        and scale_value is not None
        and check_attribute_value(nodes[7], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[8], "perm", [0, 2, 1, 3])
    ):
        return [], [], []

    _, add_input_shape_0 = matcher.get_type_and_shape(nodes[4].input[0])
    _, add_input_shape_1 = matcher.get_type_and_shape(nodes[4].input[1])
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[1].input[0],
        nodes[2].input[0],
        nodes[7].input[0],
        nodes[8].output[0],
        nodes[17].input[0],
        nodes[18].output[0],
        nodes[19].output[0],
        nodes[20].output[0],
        nodes[4].input[1],
        add_input_shape_0 != add_input_shape_1,
        1 / float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        0.0,
        False,
    )
    return nodes, nodes_to_add, new_value_infos


# No causal mask, no attention mask, without Dropout.
_PATTERN_2: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Mul", True, [(0, 0, 0)]),  # 1
    ("Mul", True, [(0, 0, 1)]),  # 2
    ("Cast", True, [(1, 0, 0)]),  # 3
    ("Cast", True, [(2, 0, 0)]),  # 4
    ("Transpose", True, [(3, 0, 0)]),  # 5
    ("Transpose", True, [(4, 0, 0)]),  # 6
    ("Softmax", False, [(0, 0, 0)]),  # 7
    ("Cast", False, [(7, 0, 0)]),  # 8
    ("MatMul", False, [(8, 0, 0)]),  # 9
    ("Transpose", True, [(9, 0, 1)]),  # 10
    ("Transpose", False, [(9, 0, 0)]),  # 11
    ("FusedMatMul", False, [(10, 0, 1)]),  # 12
    ("Cast", False, [(12, 0, 0)]),  # 13
    ("SoftmaxGrad_13", False, [(13, 0, 0), (7, 0, 1)]),  # 14
    ("FusedMatMul", False, [(2, 0, 1), (14, 0, 0)]),  # 15
    ("FusedMatMul", False, [(1, 0, 0), (14, 0, 1)]),  # 16
    ("Mul", False, [(15, 0, 0)]),  # 17
    ("Mul", False, [(16, 0, 0)]),  # 18
    ("Identity", False, [(17, 0, 0)]),  # 19
    ("Identity", False, [(18, 0, 0)]),  # 20
    ("Cast", False, [(19, 0, 0)]),  # 21
    ("Cast", False, [(20, 0, 0)]),  # 22
    ("Transpose", False, [(21, 0, 0)]),  # 23
    ("Transpose", False, [(22, 0, 0)]),  # 24
    ("FusedMatMul", False, [(8, 0, 0)]),  # 25
    ("Transpose", True, [(25, 0, 1)]),  # 26
    ("Transpose", False, [(25, 0, 0)]),  # 27
]


def _optimize_for_pattern_2(matcher: GraphMatcher, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value_1 = matcher.get_constant_value(nodes[1].input[1])
    scale_value_1 = scale_value_1[0] if isinstance(scale_value_1, list) else scale_value_1
    scale_value_2 = matcher.get_constant_value(nodes[2].input[1])
    scale_value_2 = scale_value_2[0] if isinstance(scale_value_2, list) else scale_value_2
    if not (
        check_attribute_value(nodes[3], "to", 1)
        and check_attribute_value(nodes[4], "to", 1)
        and check_attribute_value(nodes[5], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[6], "perm", [0, 2, 3, 1])
        and check_attribute_value(nodes[8], "to", 10)
        and check_attribute_value(nodes[10], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[11], "perm", [0, 2, 1, 3])
        and scale_value_1 == scale_value_2
    ):
        return [], [], []

    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[5].input[0],
        nodes[6].input[0],
        nodes[10].input[0],
        nodes[11].output[0],
        nodes[26].input[0],
        nodes[23].output[0],
        nodes[24].output[0],
        nodes[27].output[0],
        "",
        False,
        scale_value_1,
        0.0,
        False,
    )
    return nodes, nodes_to_add, new_value_infos


# Has causal mask, no attention mask, without Dropout.
_PATTERN_3: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Mul", True, [(0, 0, 0)]),  # 1
    ("Mul", True, [(0, 0, 1)]),  # 2
    ("Cast", True, [(1, 0, 0)]),  # 3
    ("Cast", True, [(2, 0, 0)]),  # 4
    ("Transpose", True, [(3, 0, 0)]),  # 5
    ("Transpose", True, [(4, 0, 0)]),  # 6
    ("Add", False, [(0, 0, 0)]),  # 7
    ("Cast", True, [(7, 0, 1)]),  # 8
    ("Slice", True, [(8, 0, 0)]),  # 9
    ("Slice", True, [(9, 0, 0)]),  # 10
    ("Unsqueeze", True, [(9, 0, 2)]),  # 11
    ("Gather", True, [(11, 0, 0)]),  # 12
    ("Shape", True, [(12, 0, 0)]),  # 13
    ("Softmax", False, [(7, 0, 0)]),  # 14
    ("Cast", False, [(14, 0, 0)]),  # 15
    ("MatMul", False, [(15, 0, 0)]),  # 16
    ("Transpose", True, [(16, 0, 1)]),  # 17
    ("Transpose", False, [(16, 0, 0)]),  # 18
    ("FusedMatMul", False, [(17, 0, 1)]),  # 19
    ("Cast", False, [(19, 0, 0)]),  # 20
    ("SoftmaxGrad_13", False, [(20, 0, 0), (14, 0, 1)]),  # 21
    ("Identity", False, [(21, 0, 0)]),  # 22
    ("FusedMatMul", False, [(2, 0, 1), (22, 0, 0)]),  # 23
    ("FusedMatMul", False, [(1, 0, 0), (22, 0, 1)]),  # 24
    ("Mul", False, [(23, 0, 0)]),  # 25
    ("Mul", False, [(24, 0, 0)]),  # 26
    ("Identity", False, [(25, 0, 0)]),  # 27
    ("Identity", False, [(26, 0, 0)]),  # 28
    ("Cast", False, [(27, 0, 0)]),  # 29
    ("Cast", False, [(28, 0, 0)]),  # 30
    ("Transpose", False, [(29, 0, 0)]),  # 31
    ("Transpose", False, [(30, 0, 0)]),  # 32
    ("FusedMatMul", False, [(15, 0, 0)]),  # 33
    ("Transpose", True, [(33, 0, 1)]),  # 34
    ("Transpose", False, [(33, 0, 0)]),  # 35
]


def _optimize_for_pattern_3(matcher: GraphMatcher, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value_1 = matcher.get_constant_value(nodes[1].input[1])
    scale_value_1 = scale_value_1[0] if isinstance(scale_value_1, list) else scale_value_1
    scale_value_2 = matcher.get_constant_value(nodes[2].input[1])
    scale_value_2 = scale_value_2[0] if isinstance(scale_value_2, list) else scale_value_2
    if not (
        check_attribute_value(nodes[3], "to", 1)
        and check_attribute_value(nodes[4], "to", 1)
        and check_attribute_value(nodes[5], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[6], "perm", [0, 2, 3, 1])
        and check_attribute_value(nodes[15], "to", 10)
        and check_attribute_value(nodes[17], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[18], "perm", [0, 2, 1, 3])
        and scale_value_1 == scale_value_2
    ):
        return [], [], []

    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[5].input[0],
        nodes[6].input[0],
        nodes[17].input[0],
        nodes[18].output[0],
        nodes[34].input[0],
        nodes[31].output[0],
        nodes[32].output[0],
        nodes[35].output[0],
        "",
        False,
        scale_value_1,
        0.0,
        True,
    )
    return nodes, nodes_to_add, new_value_infos


_PATTERNS = [
    (_PATTERN_0, _optimize_for_pattern_0),
    (_PATTERN_1, _optimize_for_pattern_1),
    (_PATTERN_2, _optimize_for_pattern_2),
    (_PATTERN_3, _optimize_for_pattern_3),
]


@register_graph_optimizer(devices="cuda")
def optimize_graph_for_aten_efficient_attention(graph: GraphProto):
    nodes_to_remove = []
    nodes_to_add = []
    new_value_infos = []
    matcher = GraphMatcher(graph)
    idx = 0
    for pattern_tuple in _PATTERNS:
        for nodes in matcher.match_pattern(pattern_tuple[0]):
            remove_nodes, add_nodes, add_value_infos = pattern_tuple[1](matcher, idx, nodes)
            if len(add_nodes) > 0:
                nodes_to_remove.extend(remove_nodes)
                nodes_to_add.extend(add_nodes)
                new_value_infos.extend(add_value_infos)
                idx += 1
    update_graph(graph, nodes_to_remove, nodes_to_add, new_value_infos)
