# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
"""
the optimizer will do two optimization:
1. if output value of shape nodes' is same then we can rewrite the graph to share the shape nodes,
even the output value is symbolic after shape inference, this will simplify the graph and also may
help CSE optimizer find more same subgraph and thus improve the perf.

for example: the pattern "tensor_x > shape > triu" occurs in every transformer layers
to generate causal mask, without the optimizer they can't be fuse to one,
while actually it can be fuse as tensor_x's shape is same.

2. if part of the value of shape's output is const, then we may do constant folding to the it.

for example: "tensor_x > shape > gather > ...", if shape's output is (x, 100, y)
and gather want to take the 2nd elem, then we can do constant folding.
"""


from collections import defaultdict
import logging
from onnx import helper
from onnxruntime.transformers.onnx_model import OnnxModel


logger = logging.getLogger(__name__)


def modify_graph(graph, node_to_add, node_to_remove):
    logger.debug(f"onnx graph modified, node_to_add: {node_to_add}, node_to_remove: {node_to_remove}")
    nodes = [node for node in graph.node if node not in node_to_remove]
    nodes.extend(node_to_add)
    graph.ClearField("node")
    graph.node.extend(nodes)
    node_to_add.clear()
    node_to_remove.clear()


def sym_data_is_const(sym_data):
    if isinstance(sym_data, int):
        return True
    if isinstance(sym_data, list):
        return all(isinstance(i, int) for i in sym_data)

    return False


def replace_node_to_constant(graph, node_output, value):
    def find_node_by_output_name(graph, output_name):
        for node in graph.node:
            if [output_name] == node.output:
                return node
        return None

    old_node = find_node_by_output_name(graph, node_output)
    if old_node is None:
        return None, None

    value_info_dict = {vi.name: vi.type for vi in graph.value_info}
    dtype = value_info_dict[node_output].tensor_type.elem_type
    dims = [1] if isinstance(value, list) else []
    value = value if isinstance(value, list) else [value]
    new_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[node_output],
        value=helper.make_tensor(name=f"symblic_optimizer_{node_output}", data_type=dtype, dims=dims, vals=value),
    )
    return new_node, old_node


def node_to_const_if_possible(symbolic_shape_inference):
    # from sym_data_, we can know node's output value is const or not
    # if it is const, then we can replace the node by a Constant node
    graph = symbolic_shape_inference.out_mp_.graph
    node_to_add, node_to_remove = [], []
    for output_name, value in symbolic_shape_inference.sympy_data_.items():
        if sym_data_is_const(value):
            new_node, old_node = replace_node_to_constant(graph, output_name, value)
            node_to_add.append(new_node)
            node_to_remove.append(old_node)
    modify_graph(graph, node_to_add, node_to_remove)


def shape_value_to_nodes(symbolic_shape_inference):
    # return a dict
    # whose "key" is the output value, "value" is shape nodes which share the same output value
    def shape_output_name_to_node(symbolic_shape_inference):
        graph = symbolic_shape_inference.out_mp_.graph
        res = {}
        # graph.node should be in topological order
        OnnxModel.graph_topological_sort(graph)
        for node in graph.node:
            if node.op_type == "Shape":
                res[node.output[0]] = node
        return res

    res = defaultdict(list)
    output_name_to_node = shape_output_name_to_node(symbolic_shape_inference)
    for name, shape in symbolic_shape_inference.sympy_data_.items():
        if name not in output_name_to_node:
            continue
        node = output_name_to_node[name]
        if isinstance(shape, list):
            res[tuple(shape)].append(node)
        elif isinstance(shape, int):
            res[(shape,)].append(node)
    return res


def share_shape_ops_if_possible(symbolic_shape_inference):
    # if shape output is same, then actually we only need the first shape node in topology order
    graph = symbolic_shape_inference.out_mp_.graph
    shape_value_to_nodes_dict = shape_value_to_nodes(symbolic_shape_inference)
    node_to_remove = []
    for nodes in shape_value_to_nodes_dict.values():
        if len(nodes) <= 1:
            continue
        shape_node_to_keep = nodes[0]
        for shape_node_to_remove in nodes[1:]:
            for node in graph.node:
                if shape_node_to_remove.output[0] in node.input:
                    new_input = shape_node_to_keep.output[0]
                    old_input = shape_node_to_remove.output[0]
                    new_inputs = [new_input if i == old_input else i for i in node.input]
                    node.ClearField("input")
                    node.input.extend(new_inputs)

            node_to_remove.append(shape_node_to_remove)
    modify_graph(graph, [], node_to_remove)


def optimize_graph(symbolic_shape_inference):
    node_to_const_if_possible(symbolic_shape_inference)
    share_shape_ops_if_possible(symbolic_shape_inference)
