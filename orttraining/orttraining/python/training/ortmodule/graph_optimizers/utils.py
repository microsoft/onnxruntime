# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from onnx import GraphProto, NodeProto, TensorProto, helper, numpy_helper


def _get_attribute(node: NodeProto, attr_name: str, default_value: Any = None) -> Any:
    """Get attribute value from node by attribute key."""
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value


def _to_numpy_array(node: Any) -> np.ndarray:
    """Convert Constant node or TensorProto to Python value."""
    tensor = node
    if isinstance(node, NodeProto):
        tensor = _get_attribute(node, "value")
    assert isinstance(tensor, TensorProto)
    return numpy_helper.to_array(tensor).tolist()


class GraphMatcher:
    """Sub-graph matcher with given pattern.

    GraphMatcher takes an ONNX graph to initialize. It tries to match sub-graphs to a given pattern and yield
    matched sub-graphs (a list of matched nodes for each sub-graph) one by one.

    Pattern is described by a list. Each entry of the list is a Tuple:

        Tuple[str, bool, List[Tuple[int, int, int]]], e.g., ("FusedMatMul", False, [(2, 0, 1), (15, 0, 0)])

        * First string is the Op type, e.g., "FusedMatMul".
        * Second bool indicates it's producer node or consumer node for source node.
        * There is a list to describe the edge infos of this node to other nodes, each edge is a tuple with 3 integers,
          first integer is the index of the target node in the list, second integer is the output index of the edge,
          and thrid integer is the input index of the edge.

    For each entry, GraphMatcher used the first edge to lookup target node, and try to use make sure the sug-graph also
    matches rest edge infos.

    Note that when lookup target node, it will only take the first matched node as target node. For example, if a source
    node has multiple "MatMul" consumers nodes comsuming same output, only the first "MatMul" node will be returned.
    You need to avoid using such confusing edge info as the first edge info for node lookup. Try to use other edge to
    avoid such confusion if possible.
    """

    def __init__(self, graph: GraphProto):
        self._graph: GraphProto = graph
        self._op_type_to_nodes: Dict[str, List[NodeProto]] = {}
        self._consumer_count: Dict[str, int] = {}
        for node in graph.node:
            if node.op_type not in self._op_type_to_nodes:
                self._op_type_to_nodes[node.op_type] = []
            self._op_type_to_nodes[node.op_type].append(node)
            for input in node.input:
                self._consumer_count[input] = self._consumer_count.get(input, 0) + 1

    def _get_producer(self, arg: str, op_type: str, output_idx: int):
        for node in self._op_type_to_nodes.get(op_type, []):
            if (output_idx >= 0 and len(node.output) > output_idx and node.output[output_idx] == arg) or (
                output_idx == -1 and arg in node.output
            ):
                return node
        return None

    def _get_consumer(self, arg: str, op_type: str, input_idx: int):
        for node in self._op_type_to_nodes.get(op_type, []):
            if (input_idx >= 0 and len(node.input) > input_idx and node.input[input_idx] == arg) or (
                input_idx == -1 and arg in node.input
            ):
                return node
        return None

    def get_consumer_count(self, arg: str):
        return self._consumer_count.get(arg, 0)

    def get_constant_value(self, arg: str):
        node_or_initializer = None
        if "Constant" in self._op_type_to_nodes:
            for node in self._op_type_to_nodes["Constant"]:
                if arg in node.output:
                    node_or_initializer = node
                    break
        if node_or_initializer is None:
            for initializer in self._graph.initializer:
                if arg == initializer.name:
                    node_or_initializer = initializer
                    break
        if node_or_initializer is None:
            return None
        return _to_numpy_array(node_or_initializer)

    def get_type_and_shape(self, arg: str):
        value_infos = [
            value_info
            for value_info in itertools.chain(self._graph.input, self._graph.value_info)
            if value_info.name == arg
        ]
        if len(value_infos) > 0 and value_infos[0].type.tensor_type.HasField("shape"):
            shape = []
            for dim in value_infos[0].type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            return value_infos[0].type.tensor_type.elem_type, shape
        initializers = [initializer for initializer in self._graph.initializer if initializer.name == arg]
        if len(initializers) > 0:
            return initializers[0].data_type, initializers[0].dims
        return None, None

    def _match_pattern(self, node: NodeProto, pattern: List[Tuple[str, bool, List[Tuple[int, int, int]]]]):
        nodes = [node]
        for i in range(1, len(pattern)):
            next_op_type = pattern[i][0]
            is_producer = pattern[i][1]
            node_idx, output_idx, input_idx = pattern[i][2][0]
            next_node = (
                self._get_producer(nodes[node_idx].input[input_idx], next_op_type, output_idx)
                if is_producer
                else self._get_consumer(nodes[node_idx].output[output_idx], next_op_type, input_idx)
            )
            if next_node is None:
                return []
            for j in range(1, len(pattern[i][2])):
                node_idx, output_idx, input_idx = pattern[i][2][j]
                assert output_idx >= 0 and input_idx >= 0
                if (not is_producer and nodes[node_idx].output[output_idx] != next_node.input[input_idx]) or (
                    is_producer and next_node.output[output_idx] != nodes[node_idx].input[input_idx]
                ):
                    return []
            nodes.append(next_node)
        return nodes

    def match_pattern(self, pattern: List[Tuple[str, bool, List[Tuple[int, int, int]]]]):
        for node in self._op_type_to_nodes.get(pattern[0][0], []):
            result = self._match_pattern(node, pattern)
            if len(result) == len(pattern):
                yield result


def check_attribute_value(node: NodeProto, attr_name: str, expected_value: Any):
    """Check if the attribute of given node has expected value."""
    value = _get_attribute(node, attr_name)
    return value == expected_value


def make_constant_node(name: str, dtype: TensorProto.DataType, dims: Sequence[int], vals: Any):
    """Create a constant node with given constant tensor (data type, shape, and data)."""
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=dtype, dims=dims, vals=vals),
    )


def update_graph(
    graph: GraphProto,
    nodes_to_remove: List[NodeProto],
    nodes_to_add: List[NodeProto],
    new_value_infos: List[TensorProto] = [],  # noqa: B006
):
    """Update an ONNX graph by removing some nodes, and adding some new nodes and value infos."""
    nodes = [node for node in graph.node if node not in nodes_to_remove]
    nodes.extend(nodes_to_add)
    graph.ClearField("node")
    graph.node.extend(nodes)
    if len(new_value_infos) > 0:
        graph.value_info.extend(new_value_infos)
