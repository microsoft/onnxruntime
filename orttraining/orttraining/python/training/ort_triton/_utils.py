# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from onnx import GraphProto, NodeProto, TensorProto, helper, numpy_helper


def gen_unique_name(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:8]


def _topological_sort_internal(node, visited, output_consumers, sorted_nodes):
    visited.add(node.name)
    for next_node in output_consumers[node.name]:
        if next_node.name not in visited:
            _topological_sort_internal(next_node, visited, output_consumers, sorted_nodes)

    sorted_nodes.insert(0, node)


# Topological sort of nodes given the input names. The list of nodes contain both constant and non-constant nodes.
def topological_sort(inputs: list[str], nodes: list[NodeProto]) -> list[NodeProto]:
    const_nodes = []
    non_const_nodes = []
    for node in nodes:
        if not node.name:
            node.name = gen_unique_name(node.op_type)
        if node.op_type == "Constant":
            inputs.append(node.output[0])
            const_nodes.append(node)
        else:
            non_const_nodes.append(node)

    # Build relationship between nodes.
    graph_input_consumers = defaultdict(list)
    output_consumers = defaultdict(list)
    input_set = set(inputs)
    for node in non_const_nodes:
        for input in node.input:
            if input in input_set:
                graph_input_consumers[input].append(node)
        for output in node.output:
            if not output:
                continue
            for consumer in non_const_nodes:
                if output in consumer.input:
                    output_consumers[node.name].append(consumer)

    # Topological sort.
    visited = set()
    sorted_nodes = []
    for input in inputs:
        for node in graph_input_consumers[input]:
            if node.name not in visited:
                _topological_sort_internal(node, visited, output_consumers, sorted_nodes)

    return const_nodes + sorted_nodes


# Get attribute value from node by attribute key.
def get_attribute(node: NodeProto, attr_name: str, default_value: Any = None) -> Any:
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value


# Convert Constant node or TensorProto to torch.Tensor.
def to_torch_tensor(node: Any) -> torch.Tensor:
    tensor = node
    if isinstance(node, NodeProto):
        tensor = get_attribute(node, "value")
    assert isinstance(tensor, TensorProto)
    torch_tensor = torch.from_numpy(numpy_helper.to_array(tensor))
    # numpy does not support bfloat16 and create a float32 tensor instead.
    if tensor.data_type == TensorProto.BFLOAT16:
        torch_tensor = torch_tensor.to(torch.bfloat16)
    return torch_tensor


def to_torch_dtype(tensor_type: TensorProto.DataType) -> torch.dtype:
    # Native numpy does not support bfloat16.
    if tensor_type == TensorProto.BFLOAT16:
        return torch.bfloat16
    return torch.from_numpy(np.zeros(1, dtype=helper.tensor_dtype_to_np_dtype(tensor_type))).dtype


# Generate a unique variable name based on the node arg name.
def gen_variable_name(name: str, prefix: str, existing_names: set) -> str:
    pos = name.rfind("/")
    if pos != -1:
        name = name[pos + 1 :]
    pos = name.rfind(".")
    if pos != -1:
        name = name[pos + 1 :]
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    if len(name) > 20:
        name = name[-20:]

    name = f"{prefix}_{name}"
    while name in existing_names:
        name = name + "_1"

    existing_names.add(name)
    return name


def may_add_brackets(name: str) -> str:
    if not re.match("^[A-Za-z0-9_.]*$", name):
        return f"({name})"
    return name


def sort_reduce_axes(axes: list[int], rank: int, check_contiguous: bool = True) -> list[int]:
    axes = [axis + rank if axis < 0 else axis for axis in axes]
    axes.sort()
    if check_contiguous:
        for i in range(1, len(axes)):
            assert axes[i] == axes[i - 1] + 1
    return axes


# Get the keep_dims attribute and reduce axes from a reduce node.
def get_reduce_info(node: NodeProto, graph: GraphProto, input_rank: int) -> tuple[int, list[int]]:
    keep_dims = get_attribute(node, "keepdims", 1)
    noop_with_empty_axes = get_attribute(node, "noop_with_empty_axes", 0)
    axes = get_attribute(node, "axes", None)
    if axes is None and len(node.input) > 1:
        axes_initializer = None
        for initializer in graph.initializer:
            if initializer.name == node.input[1]:
                axes_initializer = initializer
                break
        assert axes_initializer is not None
        axes = to_torch_tensor(axes_initializer).tolist()
    if axes is None:
        axes = list(range(input_rank)) if noop_with_empty_axes == 0 else []
    axes = sort_reduce_axes(axes, input_rank, check_contiguous=False)
    return keep_dims, axes


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def is_number(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return name.startswith("float(") and name.endswith(")")
