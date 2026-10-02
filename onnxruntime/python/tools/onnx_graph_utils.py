# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections.abc import Iterable, Mapping, Sequence

from onnx import NodeProto


def input_name_to_nodes(nodes: Iterable[NodeProto]) -> dict[str, list[NodeProto]]:
    input_name_to_nodes = {}
    for node in nodes:
        for input_name in node.input:
            if input_name:
                input_name_to_nodes.setdefault(input_name, []).append(node)
    return input_name_to_nodes


def output_name_to_node(nodes: Iterable[NodeProto]) -> dict[str, NodeProto]:
    output_name_to_node = {}
    for node in nodes:
        for output_name in node.output:
            if output_name:
                output_name_to_node[output_name] = node
    return output_name_to_node


def get_children(
    node: NodeProto,
    input_name_to_nodes: Mapping[str, Sequence[NodeProto]],
    output_index: int | None = None,
) -> list[NodeProto]:
    children = []
    if output_index is not None:
        if output_index < len(node.output):
            output = node.output[output_index]
            if output in input_name_to_nodes:
                children = list(input_name_to_nodes[output])
        return children

    for output in node.output:
        if output in input_name_to_nodes:
            children.extend(input_name_to_nodes[output])
    return children


def get_parents(
    node: NodeProto,
    output_name_to_node: Mapping[str, NodeProto],
) -> list[NodeProto]:
    parents = []
    for input_name in node.input:
        if input_name in output_name_to_node:
            parents.append(output_name_to_node[input_name])
    return parents


def get_parent(
    node: NodeProto,
    input_index: int,
    output_name_to_node: Mapping[str, NodeProto],
) -> NodeProto | None:
    if len(node.input) <= input_index:
        return None

    input_name = node.input[input_index]
    if input_name not in output_name_to_node:
        return None

    return output_name_to_node[input_name]
