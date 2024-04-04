# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Union

import onnx


def get_output_from_output_name(onnx_model: onnx.ModelProto, output_name: str) -> onnx.ValueInfoProto:
    """Returns the graph output for the given output name"""

    for go in onnx_model.graph.output:
        if go.name == output_name:
            return go

    raise LookupError(f"The provided output name {output_name} is not a graph output.")


def get_input_from_input_name(onnx_model: onnx.ModelProto, input_name: str) -> onnx.ValueInfoProto:
    """Returns the graph input for the given input name"""

    for gi in onnx_model.graph.input:
        if gi.name == input_name:
            return gi

    raise LookupError(f"The provided output name {input_name} is not a graph input.")


_GRAPH_TOKEN = 0


def _get_token() -> int:
    """Return a token that is one more than the previous token retrieved by calling this function."""

    global _GRAPH_TOKEN  # pylint: disable=global-statement  # noqa: PLW0603
    _GRAPH_TOKEN += 1
    return _GRAPH_TOKEN


def generate_graph_name(token: str) -> str:
    """Return a string that can be used in the graph as a graph attribute name."""

    return f"onnx::{token}::{_get_token()}"


def register_graph_outputs(model: onnx.ModelProto, output_names: Union[List[str], str]) -> None:
    """Register the given output names as graph outputs.

    The graph outputs shape information is extracted from the graph value_infos and
    existing graph outputs. The graph output can only be added to the
    graph for those outputs whose value info is known. If value info
    is not known, an error will be raised.
    """

    if isinstance(output_names, str):
        output_names = [output_names]

    name_value_info_mapping = {value_info.name: value_info for value_info in model.graph.value_info}
    name_graph_output_mapping = {output.name: output for output in model.graph.output}

    # collect all new graph outputs (i.e. graph outputs that are not
    # already graph outputs)
    graph_outputs = []
    for output_name in output_names:
        if output_name in name_graph_output_mapping:
            graph_outputs.append(name_graph_output_mapping[output_name])
        elif output_name in name_value_info_mapping:
            graph_outputs.append(name_value_info_mapping[output_name])
        else:
            raise LookupError(f"The provided name {output_name} is not a graph value info or a graph output.")

    del model.graph.output[:]

    model.graph.output.extend(graph_outputs)


def node_arg_exists(model: onnx.ModelProto, node_arg_name: str) -> bool:
    """Returns True if the given node_arg_name exists in the model graph."""

    for node in model.graph.node:
        if node_arg_name in node.input:
            return True

        if node_arg_name in node.output:
            return True

    return False
