# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from onnx import helper
from typing import Any, Sequence

import numpy as np
import onnx

from onnx_model import OnnxModel

logger = getLogger(__name__)


class DynamoOnnxHelper:
    """
    Helper class for processing ONNX models exported by Torch Dynamo.
    """

    def __init__(self, model: onnx.ModelProto):
        self.model = OnnxModel(model)

    def update_edges(self, edge_mapping: dict) -> None:
        """
        Updates the edges in the model according to the given mapping.
        """
        for node in self.model.model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] in edge_mapping:
                    node.input[i] = edge_mapping[node.input[i]]
            for i in range(len(node.output)):
                if node.output[i] in edge_mapping:
                    node.output[i] = edge_mapping[node.output[i]]

        for graph_input in self.model.model.graph.input:
            if graph_input.name in edge_mapping:
                graph_input.name = edge_mapping[graph_input.name]
        for graph_output in self.model.model.graph.output:
            if graph_output.name in edge_mapping:
                graph_output.name = edge_mapping[graph_output.name]

    def unroll_function(self, func_name: str) -> None:
        """
        Unrolls the function with the given name in the model.
        """
        logger.info(f"Unrolling function {func_name}...")
        nodes_to_remove = []
        nodes_to_add = []
        edges_to_remove = []
        edges_to_add = []
        for node in self.model.model.graph.node:
            if node.op_type == func_name:
                nodes_to_remove.append(node)
                edges_to_remove.extend(list(node.input) + list(node.output))

        func_to_remove = None
        for f in self.model.model.functions:
            if f.name == func_name:
                nodes_to_add.extend(list(f.node))
                edges_to_add.extend(list(f.input) + list(f.output))
                func_to_remove = f

        assert len(edges_to_remove) == len(edges_to_add)

        for node in nodes_to_remove:
            self.model.model.graph.node.remove(node)
        for node in nodes_to_add:
            self.model.model.graph.node.append(node)
        if func_to_remove is not None:
            self.model.model.functions.remove(func_to_remove)

        edge_mapping = {}
        for i in range(len(edges_to_remove)):
            k = edges_to_remove[i]
            v = edges_to_add[i]
            if k != v:
                edge_mapping[k] = v

        return self.update_edges(edge_mapping)

    def remove_function(self, func_name: str, input_id: int, output_id: int) -> None:
        """
        Removes the function in the model.
        """
        edge_mapping = {}
        nodes_to_remove = []
        for node in self.model.model.graph.node:
            if node.op_type.find(func_name) != -1:
                edge_mapping[node.input[input_id]] = node.output[output_id]
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.model.model.graph.node.remove(node)

        self.update_edges(edge_mapping)

    def remove_dropout_layer(self) -> None:
        """
        Removes the dropout layer in the model.
        """
        logger.info("Removing dropout layer...")
        self.remove_function("Dropout", 0, 0)

    def remove_lm_head_layer(self) -> None:
        """
        Removes the LM head layer in the model.
        """
        logger.info("Removing LM head layer...")
        # bugbug: need to copy the right vi over
        self.remove_function("Linear_lm_head", 2, 0)

    def add_initializer(self, name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = True):
        if raw:
            np_type = helper.tensor_dtype_to_np_dtype(data_type)
            if not isinstance(vals, np.ndarray):
                bytes = np.array(vals, dtype=np_type).tobytes()
            else:
                bytes = vals.astype(np_type).tobytes()
            tensor = helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=bytes,
                raw=True,
            )
        else:
            tensor = helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=vals,
                raw=False,
            )

        self.model.add_initializer(tensor)
        return tensor

    def convert_constants_to_initializers(self, minimum = 100) -> None:
        """
        Converts Constant ops of size [minimum] or higher to initializers
        """
        logger.info(f"Converting constants greater than size {minimum} to initializers")

        constant_nodes = self.model.get_nodes_by_op_type("Constant")
        nodes_to_remove = []

        for node in constant_nodes:
            # Get info from Constant op
            np_data = self.model.get_constant_value(node.output[0])

            # Skip if there are less than [minimum] elements
            if np_data is None or np_data.size < minimum:
                continue

            # Add new initializer with same name as Constant op's output
            self.add_initializer(
                name=node.output[0],
                data_type=node.attribute[0].t.data_type,
                dims=list(np_data.shape),
                vals=np_data,
            )

            nodes_to_remove.append(node)

            # Update nodes that use output of Constant op to use initializer name
            # for child_node in input_name_to_nodes[node.output[0]]:
            #     for i, child_input in enumerate(child_node.input):
            #         if child_input == node.output[0]:
            #             child_node.input[i] = INITIALIZER_NAME_WHEN_CREATED
            #             break

        # Remove Constant ops from graph
        self.model.remove_nodes(nodes_to_remove)

    def clear_metadata(self) -> None:
        """
        Clear metadata fields in all nodes
        """
        for graph in self.model.graphs():
            graph.ClearField("metadata_props")
        for node in self.model.nodes():
            node.ClearField("metadata_props")
