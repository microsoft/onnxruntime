# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os

import onnx


class DynamoOnnxHelper:
    """
    Helper class for processing ONNX models exported by torch Dynamo.
    """

    def __init__(self):
        pass

    def update_edges(self, model: onnx.ModelProto, edge_mapping: dict) -> onnx.ModelProto:
        """
        Updates the edges in the model according to the given mapping.
        """
        for node in model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] in edge_mapping:
                    node.input[i] = edge_mapping[node.input[i]]
            for i in range(len(node.output)):
                if node.output[i] in edge_mapping:
                    node.output[i] = edge_mapping[node.output[i]]

        for graph_input in model.graph.input:
            if graph_input.name in edge_mapping:
                graph_input.name = edge_mapping[graph_input.name]
        for graph_output in model.graph.output:
            if graph_output.name in edge_mapping:
                graph_output.name = edge_mapping[graph_output.name]

        return model

    def unroll_function(self, model: onnx.ModelProto, func_name: str) -> onnx.ModelProto:
        """
        Unrolls the function with the given name in the model.
        """
        logging.info(f"Unrolling function {func_name}...")
        nodes_to_remove = []
        nodes_to_add = []
        edges_to_remove = []
        edges_to_add = []
        for node in model.graph.node:
            if node.op_type == func_name:
                nodes_to_remove.append(node)
                edges_to_remove.extend(list(node.input) + list(node.output))

        func_to_remove = None
        for f in model.functions:
            if f.name == func_name:
                nodes_to_add.extend(list(f.node))
                edges_to_add.extend(list(f.input) + list(f.output))
                func_to_remove = f

        assert len(edges_to_remove) == len(edges_to_add)

        for node in nodes_to_remove:
            model.graph.node.remove(node)
        for node in nodes_to_add:
            model.graph.node.append(node)
        if func_to_remove is not None:
            model.functions.remove(func_to_remove)

        edge_mapping = {}
        for i in range(len(edges_to_remove)):
            k = edges_to_remove[i]
            v = edges_to_add[i]
            if k != v:
                edge_mapping[k] = v

        return self.update_edges(model, edge_mapping)

    def remove_dropout_layer(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Removes the dropout layer in the model.
        """
        logging.info("Removing dropout layer...")
        edge_mapping = {}
        nodes_to_remove = []
        for node in model.graph.node:
            if node.op_type.find("Dropout") != -1:
                assert len(node.input) == 1
                assert len(node.output) == 1
                edge_mapping[node.output[0]] = node.input[0]
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            model.graph.node.remove(node)

        return self.update_edges(model, edge_mapping)

    def erase_onnx_model(self, onnx_path: str) -> None:
        assert onnx_path.endswith(".onnx")
        if not os.path.exists(onnx_path):
            return

        model = onnx.load_model(onnx_path, load_external_data=False)
        onnx_data_path = None
        for initializer in model.graph.initializer:
            if initializer.data_location == 1 and initializer.external_data[0].key == "location":
                onnx_data_path = "./" + initializer.external_data[0].value
                break
        logging.info(f"Erasing {onnx_path}...")
        os.remove(onnx_path)
        if onnx_data_path is not None:
            logging.info(f"Erasing {onnx_data_path}...")
            os.remove(onnx_data_path)
