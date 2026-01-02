# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Define NonZero shape inference."""

import logging

import numpy as np
import onnx

from ... import fusions, onnx_model


class ShapeNonZero(fusions.Fusion):
    """Shape inference for NonZero.

    NonZero node produces dynamically shaped output tensor, causing the tensor shapes of following nodes undetermined
    as well. QNN expects NonZero having its shape set to maximum size (i.e., number of total input elements) and let
    runtime handle the dynamic shape later.
    """

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.
        Args:
            model: An onnx_model.ONNXModel instance.
        """
        super().__init__(model, "", "NonZero")

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ) -> bool:
        """Infer shape for NonZero.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., NonZero).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.

        Returns:
            A bool indicating whether the node is updated.
        """
        logging.warning(
            "The model contains a NonZero node which produces a dynamically shaped output tensor."
            "Following QNN requirements, its output shape will be deliberately set to the maximum size."
        )

        if (input_tensor_type := self.model.get_tensor_type(node.input[0])) is None or (
            output_tensor_type := self.model.get_tensor_type(node.output[0])
        ) is None:
            return False

        if not (input_tensor_shape := self.tensor_shape_to_list(input_tensor_type)):
            return False

        if not all(isinstance(dim, int) for dim in input_tensor_shape):
            return False

        output_tensor_type.shape.dim[1].dim_value = np.prod(input_tensor_shape)
        return True

    def apply(self) -> bool:
        """Apply fusion.

        This method is overridden to execute shape inference again since NonZero will have fixed shape.

        Returns:
            A bool indicating whether the model is updated.
        """
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        updated = False
        for node in self.model.nodes():
            if node.op_type == self.search_op_type:
                updated |= self.fuse(node, input_name_to_nodes, output_name_to_node)

        if updated:
            self.model.model = onnx.shape_inference.infer_shapes(self.model.model)

        return updated
