# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from .qdq_base_operator import QDQOperatorBase


class QDQNormalization(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "InstanceNormalization" or node.op_type == "LayerNormalization"

        # Input
        self.quantizer.quantize_activation_tensor(node.input[0])

        # Scale
        scale_is_initializer = bool(find_by_name(node.input[1], self.quantizer.model.initializer()))

        if self.quantizer.is_per_channel() and scale_is_initializer:
            channel_axis = self.quantizer.qdq_op_type_per_channel_support_to_axis.get(node.op_type, 1)
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], axis=channel_axis)
        elif scale_is_initializer:
            self.quantizer.quantize_weight_tensor(node.input[1])
        else:
            self.quantizer.quantize_activation_tensor(node.input[1])

        # Bias
        self.quantizer.quantize_bias_tensor(node.input[2], node.input[0], node.input[1])

        # Output
        if not self.disable_qdq_for_node_output:
            for output_name in node.output:
                self.quantizer.quantize_activation_tensor(output_name)

