import numpy as np
import onnx
from onnx import onnx_pb as onnx_proto

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType, attribute_to_kwarg
from .base_operator import QuantOperatorBase


class QLinearConvTranspose(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "ConvTranspose"
        print(f"Custom quantization code for {node.op_type}")

        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])

        if self.quantizer.is_input_a_initializer(node.input[1]) and self.quantizer.is_per_channel():
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])
            quant_weight_tuple = self.quantizer.quantize_weight_per_channel(
                node.input[1], onnx_proto.TensorProto.INT8, 0
            )
            quantized_input_names.append(quant_weight_tuple[0])
            zero_point_names.append(quant_weight_tuple[1])
            scale_names.append(quant_weight_tuple[2])
        else:
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])

            (
                quantized_input_names_weight,
                zero_point_names_weight,
                scale_names_weight,
                nodes_weight,
            ) = self.quantizer.quantize_weight(node, [1], reduce_range=self.quantizer.reduce_range)
            quantized_input_names.extend(quantized_input_names_weight)
            zero_point_names.extend(zero_point_names_weight)
            scale_names.extend(scale_names_weight)
            nodes.extend(nodes_weight)

        if not data_found or quantized_input_names is None:
            return super().quantize()

        quantized_bias_name = ""
        bias_present = False
        if len(node.input) == 3:
            quantized_bias_name = self.quantizer.quantize_bias_static(node.input[2], node.input[0], node.input[1])
            bias_present = True

        qlinear_conv_output = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        qlinear_conv_name = qlinear_conv_name = node.name + "_quant" if node.name != "" else ""

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        qlinear_conv_inputs = []

        # Input 0
        qlinear_conv_inputs.append(quantized_input_names[0])
        qlinear_conv_inputs.append(scale_names[0])
        qlinear_conv_inputs.append(zero_point_names[0])

        # Input 1
        qlinear_conv_inputs.append(quantized_input_names[1])
        qlinear_conv_inputs.append(scale_names[1])
        qlinear_conv_inputs.append(zero_point_names[1])

        # Output
        qlinear_conv_inputs.append(output_scale_name)
        qlinear_conv_inputs.append(output_zp_name)

        if bias_present:
            qlinear_conv_inputs.append(quantized_bias_name)

        qlinear_conv_node = onnx.helper.make_node(
            "QLinearConvTranspose", qlinear_conv_inputs, [qlinear_conv_output], qlinear_conv_name, **kwargs
        )
        nodes.append(qlinear_conv_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            qlinear_conv_output,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes
