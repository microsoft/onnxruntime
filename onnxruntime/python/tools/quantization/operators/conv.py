import onnx
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase
from ..quant_utils import find_by_name, get_mul_node, QuantizedValue, QuantizedValueType, attribute_to_kwarg, BiasToQuantize
from onnx import onnx_pb as onnx_proto


class ConvInteger(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1])

        # quantize bias if exist
        quantized_bias_name = ""
        bias_present = False
        if len(node.input) == 3:
            quantized_bias_name = self.quantizer.quantize_bias_dynamic(node.input[2], node.input[0], node.input[1],
                                                                       nodes)
            bias_present = True

        conv_integer_output = node.output[0] + "_output_quantized"
        conv_integer_name = node.name + "_quant" if node.name != "" else ""

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        conv_integer_node = onnx.helper.make_node("ConvInteger", quantized_input_names + zero_point_names,
                                                  [conv_integer_output], conv_integer_name, **kwargs)
        nodes.append(conv_integer_node)

        # Add bias add nodes
        if bias_present:
            conv_integer_output = self.quantizer.get_bias_add_nodes(nodes, node, conv_integer_output,
                                                                    quantized_bias_name)

        # Add cast operation to cast convInteger output to float.
        cast_op_output = conv_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [conv_integer_output], [cast_op_output],
                                          conv_integer_output + "_cast",
                                          to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert (len(scale_names) == 2)
        if conv_integer_name != "":
            scales_mul_op = conv_integer_name + "_scales_mul"
        else:
            scales_mul_op = scale_names[0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = find_by_name(scales_mul_op, self.quantizer.new_nodes)
        if scales_mul_node is None:
            scales_mul_node = get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of ConvInteger
        # and make the output of this node the same as output of original conv node.
        output_scale_mul_op = conv_integer_name + "_output_scale_mul" if conv_integer_name != "" else ""
        nodes.append(get_mul_node([cast_op_output, scales_mul_op_output], node.output[0], output_scale_mul_op))

        self.quantizer.new_nodes += nodes


class QLinearConv(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Conv")

        if self.quantizer.is_input_a_weight(node.input[1]) and self.quantizer.is_per_channel():
            (quantized_input_names, zero_point_names, scale_names, nodes) = \
                self.quantizer.quantize_inputs(node, [0])
            quant_weight_tuple = self.quantizer.quantize_weight_per_channel(node.input[1], onnx_proto.TensorProto.INT8,
                                                                            0)
            quantized_input_names.append(quant_weight_tuple[0])
            zero_point_names.append(quant_weight_tuple[1])
            scale_names.append(quant_weight_tuple[2])
        else:
            (quantized_input_names, zero_point_names, scale_names, nodes) = \
                self.quantizer.quantize_inputs(node, [0, 1])

        quantized_bias_name = ""
        bias_present = False
        if len(node.input) == 3:
            quantized_bias_name = self.quantizer.quantize_bias_static(node.input[2], node.input[0], node.input[1])
            bias_present = True
        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])

        if not data_found:
            raise ValueError("Quantization parameters for output:\"{}\" of node:\"{}\" not specified".format(
                node.output[0], node.name))

        qlinear_conv_output = node.output[0] + "_quantized"
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

        qlinear_conv_node = onnx.helper.make_node("QLinearConv", qlinear_conv_inputs, [qlinear_conv_output],
                                                  qlinear_conv_name, **kwargs)
        nodes.append(qlinear_conv_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_conv_output, output_scale_name, output_zp_name,
                                  QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes


class QDQConv(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Conv")

        self.quantizer.quantize_tensor(node.input[0])
        if self.quantizer.is_per_channel():
            self.quantizer.quantize_tensor_per_channel(node.input[1], 0)
        else:
            self.quantizer.quantize_tensor(node.input[1])

        if len(node.input) == 3:
            self.quantizer.quantize_bias_tensor(node.input[2], node.input[0], node.input[1])
