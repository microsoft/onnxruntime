import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import find_by_name, get_mul_node, QuantizedValue, QuantizedValueType
from onnx import onnx_pb as onnx_proto
'''
    Used when quantize mode is QuantizationMode.IntegerOps.
'''


class MatMulInteger(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1], reduce_range=True, op_level_per_channel=True)

        matmul_integer_output = node.output[0] + "_output_quantized"
        matmul_integer_name = node.name + "_quant" if node.name != "" else ""
        matmul_integer_node = onnx.helper.make_node("MatMulInteger", quantized_input_names + zero_point_names,
                                                    [matmul_integer_output], matmul_integer_name)
        nodes.append(matmul_integer_node)

        # Add cast operation to cast matmulInteger output to float.
        cast_op_output = matmul_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [matmul_integer_output], [cast_op_output],
                                          matmul_integer_output + "_cast",
                                          to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert (len(scale_names) == 2)
        scales_mul_op = matmul_integer_name + "_scales_mul" if matmul_integer_name != "" else scale_names[
            0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = find_by_name(scales_mul_op, self.quantizer.new_nodes)
        if scales_mul_node is None:
            scales_mul_node = get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of MatMulInteger
        # and make the output of this node the same as output of original matmul node.
        output_scale_mul_op = ""
        if matmul_integer_name != "":
            output_scale_mul_op = matmul_integer_name + "_output_scale_mul"
        nodes.append(get_mul_node([cast_op_output, scales_mul_op_output], node.output[0], output_scale_mul_op))
        self.quantizer.new_nodes += nodes


'''
    Used when quantize mode is QuantizationMode.QLinearOps
'''


class QLinearMatMul(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1], reduce_range=True, op_level_per_channel=True)

        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])

        if not data_found:
            raise ValueError("Quantization parameters for output:\"{}\" of node:\"{}\" not specified".format(
                node.output[0], node.name))

        qlinear_matmul_output = node.output[0] + "_quantized"
        qlinear_matmul_name = node.name + "_quant" if node.name != "" else ""

        qlinear_matmul_inputs = []
        # Input 0
        qlinear_matmul_inputs.append(quantized_input_names[0])
        qlinear_matmul_inputs.append(scale_names[0])
        qlinear_matmul_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_matmul_inputs.append(quantized_input_names[1])
        qlinear_matmul_inputs.append(scale_names[1])
        qlinear_matmul_inputs.append(zero_point_names[1])
        # Output quantization parameter
        qlinear_matmul_inputs.append(output_scale_name)
        qlinear_matmul_inputs.append(output_zp_name)

        qlinear_matmul_node = onnx.helper.make_node("QLinearMatMul", qlinear_matmul_inputs, [qlinear_matmul_output],
                                                    qlinear_matmul_name)
        nodes.append(qlinear_matmul_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_matmul_output, output_scale_name, output_zp_name,
                                  QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes
