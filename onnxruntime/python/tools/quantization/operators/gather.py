from onnx import onnx_pb as onnx_proto

from ..quant_utils import QuantizedValue, QuantizedValueType
from .base_operator import QOperatorBase

"""
    Quantize Gather
"""


class GatherQuant(QOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def do_quantization(self):
        node = self.node
        assert node.op_type == "Gather"
        if not self.quantizer.is_valid_quantize_weight(node.input[0]):
            super().do_quantization()
            return

        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_inputs(node, [0])
        if quantized_input_names is None:
            return super().do_quantization()

        gather_new_output = node.output[0] + "_quantized"

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            gather_new_output,
            scale_names[0],
            zero_point_names[0],
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        node.output[0] = gather_new_output
        node.input[0] = quantized_input_names[0]
        nodes.append(node)

        self.quantizer.new_nodes += nodes
