import itertools
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, quantize_nparray


class QDQOperatorBase:
    def __init__(self, onnx_quantizer, onnx_node):
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        self.disable_qdq_for_node_output = True if onnx_node.op_type in onnx_quantizer.op_types_to_exclude_output_quantization \
                                           else False

    def quantize(self):
        node = self.node

        if self.disable_qdq_for_node_output:
            nodes_to_iterate = node.input
        else:
            nodes_to_iterate = itertools.chain(node.input, node.output)

        for tensor_name in nodes_to_iterate:
            if self.quantizer.is_per_channel():
                if node.op_type in self.quantizer.op_types_support_per_channel_quantization :
                    self.quantizer.quantize_tensor_per_channel(tensor_name, self.quantizer.qdq_channel_axis)
                else:
                    self.quantizer.quantize_tensor(tensor_name)
            else:
                self.quantizer.quantize_tensor(tensor_name)
