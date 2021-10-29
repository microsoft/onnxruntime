import itertools
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, quantize_nparray


class QDQOperatorBase:
    def __init__(self, onnx_quantizer, onnx_node, enable_qdq_for_node_output=False):
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        self.enable_qdq_for_node_output = enable_qdq_for_node_output  

    def quantize(self):
        node = self.node

        if self.enable_qdq_for_node_output:
            nodes_to_iterate = itertools.chain(node.input, node.output)
        else:
            nodes_to_iterate = node.input

        for tensor_name in nodes_to_iterate:
            self.quantizer.quantize_tensor(tensor_name)
