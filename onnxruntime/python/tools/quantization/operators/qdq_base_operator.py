import itertools
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, quantize_nparray


class QDQOperatorBase:
    def __init__(self, onnx_quantizer, onnx_node):
        self.quantizer = onnx_quantizer
        self.node = onnx_node

    def quantize(self):
        node = self.node

        for tensor_name in itertools.chain(node.input, node.output):
            self.quantizer.quantize_tensor(tensor_name)
