import onnx
import numpy as np
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, quantize_nparray, TensorToQuantize


class QDQOperatorBase(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        for input in node.input:
            print(input)
            self.quantizer.add_tensor_to_quantize(TensorToQuantize(input))
