from ..quant_utils import FLOAT8_TYPES
from .base_operator import QuantOperatorBase
from .direct_q8 import Direct8BitOp, QDQDirect8BitOp


class QMaxPool(Direct8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "MaxPool"

        # if version is less than 12, go to normal quantize.
        if self.quantizer.opset_version < 12:
            QuantOperatorBase.quantize(self)
            return

        # FP8 types are not supported for MaxPool; emit node unquantized.
        if self.quantizer.activation_qType in FLOAT8_TYPES:
            return QuantOperatorBase.quantize(self)

        # Direct 8bits op
        return super().quantize()


class QDQMaxPool(QDQDirect8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def reg2quant(self):
        node = self.node
        assert node.op_type == "MaxPool"

        # if version is less than 12, just no change
        if self.quantizer.opset_version < 12:
            return

        # FP8 types are not supported for MaxPool; leave node unquantized.
        if self.quantizer.activation_qType in FLOAT8_TYPES:
            return

        # Direct 8bits op
        return super().reg2quant()
