from python.tools.quantization.onnx_model_converter_base import ConverterBase
from python.tools.quantization.onnx_model import ONNXModel


class FP16ConverterFinal(ConverterBase):
    def __init__(self, model=None, allow_list=None):
        super().__init__(model, allow_list)

    def process(self, keep_io_types=True):
        return None
