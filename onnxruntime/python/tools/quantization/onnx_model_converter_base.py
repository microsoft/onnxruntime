import argparse

from python.tools.quantization.onnx_model import ONNXModel
from python.tools.quantization.onnx_model_processor_base import ONNXModelProcessorBase


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Graph fp16 conversion tool for ONNX Runtime."
        "It convert ONNX graph from fp32 to fp16 using --allow_list."
    )
    parser.add_argument("--input", required=True, type=str, help="input onnx model path")

    parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")
    parser.add_argument(
        "--allow_list",
        required=False,
        default=[],
        nargs="+",
        help="allow list which contains all supported ops that can be converted into fp16.",
    )
    parser.add_argument(
        "--use_external_data_format",
        required=False,
        action="store_true",
        default=False,
        help="use external data format to store large model (>2GB)",
    )
    parser.set_defaults(use_external_data_format=False)
    parser.add_argument(
        "--keep_io_types",
        type=bool,
        required=False,
        help="keep input and output types as float32",
        default=False,
    )

    args = parser.parse_args()
    return args


class ConverterBase(ONNXModelProcessorBase):
    default_allow_list = ["Conv", "MatMul"]

    def __init__(self, model=None, allow_list=None):
        if model is not None:
            model = ONNXModel(model).topological_sort()
        self.allow_list = allow_list if allow_list is not None else self.default_allow_list
        super().__init__(model)

    def set_allow_list(self, allow_list: list = None):
        self.allow_list = allow_list if allow_list is None else self.default_allow_list

    def process(self):
        raise NotImplementedError
