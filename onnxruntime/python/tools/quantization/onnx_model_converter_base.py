from onnxruntime.quantization.onnx_model_processor_base import ONNXModelProcessorBase


class ConverterBase(ONNXModelProcessorBase):
    default_allow_list = [
        "Conv",
        "MatMul",
        "Relu",
        "MaxPool",
        "GlobalAveragePool",
        "Gemm",
        "Add",
        "Reshape",
    ]

    def __init__(self, model=None, allow_list=None):
        self.allow_list = allow_list if allow_list is not None else self.default_allow_list
        super().__init__(model, True)

    def set_allow_list(self, allow_list: list = None):
        self.allow_list = allow_list if allow_list is None else self.default_allow_list

    def import_model_from_path(self, model_path: str, optimize_model=True, sort_model=True):
        super().import_model_from_path(model_path, optimize_model, sort_model)

    @staticmethod
    def parse_arguments():
        return ConverterBase.get_parser().parse_args()

    @staticmethod
    def get_parser():
        parser = ONNXModelProcessorBase.get_parser()
        parser.add_argument(
            "--allow_list",
            required=False,
            default=None,
            nargs="+",
            help="allow list which contains all supported ops that can be converted into fp16.",
        )
        parser.add_argument(
            "--keep_io_types",
            help="keep input and output types as float32",
            default=True,
        )
        parser.description = (
            "Graph fp16 conversion tool for ONNX Runtime.It convert ONNX graph from fp32 to fp16 using " "--allow_list."
        )
        return parser
