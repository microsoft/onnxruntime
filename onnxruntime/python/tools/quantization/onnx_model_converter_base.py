import numpy as np
from onnx import TensorProto, ValueInfoProto, helper, numpy_helper
from python.tools.quantization.onnx_model import ONNXModel
from python.tools.quantization.onnx_model_processor_base import ONNXModelProcessorBase


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

    @staticmethod
    def _convert_np_float16_to_int(np_array: np.ndarray(shape=(), dtype=np.float16)) -> list[int]:
        """
        Convert numpy float16 to python int.

        :param np_array: numpy float16 list
        :return int_list: python int list
        """
        return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_array]

    def _convert_tensor_float_to_float16(self, tensor: TensorProto) -> TensorProto:
        """Convert tensor float to float16.

        Args:
            tensor (TensorProto): the tensor to convert.
        Raises:
            ValueError: input type is not TensorProto.

        Returns:
            TensorProto: the converted tensor.
        """

        if not isinstance(tensor, TensorProto):
            raise ValueError("Expected input type is an ONNX TensorProto but got %s" % type(tensor))
        if tensor.data_type == TensorProto.FLOAT16:
            return tensor

        tensor.data_type = TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = self._convert_np_float_to_float16(np.array(tensor.float_data))
            int_list = self._convert_np_float16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.frombuffer(tensor.raw_data, dtype="float32")
            # convert float to float16
            float16_list = self._convert_np_float_to_float16(float32_list)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tobytes()
        return tensor

    @staticmethod
    def _convert_np_float_to_float16(
        np_array: np.ndarray(shape=(), dtype=np.float32),
    ) -> np.ndarray(shape=(), dtype=np.float16):
        """
        Convert float32 numpy array to float16 without changing sign or finiteness.
        Positive values less than min_positive_val are mapped to min_positive_val.
        Positive finite values greater than max_finite_val are mapped to max_finite_val.
        Similar for negative values. NaN, 0, inf, and -inf are unchanged.
        """

        min_positive_val = 5.96e-08
        max_finite_val = 65504.0

        def between(a, b, c):
            return np.logical_and(a < b, b < c)

        np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
        np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
        np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
        np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
        return np.float16(np_array)

    @staticmethod
    def _make_value_info_from_tensor(tensor: TensorProto) -> ValueInfoProto:
        if not isinstance(tensor, TensorProto):
            raise ValueError("Expected input type is an ONNX TensorProto but got %s" % type(tensor))
        shape = numpy_helper.to_array(tensor).shape
        return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)

    @staticmethod
    def parse_arguments():
        ConverterBase.get_parser().parse_args()

    @staticmethod
    def get_parser():

        parser = ONNXModelProcessorBase.get_parser()
        parser.add_argument(
            "--allow_list",
            required=False,
            default=[],
            nargs="+",
            help="allow list which contains all supported ops that can be converted into fp16.",
        )
        parser.add_argument(
            "--keep_io_types",
            type=bool,
            required=False,
            help="keep input and output types as float32",
            default=False,
        )
        parser.description = (
            "Graph fp16 conversion tool for ONNX Runtime.It convert ONNX graph from fp32 to fp16 using " "--allow_list."
        )
        return parser
