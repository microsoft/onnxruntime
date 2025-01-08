import numpy as np
import onnx

from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

INPUT0_SHAPE = (1, 3, 8, 8)
INPUT0_NAME = "input_0"


def create_f32_model():
    input_0 = onnx.helper.make_tensor_value_info(INPUT0_NAME, onnx.TensorProto.FLOAT, INPUT0_SHAPE)
    output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, None)
    weight_data = [
        [
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
        ],
        [
            [[-1.0, -0.8, -0.6], [-0.1333, 0.0, -0.1333], [0.6, 0.8, 1.0]],  # range = 2.0, scale = 2.0/15, zp = -3
            [[-1.0, -0.8, -0.6], [-0.1333, 0.0, -0.1333], [0.6, 0.8, 1.0]],  # range = 2.0, scale = 2.0/15, zp = -3
            [[-1.0, -0.8, -0.6], [-0.1333, 0.0, -0.1333], [0.6, 0.8, 1.0]],  # range = 2.0, scale = 2.0/15, zp = -3
        ],
        [
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
        ],
        [
            [[-1.0, -0.8, -0.6], [-0.1333, 0.0, -0.1333], [0.6, 0.8, 1.0]],  # range = 2.0, scale = 2.0/15, zp = -3
            [[-1.0, -0.8, -0.6], [-0.1333, 0.0, -0.1333], [0.6, 0.8, 1.0]],  # range = 2.0, scale = 2.0/15, zp = -3
            [[-1.0, -0.8, -0.6], [-0.1333, 0.0, -0.1333], [0.6, 0.8, 1.0]],  # range = 2.0, scale = 2.0/15, zp = -3
        ],
        [
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
            [[-1.5, -1.0, -0.5], [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]],  # range = 3.0, scale = 3.0/15, zp = 0
        ],
    ]
    weight = onnx.numpy_helper.from_array(np.array(weight_data, dtype=np.float32), "weight")
    bias_data = [-10.0, -8.0, 0.0, 8.0, 10.0]
    bias = onnx.numpy_helper.from_array(np.array(bias_data, dtype=np.float32), "bias")

    conv_node = onnx.helper.make_node("Conv", [INPUT0_NAME, "weight", "bias"], ["output_0"], name="Conv0")
    graph = onnx.helper.make_graph(
        [conv_node],
        "Convf32",
        [input_0],
        [output_0],
        initializer=[weight, bias],
    )
    opset_imports = [onnx.helper.make_opsetid("", 21)]
    model = onnx.helper.make_model(graph, opset_imports=opset_imports)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, True)

    return model


class DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = None
        self.data_list = []

        # Generate 10 random input values for calibration
        for _ in range(10):
            input_data = {INPUT0_NAME: np.random.random(INPUT0_SHAPE).astype(np.float32)}
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def create_qdq_model(model_f32):
    # Use tensor quantization overrides to quantize Conv's weight input to 4 bits on axis 0.
    init_overrides = {"weight": [{"quant_type": QuantType.QInt4, "axis": 0, "symmetric": True}]}
    qnn_config = get_qnn_qdq_config(
        model_f32,
        DataReader(),
        init_overrides=init_overrides,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
    )

    quantize(model_f32, "conv.int4_weights.qdq.onnx", qnn_config)


if __name__ == "__main__":
    model_f32 = create_f32_model()
    create_qdq_model(model_f32)
