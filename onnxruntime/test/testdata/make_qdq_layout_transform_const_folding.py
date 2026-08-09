# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx

import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        inputs = session.get_inputs()

        self.data_list = []

        # Generate 10 random float32 inputs
        for _ in range(10):
            input_data = {inp.name: np.random.random(inp.shape).astype(np.float32) for inp in inputs}
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


if __name__ == "__main__":
    """
    Creates a QDQ model with a shared initializer.
    The transpose optimizer will generate a (weight -> Transpose -> Squeeze) sequence that can be constant folded
    by the tranpose optimizer itself.
    """
    shape = (1, 3, 3, 3)

    input0 = onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, shape)
    input1 = onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, shape)
    output0 = onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, None)
    output1 = onnx.helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT, None)

    # Shared weight (will be unsqueezed and fed into a Squeeze op by layout transformation).
    const_1_weight = onnx.numpy_helper.from_array(np.array([1.0] * shape[-1], dtype=np.float32), "const_1_weight")

    # Transpose with channel-first perm
    transpose_node = onnx.helper.make_node(
        "Transpose", ["input0"], ["transpose_out"], name="transpose_node", perm=(0, 3, 1, 2)
    )

    # Mul0
    mul0_node = onnx.helper.make_node("Mul", ["transpose_out", "const_1_weight"], ["mul0_out"], name="mul0_node")

    # Mul1
    mul1_node = onnx.helper.make_node("Mul", ["input1", "const_1_weight"], ["output1"], name="mul1_node")

    # Conv0
    conv_w_shape = (1, 3, 2, 2)
    conv_weight_data = np.random.normal(-1.0, 1.0, conv_w_shape).astype(np.float32)
    conv_weight = onnx.numpy_helper.from_array(conv_weight_data, "conv_weight")
    conv_node = onnx.helper.make_node("Conv", ["mul0_out", "conv_weight"], ["output0"], name="conv_node")

    graph = onnx.helper.make_graph(
        [transpose_node, mul0_node, mul1_node, conv_node],
        "layout_transform_const_folding",
        [input0, input1],
        [output0, output1],
        initializer=[const_1_weight, conv_weight],
    )
    opset_imports = [
        onnx.helper.make_opsetid("", 19),
    ]
    f32_model = onnx.helper.make_model(graph, opset_imports=opset_imports)

    print("[INFO]: Running onnx.checker on f32 model")
    f32_model = onnx.shape_inference.infer_shapes(f32_model)
    onnx.checker.check_model(f32_model, True)
    f32_model_path = "layout_transform_const_folding.f32.onnx"

    print(f"[INFO]: Saving {f32_model_path}")
    onnx.save_model(f32_model, f32_model_path)

    # Quantize model
    qdq_model_path = "layout_transform_const_folding.qdq.onnx"
    print("[INFO]: Creating QDQ model")
    quantize_static(
        f32_model_path,
        qdq_model_path,
        DataReader(f32_model_path),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=[node.op_type for node in f32_model.graph.node],
        extra_options={"DedicatedQDQPair": True, "ForceQuantizeNoInputCheck": True},
    )
    quant_pre_process(qdq_model_path, qdq_model_path)
    qdq_model = onnx.load_model(qdq_model_path)
    onnx.checker.check_model(qdq_model, True)
    onnx.save_model(qdq_model, qdq_model_path)
    print(f"[INFO]: Created QDQ model {qdq_model_path}")
