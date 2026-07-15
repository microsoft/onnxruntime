# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx


def main():
    inp_shape = (1, 2, 8, 8)
    input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape)
    output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, None)

    weight_data = [
        [[[-1.5, 0.0], [0.2, 1.5]], [[-1.5, 0.0], [0.2, 1.5]]],
        [[[-1.0, 0.0], [0.1333, 1.0]], [[-1.0, 0.0], [0.1333, 1.0]]],
    ]
    weight = onnx.numpy_helper.from_array(np.array(weight_data, dtype=np.float32), "weight")
    bias = onnx.numpy_helper.from_array(np.array([0.0, 0.0], dtype=np.float32), "bias")
    conv_node = onnx.helper.make_node("Conv", ["input_0", "weight", "bias"], ["output_0"], name="Conv0")
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
    onnx.save_model(model, "conv_default_attrs.onnx")


if __name__ == "__main__":
    main()
