from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).parent.resolve(strict=True)
TEST = False

if TEST:
    import onnxruntime


def generate_fuse_pad_avgpool():
    parameters = {
        "fuse-pad-avgpool": (
            {},
            [[1.333333, 2.333333, 1.777778], [3.0, 5.0, 3.666667], [2.666667, 4.333333, 3.111111]],
        ),
        "fuse-pad-avgpool_with_pad": (
            {"pads": [1, 1, 0, 0], "count_include_pad": 1},
            [
                [0.111111, 0.333333, 0.666667, 0.555556],
                [0.555556, 1.333333, 2.333333, 1.777778],
                [1.333333, 3.0, 5.0, 3.666667],
                [1.222222, 2.666667, 4.333333, 3.111111],
            ],
        ),
        "fuse-pad-avgpool_with_pad-nofuse": (
            {"pads": [1, 1, 0, 0]},
            [
                [0.25, 0.5, 1.0, 0.833333],
                [0.833333, 1.333333, 2.333333, 1.777778],
                [2.0, 3.0, 5.0, 3.666667],
                [1.833333, 2.666667, 4.333333, 3.111111],
            ],
        ),
    }
    for name in parameters:
        model_path = HERE / f"{name}.onnx"
        input_ = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (1, 1, 3, 3))
        pad = onnx.helper.make_node("Pad", ["input"], ["tp"], mode="constant", pads=[0, 0, 1, 1, 0, 0, 1, 1])
        pool = onnx.helper.make_node("AveragePool", ["tp"], ["output"], kernel_shape=[3, 3], **parameters[name][0])
        nodes = [pad, pool]
        output_shape = (1, 1, 3, 3) if name == "fuse-pad-avgpool" else (1, 1, 4, 4)
        output_ = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
        graph = onnx.helper.make_graph(nodes, name, [input_], [output_])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 7)])
        onnx.checker.check_model(model)
        onnx.save_model(model, model_path)
        if TEST:
            input_array = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
            expected = np.array(parameters[name][1], dtype=np.float32)
            session_options = onnxruntime.SessionOptions()
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = onnxruntime.InferenceSession(model_path, session_options)
            out = session.run(["output"], {"input": input_array})
            actual = out[0].squeeze()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0.0)
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = onnxruntime.InferenceSession(model_path, session_options)
            out = session.run(["output"], {"input": input_array})
            actual = out[0].squeeze()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0.0)


if __name__ == "__main__":
    generate_fuse_pad_avgpool()
