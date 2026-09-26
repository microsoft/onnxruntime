# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from parameterized import parameterized

import onnxruntime as ort
from onnxruntime.quantization.shape_inference import quant_pre_process


class TestQuantPreprocessOptionalStages(unittest.TestCase):
    @staticmethod
    def make_model():
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Identity", ["input"], ["identity"]),
                onnx.helper.make_node("Add", ["identity", "bias"], ["output"]),
            ],
            "preprocess_optional_stages",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
            [onnx.numpy_helper.from_array(np.array([[1, 2, 3, 4]], dtype=np.float32), "bias")],
        )
        return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=9)

    @parameterized.expand(list(itertools.product([False, True], repeat=4)))
    def test_optional_stages_preserve_optimization_and_output(
        self, skip_symbolic_shape, skip_onnx_shape, skip_optimization, use_model_proto
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output_path = root / "input.onnx", root / "output.onnx"
            model = self.make_model()
            onnx.save_model(model, input_path)
            before = input_path.read_bytes()
            quant_pre_process(
                model if use_model_proto else input_path,
                output_path,
                skip_symbolic_shape=skip_symbolic_shape,
                skip_onnx_shape=skip_onnx_shape,
                skip_optimization=skip_optimization,
            )
            # The preprocessing temporary directory has already been removed.
            processed = onnx.load_model(output_path)
            onnx.checker.check_model(processed)
            self.assertEqual(
                [node.op_type for node in processed.graph.node], ["Identity", "Add"] if skip_optimization else ["Add"]
            )
            self.assertEqual(input_path.read_bytes(), before)
            metadata = {entry.key: entry.value for entry in processed.metadata_props}
            self.assertEqual(metadata.get("onnx.quant.pre_process"), "onnxruntime.quant")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(output_path), options, providers=["CPUExecutionProvider"])
            inputs = np.array([[0.5, -2.0, 10.0, 3.0]], dtype=np.float32)
            np.testing.assert_array_equal(
                session.run(None, {"input": inputs})[0], np.array([[1.5, 0.0, 13.0, 7.0]], dtype=np.float32)
            )

    @parameterized.expand(list(itertools.product([False, True], repeat=3)))
    def test_external_weights_survive_optional_shape_stages(
        self, skip_symbolic_shape, skip_onnx_shape, single_external_file
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, outputs = root / "inputs", root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            input_path, output_path = inputs / "model.onnx", outputs / "model.onnx"
            onnx.save_model(
                self.make_model(),
                input_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="input.data",
                size_threshold=0,
            )
            original_data = (inputs / "input.data").read_bytes()
            quant_pre_process(
                input_path,
                output_path,
                skip_symbolic_shape=skip_symbolic_shape,
                skip_onnx_shape=skip_onnx_shape,
                save_as_external_data=True,
                all_tensors_to_one_file=single_external_file,
                external_data_location="weights.data" if single_external_file else None,
                external_data_size_threshold=0,
            )
            processed = onnx.load_model(output_path)
            self.assertEqual([node.op_type for node in processed.graph.node], ["Add"])
            np.testing.assert_array_equal(onnx.numpy_helper.to_array(processed.graph.initializer[0]), [[1, 2, 3, 4]])
            self.assertEqual((inputs / "input.data").read_bytes(), original_data)
            onnx.checker.check_model(str(output_path))
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            np.testing.assert_array_equal(session.run(None, {"input": np.zeros((1, 4), np.float32)})[0], [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
