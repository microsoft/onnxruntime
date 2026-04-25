# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import tempfile
import unittest
import warnings


def ignore_warnings(warns):
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


class TestQuantIssues(unittest.TestCase):
    @ignore_warnings(DeprecationWarning)
    def test_minimal_model(self):
        folder = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
        onnx_path = os.path.join(folder, "qdq_minimal_model.onnx")
        if not os.path.exists(onnx_path):
            # The file does seem to be the same location in every CI job.
            raise unittest.SkipTest("unable to find {onnx_path!r}")

        import numpy as np  # noqa: PLC0415

        import onnxruntime.quantization as oq  # noqa: PLC0415

        class Mock:
            def __init__(self):
                self.i = 0

            def get_next(self):
                if self.i > 10:
                    return None
                self.i += 1
                return {"input": np.random.randint(0, 255, size=(1, 3, 32, 32), dtype=np.uint8)}

        with tempfile.TemporaryDirectory() as temp:
            preprocessed_path = os.path.join(temp, "preprocessed.onnx")
            quantized_path = os.path.join(temp, "quantized.onnx")
            oq.quant_pre_process(onnx_path, preprocessed_path, skip_symbolic_shape=True)
            oq.quantize_static(
                preprocessed_path,
                quantized_path,
                Mock(),
                calibrate_method=oq.CalibrationMethod.Percentile,
                op_types_to_quantize=["Conv", "Mul", "Gemm"],
            )
            assert os.path.exists(preprocessed_path), f"missing output {preprocessed_path!r}"
            assert os.path.exists(quantized_path), f"missing output {quantized_path!r}"

    def test_dynamic_quantize_per_channel_emits_axis_attribute(self):
        """Per-channel dynamic quantization must emit axis on DequantizeLinear nodes.

        Regression test for https://github.com/microsoft/onnxruntime/issues/19997.
        `quantize_dynamic(per_channel=True)` previously constructed QuantizedValue
        with axis=None and built DequantizeLinear without an axis attribute, producing
        an invalid per-tensor dequantization for per-channel quantized weights.
        When the per-channel quantized weight also appears as a graph output,
        _dequantize_outputs calls _dequantize_value, which triggered an assertion
        error (scale not scalar) and would have emitted a DequantizeLinear lacking
        the required axis attribute.
        """
        try:
            import numpy as np  # noqa: PLC0415
            import onnx  # noqa: PLC0415
            from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

            from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415
        except ImportError as exc:
            raise unittest.SkipTest(f"Required import missing: {exc}") from exc

        # Build a model: input (5, 4) @ weight (4, 8) -> output (5, 8).
        # The weight is also passed through Identity and exposed as a second graph
        # output so that _dequantize_outputs calls _dequantize_value on the
        # per-channel-quantized weight initializer.
        # Weight axis=1 is the output-feature axis (per-channel quantization target).
        np.random.seed(42)
        weight_data = np.random.normal(0, 0.1, (4, 8)).astype(np.float32)
        weight_init = numpy_helper.from_array(weight_data, name="weight")

        input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [5, 4])
        output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 8])
        weight_out_vi = helper.make_tensor_value_info("weight_out", TensorProto.FLOAT, [4, 8])

        matmul_node = helper.make_node("MatMul", ["input", "weight"], ["output"])
        identity_node = helper.make_node("Identity", ["weight"], ["weight_out"])

        graph = helper.make_graph(
            [matmul_node, identity_node],
            "test_graph",
            [input_vi],
            [output_vi, weight_out_vi],
            [weight_init],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        with tempfile.TemporaryDirectory() as tmp:
            model_fp_path = os.path.join(tmp, "model_fp.onnx")
            model_q_path = os.path.join(tmp, "model_q.onnx")
            onnx.save(model, model_fp_path)

            # This must not raise AssertionError due to per-channel scale not being scalar.
            quantize_dynamic(
                model_fp_path,
                model_q_path,
                per_channel=True,
                weight_type=QuantType.QInt8,
            )

            q_model = onnx.load(model_q_path)

        # Find the DequantizeLinear node that dequantizes the weight initializer.
        init_names = {init.name for init in q_model.graph.initializer}
        dq_nodes = [n for n in q_model.graph.node if n.op_type == "DequantizeLinear"]
        self.assertGreater(len(dq_nodes), 0, "Expected at least one DequantizeLinear node")

        weight_dq = None
        for node in dq_nodes:
            if node.input[0] in init_names:
                weight_dq = node
                break
        self.assertIsNotNone(weight_dq, "No DequantizeLinear node found with a weight initializer as input")

        # The axis attribute must be present.
        # MatMulInteger passes axis=-1 (last dimension) to quantize_weight_per_channel.
        axis_attrs = [attr for attr in weight_dq.attribute if attr.name == "axis"]
        self.assertEqual(len(axis_attrs), 1, "DequantizeLinear node is missing the 'axis' attribute")
        # MatMulInteger quantizes weight with axis=-1 (default in __quantize_inputs).
        self.assertEqual(axis_attrs[0].i, -1, f"Expected axis=-1, got axis={axis_attrs[0].i}")

        # The scale initializer must be 1-D with size > 1 (truly per-channel, not collapsed).
        scale_name = weight_dq.input[1]
        scale_init = next((i for i in q_model.graph.initializer if i.name == scale_name), None)
        self.assertIsNotNone(scale_init, f"Scale initializer '{scale_name}' not found")
        scale_array = numpy_helper.to_array(scale_init)
        self.assertEqual(scale_array.ndim, 1, f"Expected 1-D scale, got shape {scale_array.shape}")
        self.assertGreater(scale_array.size, 1, "Scale has only one element; expected per-channel scale")


if __name__ == "__main__":
    unittest.main(verbosity=2)
