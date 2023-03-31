# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=W0622

"""Test export of PyTorch operators using ONNX Runtime contrib ops."""

import copy
import io
import unittest

import numpy as np
import onnx
import parameterized
import torch

import onnxruntime
from onnxruntime.tools import pytorch_export_contrib_ops


def _torch_version_lower_than(version: str):
    from packaging.version import Version as LooseVersion  # pylint: disable=C0415

    return LooseVersion(torch.__version__) < LooseVersion(version)


def ort_test_with_input(ort_sess, input, output, rtol, atol):
    input, _ = torch.jit._flatten(input)
    output, _ = torch.jit._flatten(output)

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

    inputs = list(map(to_numpy, input))
    outputs = list(map(to_numpy, output))

    ort_inputs = {ort_sess.get_inputs()[i].name: input for i, input in enumerate(inputs)}
    ort_outs = ort_sess.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


# These set of tests verify ONNX model export and compares outputs between
# PyTorch and ORT.
class ONNXExporterTest(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version

    opset_version = _export_onnx_opset_version
    keep_initializers_as_inputs = True  # For IR version 3 type export.

    def setUp(self):
        torch.manual_seed(0)
        pytorch_export_contrib_ops.register()

    def run_test(
        self,
        model,
        input=None,
        custom_opsets=None,
        batch_size=2,
        rtol=0.001,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        test_with_inputs=None,
        input_names=None,
        output_names=None,
    ):
        model.eval()

        if input is None:
            input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

        with torch.no_grad():
            if isinstance(input, torch.Tensor):
                input = (input,)
            # In-place operators will update input tensor data as well.
            # Thus inputs are replicated before every forward call.
            input_copy = copy.deepcopy(input)
            output = model(*input_copy)
            if isinstance(output, torch.Tensor):
                output = (output,)

            # export the model to ONNX
            f = io.BytesIO()
            torch.onnx.export(
                model,
                input_copy,
                f,
                opset_version=self.opset_version,
                do_constant_folding=do_constant_folding,
                keep_initializers_as_inputs=self.keep_initializers_as_inputs,
                dynamic_axes=dynamic_axes,
                input_names=input_names,
                output_names=output_names,
                custom_opsets=custom_opsets,
            )

            # compute onnxruntime output prediction
            ort_sess = onnxruntime.InferenceSession(f.getvalue(), providers=onnxruntime.get_available_providers())
            input_copy = copy.deepcopy(input)
            ort_test_with_input(ort_sess, input_copy, output, rtol, atol)

            # if additional test inputs are provided run the onnx
            # model with these inputs and check the outputs
            if test_with_inputs is not None:
                for test_input in test_with_inputs:
                    if isinstance(test_input, torch.Tensor):
                        test_input = (test_input,)  # noqa: PLW2901
                    test_input_copy = copy.deepcopy(test_input)
                    output = model(*test_input_copy)
                    if isinstance(output, torch.Tensor):
                        output = (output,)
                    ort_test_with_input(ort_sess, test_input, output, rtol, atol)

    def test_inverse(self):
        class CustomInverse(torch.nn.Module):
            def forward(self, x):
                return torch.inverse(x) + x

        x = torch.randn(2, 3, 3)
        self.run_test(CustomInverse(), x, custom_opsets={"com.microsoft": 1})

    def test_gelu(self):
        model = torch.nn.GELU()
        x = torch.randn(3, 3)
        self.run_test(model, x, custom_opsets={"com.microsoft": 1})

    def test_gelu_is_fused_by_default(self):
        model = torch.nn.GELU()

        f = io.BytesIO()
        torch.onnx.export(
            model,
            torch.randn(3, 3),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        node = onnx_model.graph.node[0]
        self.assertEqual(node.op_type, "Gelu")
        self.assertEqual(node.domain, "com.microsoft")

    @parameterized.parameterized.expand([("default_approximate", "none"), ("tanh_approximate", "tanh")])
    @unittest.skipIf(_torch_version_lower_than("1.12"), "Gelu's approximate parameter unsupported in PyTorch < 1.12")
    def test_gelu_supports_approximate_param(self, _, approximate: str):
        # The approximate param was introduced in PyTorch 1.12.
        # So we need to ignore the type checking when calling nn.Gelu
        model = torch.nn.GELU(approximate=approximate)  # type: ignore[call-arg]
        x = torch.randn(3, 3)
        self.run_test(model, x, custom_opsets={"com.microsoft": 1})

    def test_triu(self):
        for i in range(-5, 5):

            class Module(torch.nn.Module):
                def forward(self, input):
                    return input.triu(diagonal=i)  # noqa: B023

            model = Module()
            x = torch.randn(5, 4, 7, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(5, 4, 0, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(5, 0, 0, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

        for i in range(-5, 5):

            class Module2D(torch.nn.Module):
                def forward(self, input):
                    return input.triu(diagonal=i)  # noqa: B023

            model = Module2D()
            x = torch.randn(4, 7, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(0, 7, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(0, 0, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

    def test_tril(self):
        for i in range(-5, 5):

            class Module(torch.nn.Module):
                def forward(self, input):
                    return input.tril(diagonal=i)  # noqa: B023

            model = Module()
            x = torch.randn(5, 4, 7, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(5, 4, 0, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(5, 0, 0, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

        for i in range(-5, 5):

            class Module2D(torch.nn.Module):
                def forward(self, input):
                    return input.tril(diagonal=i)  # noqa: B023

            model = Module2D()
            x = torch.randn(4, 7, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(0, 7, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})

            x = torch.randn(0, 0, dtype=torch.float32)
            self.run_test(model, x, custom_opsets={"com.microsoft": 1})


# opset 9 tests, with keep_initializers_as_inputs=False for
# IR version 4 style export.
ONNXExporterTest_opset9_IRv4 = type(
    "TestONNXRuntime_opset9_IRv4",
    (unittest.TestCase,),
    dict(ONNXExporterTest.__dict__, keep_initializers_as_inputs=False),
)


if __name__ == "__main__":
    unittest.main()
