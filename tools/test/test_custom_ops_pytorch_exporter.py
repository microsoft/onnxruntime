import torch
import onnxruntime
import numpy as np
import unittest
import io
import copy
from python.register_custom_ops_pytorch_exporter import register_custom_op


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

    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


# These set of tests verify ONNX model export and compare onnxruntime outputs to pytorch.
# To register custom ops and run the tests, you should set PYTHONPATH as:
# PYTHONPATH=<path_to_onnxruntime/tools> pytest -v test_custom_ops_pytorch_exporter.py
class ONNXExporterTest(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version
    keep_initializers_as_inputs = True  # For IR version 3 type export.

    def setUp(self):
        torch.manual_seed(0)
        register_custom_op()

    def run_test(self, model, input=None,
                 custom_opsets=None,
                 batch_size=2,
                 rtol=0.001, atol=1e-7,
                 do_constant_folding=True,
                 dynamic_axes=None, test_with_inputs=None,
                 input_names=None, output_names=None):
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
            torch.onnx.export(model, input_copy, f,
                              opset_version=self.opset_version,
                              example_outputs=output,
                              do_constant_folding=do_constant_folding,
                              keep_initializers_as_inputs=self.keep_initializers_as_inputs,
                              dynamic_axes=dynamic_axes,
                              input_names=input_names, output_names=output_names,
                              custom_opsets=custom_opsets)

            # compute onnxruntime output prediction
            ort_sess = onnxruntime.InferenceSession(f.getvalue())
            input_copy = copy.deepcopy(input)
            ort_test_with_input(ort_sess, input_copy, output, rtol, atol)

            # if additional test inputs are provided run the onnx
            # model with these inputs and check the outputs
            if test_with_inputs is not None:
                for test_input in test_with_inputs:
                    if isinstance(test_input, torch.Tensor):
                        test_input = (test_input,)
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
        self.run_test(CustomInverse(), x, custom_opsets={'com.microsoft': 1})

    def test_gelu(self):
        model = torch.nn.GELU()
        x = torch.randn(3, 3)
        self.run_test(model, x, custom_opsets={'com.microsoft': 1})


# opset 10 tests
TestONNXRuntime_opset10 = type(str("TestONNXRuntime_opset10"),
                               (unittest.TestCase,),
                               dict(ONNXExporterTest.__dict__, opset_version=10))

# opset 11 tests
ONNXExporterTest_opset11 = type(str("TestONNXRuntime_opset11"),
                                (unittest.TestCase,),
                                dict(ONNXExporterTest.__dict__, opset_version=11))

# opset 12 tests
ONNXExporterTest_opset12 = type(str("TestONNXRuntime_opset12"),
                                (unittest.TestCase,),
                                dict(ONNXExporterTest.__dict__, opset_version=12))

# opset 9 tests, with keep_initializers_as_inputs=False for
# IR version 4 style export.
ONNXExporterTest_opset9_IRv4 = type(str("TestONNXRuntime_opset9_IRv4"),
                                    (unittest.TestCase,),
                                    dict(ONNXExporterTest.__dict__,
                                         keep_initializers_as_inputs=False))

# opset 10 tests, with keep_initializers_as_inputs=False for
# IR version 4 style export.
ONNXExporterTest_opset10_IRv4 = type(str("TestONNXRuntime_opset10_IRv4"),
                                     (unittest.TestCase,),
                                     dict(ONNXExporterTest.__dict__, opset_version=10,
                                          keep_initializers_as_inputs=False))


# opset 11 tests, with keep_initializers_as_inputs=False for
# IR version 4 style export.
ONNXExporterTest_opset11_IRv4 = type(str("TestONNXRuntime_opset11_IRv4"),
                                     (unittest.TestCase,),
                                     dict(ONNXExporterTest.__dict__, opset_version=11,
                                          keep_initializers_as_inputs=False))

# opset 12 tests, with keep_initializers_as_inputs=False for
# IR version 4 style export.
ONNXExporterTest_opset12_IRv4 = type(str("TestONNXRuntime_opset12_IRv4"),
                                     (unittest.TestCase,),
                                     dict(ONNXExporterTest.__dict__, opset_version=12,
                                          keep_initializers_as_inputs=False))

if __name__ == '__main__':
    unittest.main()
