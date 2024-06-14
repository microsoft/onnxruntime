"""
@brief      test log(time=3s)
"""

import copy
import unittest

import numpy as np
import torch

from onnxruntime.training.ortmodule import ORTModule


class TestOnnxOpsOrtModule(unittest.TestCase):
    def assert_values_are_close(self, tensor, other, rtol=1e-05, atol=1e-06):
        are_close = torch.allclose(tensor, other, rtol=rtol, atol=atol)
        if not are_close:
            abs_diff = torch.abs(tensor - other)
            abs_other = torch.abs(other)
            max_atol = torch.max(abs_diff - rtol * abs_other)
            max_rtol = torch.max((abs_diff - atol) / abs_other)
            raise AssertionError(f"The maximum atol is {max_atol!r}, maximum rtol is {max_rtol!r}.")

    def assert_gradients_match_and_reset_gradient(
        self, ort_model, pt_model, none_pt_params=None, reset_gradient=True, rtol=1e-05, atol=1e-06
    ):
        if none_pt_params is None:
            none_pt_params = []
        ort_named_params = list(ort_model.named_parameters())
        pt_named_params = list(pt_model.named_parameters())
        self.assertEqual(len(ort_named_params), len(pt_named_params))

        for ort_named_param, pt_named_param in zip(ort_named_params, pt_named_params):
            ort_name, ort_param = ort_named_param
            pt_name, pt_param = pt_named_param

            self.assertIn(pt_name, ort_name)
            if pt_name in none_pt_params:
                self.assertNotEmpty(pt_param.grad)
                if ort_param is not None:
                    self.assertFalse(torch.is_nonzero(torch.count_nonzero(ort_param.grad)))
            else:
                self.assert_values_are_close(ort_param.grad, pt_param.grad, rtol=rtol, atol=atol)

            if reset_gradient:
                ort_param.grad = None
                pt_param.grad = None

    def gradient_correctness(self, name, device, debug=False):
        pt_model_cls, op_grad_type, kwargs = self.get_torch_model_name(name, device)
        if kwargs is None:
            kwargs = {}
        N = 32  # noqa: N806
        pt_model = pt_model_cls().to(device)
        D_in = pt_model.fc1.in_features  # noqa: N806
        ort_model = ORTModule(copy.deepcopy(pt_model))

        def run_step(model, x):
            prediction = model(x)
            loss = prediction.sum()
            loss.backward()
            return prediction

        for _ in range(10):
            x = torch.randn(N, D_in, device=device)
            pt_prediction = run_step(pt_model, x)
            ort_prediction = run_step(ort_model, x)

            self.assert_values_are_close(ort_prediction, pt_prediction, **kwargs)
            self.assert_gradients_match_and_reset_gradient(ort_model, pt_model, **kwargs)

        onnx_graph_inf = ort_model._torch_module._execution_manager._training_manager._onnx_models.exported_model
        onnx_graph_train = ort_model._torch_module._execution_manager._training_manager._onnx_models.optimized_model
        if debug:
            with open("debug_%s_ortmodule_infer.onnx" % name, "wb") as f:
                f.write(onnx_graph_inf.SerializeToString())
            with open("debug_%s_ortmodule_train.onnx" % name, "wb") as f:
                f.write(onnx_graph_train.SerializeToString())
        self.assertIn('op_type: "%s"' % name, str(onnx_graph_inf))
        for onnx_model in [onnx_graph_inf, onnx_graph_train]:
            for oimp in onnx_model.opset_import:
                if oimp.domain == "":
                    self.assertEqual(oimp.version, 17)  # Needs to match latest default ORTModule opset
        if op_grad_type is not None:
            if isinstance(op_grad_type, tuple):
                text = str(onnx_graph_train)
                if all(map(lambda op: ('op_type: "%s"' % op) not in text, op_grad_type)):
                    raise AssertionError("Operator {} not found in {}.".format(" or ".join(op_grad_type), text))
            else:
                self.assertIn('op_type: "%s"' % op_grad_type, str(onnx_graph_train))

    def get_torch_model_name(self, name, device):
        def from_numpy(v, device=None, requires_grad=False):
            v = torch.from_numpy(v)
            if device is not None:
                v = v.to(device)
            v.requires_grad_(requires_grad)
            return v

        if name == "Softmax":

            class TestSoftmax(torch.nn.Module):
                def __init__(self, input_size=128, hidden_size=500, num_classes=100):
                    torch.nn.Module.__init__(self)
                    self.fc1 = torch.nn.Linear(input_size, hidden_size)
                    self.thfct = torch.nn.Softmax()
                    self.fc2 = torch.nn.Linear(hidden_size, num_classes)

                def forward(self, input1):
                    out = self.fc1(input1)
                    out = self.thfct(out)
                    out = self.fc2(out)
                    return out

            return TestSoftmax, ("SoftmaxGrad", "SoftmaxGrad_13"), None

        if name == "GatherElements":

            class TestGatherElement(torch.nn.Module):
                def __init__(self, input_size=32, hidden_size=500, num_classes=100):
                    torch.nn.Module.__init__(self)
                    self.fc1 = torch.nn.Linear(input_size, hidden_size)
                    rev_idx = np.array(list(np.arange(hidden_size)[::-1]), dtype=np.int64)
                    idx = np.empty((input_size, hidden_size), dtype=np.int64)
                    for i in range(idx.shape[0]):
                        idx[i, :] = rev_idx
                    self.indices = from_numpy(idx, device=device)
                    self.fc2 = torch.nn.Linear(hidden_size, num_classes)

                def forward(self, input1):
                    out = self.fc1(input1)
                    out = torch.gather(out, 1, self.indices)
                    out = self.fc2(out)
                    return out

            return TestGatherElement, "GatherElementsGrad", dict(rtol=1e-04, atol=1e-05)

        raise AssertionError("Unexpected name=%r." % name)

    def test_onnx_ops(self):
        for name in ["GatherElements", "Softmax"]:
            for device_name in ["cuda:0", "cpu"]:
                if device_name == "cuda:0" and not torch.cuda.is_available():
                    continue
                with self.subTest(name=name, device=device_name):
                    device = torch.device(device_name)
                    self.gradient_correctness(name, device)


if __name__ == "__main__":
    unittest.main()
