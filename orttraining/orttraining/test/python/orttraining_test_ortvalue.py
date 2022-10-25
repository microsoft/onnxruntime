#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=W0212,C0114,C0116

import unittest
import copy
import sys
import numpy as np
from numpy.testing import assert_almost_equal
import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue as C_OrtValue, OrtValueVector
from onnxruntime.training.ortmodule import ORTModule, _utils
from onnxruntime.capi import _pybind_state as C
import torch
from torch._C import _from_dlpack
from torch.utils.dlpack import from_dlpack
import _test_helpers


has_cuda = torch.cuda.is_available()


# Revoved code from _utils.


def C_ort_from_dlpack(dlpack):
    if hasattr(C, "ort_from_dlpack"):
        return C.ort_from_dlpack(dlpack)
    return from_dlpack(dlpack)


def _torch_tensor_from_dl_pack(dlpack, ortvalue, device):
    torch_tensor = from_dlpack(dlpack) if device.type != "ort" else C_ort_from_dlpack(dlpack)
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == "tensor(bool)" else torch_tensor


def _ortvalue_to_torch_tensor(ortvalue, device):
    dlpack_tensor = ortvalue.to_dlpack()
    return _torch_tensor_from_dl_pack(dlpack_tensor, ortvalue, device)


def _ortvalues_to_torch_tensor(vect, device):
    return [_ortvalue_to_torch_tensor(ov, device) for ov in vect]


class TestOrtValue(unittest.TestCase):
    def testOrtValueDlPack_float32(self):
        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

    def testOrtValueDlPack_bool(self):
        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.bool)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        self.assertTrue(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, True)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        self.assertTrue(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, True)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

    def testOrtValueVector_float32(self):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=np.float32),
        ]
        vect = OrtValueVector()
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a)
            vect.push_back(ortvalue._ortvalue)
        self.assertEqual(len(vect.bool_tensor_indices()), 0)
        self.assertEqual(len(vect), 2)
        for i, (ov, ar) in enumerate(zip(vect, narrays)):
            ovar = ov.numpy()
            assert_almost_equal(ar, ovar)
            self.assertEqual(ov.element_type(), vect.element_type_at(i))

    def testOrtValueVector_bool(self):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.bool_),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=np.bool_),
        ]
        vect = OrtValueVector()
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a)
            vect.push_back(ortvalue._ortvalue)
        self.assertEqual(vect.bool_tensor_indices(), [0, 1])
        self.assertEqual(len(vect), 2)
        for ov, ar in zip(vect, narrays):
            ovar = ov.numpy()
            assert_almost_equal(ar, ovar)

    def OrtValueVectorDlPackOrtValue(self, my_to_tensor, tensor_type, device, dtype=np.float32):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=dtype),
        ]
        vect = OrtValueVector()
        ptr = []
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a, device)
            vect.push_back(ortvalue._ortvalue)
            ptr.append(ortvalue.data_ptr())
        self.assertEqual(len(vect), 2)

        converted_values = vect.to_dlpacks(my_to_tensor)
        self.assertEqual(len(converted_values), len(vect))
        if my_to_tensor is None:
            self.assertIn("PyCapsule", str(type(converted_values[0])))
            converted_values = [C_OrtValue.from_dlpack(o, False) for o in converted_values]
        else:
            assert all(map(lambda v: isinstance(v, tensor_type), converted_values))

        # We make sure the function does not leak any python object.
        cf = [sys.getrefcount(o) for o in converted_values]
        dummy = [np.array([[0, 1]]), dict(a=3)]
        cf2 = [sys.getrefcount(o) for o in dummy]
        self.assertEqual(cf, cf2)  # it should be [3, 3]

        ptr2 = []
        for av1, v2 in zip(narrays, converted_values):
            ptr2.append(v2.data_ptr())
            if hasattr(v2, "cpu"):
                av2 = v2.cpu().numpy()
            else:
                av2 = v2.numpy()
            assert_almost_equal(av1, av2)
        self.assertEqual(ptr, ptr2)

    def testOrtValueVectorDlPackOrtValue_cpu(self):
        def my_to_tensor(dlpack_structure):
            return C_OrtValue.from_dlpack(dlpack_structure, False)

        self.OrtValueVectorDlPackOrtValue(my_to_tensor, C_OrtValue, "cpu")

    def testOrtValueVectorDlPackTorch_cpu(self):
        def my_to_tensor(dlpack_structure):
            return from_dlpack(dlpack_structure)

        self.OrtValueVectorDlPackOrtValue(my_to_tensor, torch.Tensor, "cpu")

    def testOrtValueVectorDlPack_Torch_cpu(self):
        def my_to_tensor(dlpack_structure):
            return _from_dlpack(dlpack_structure)

        self.OrtValueVectorDlPackOrtValue(my_to_tensor, torch.Tensor, "cpu")

    def testOrtValueVectorDlPackNone_cpu(self):
        self.OrtValueVectorDlPackOrtValue(None, None, "cpu")

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def testOrtValueVectorDlPackOrtValue_cuda(self):
        def my_to_tensor(dlpack_structure):
            return C_OrtValue.from_dlpack(dlpack_structure, False)

        self.OrtValueVectorDlPackOrtValue(my_to_tensor, C_OrtValue, "cuda")

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def testOrtValueVectorDlPackTorch_cuda(self):
        def my_to_tensor(dlpack_structure):
            return from_dlpack(dlpack_structure)

        self.OrtValueVectorDlPackOrtValue(my_to_tensor, torch.Tensor, "cuda")

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def testOrtValueVectorDlPack_Torch_cuda(self):
        def my_to_tensor(dlpack_structure):
            return _from_dlpack(dlpack_structure)

        self.OrtValueVectorDlPackOrtValue(my_to_tensor, torch.Tensor, "cuda")

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def testOrtValueVectorDlPackNone_cuda(self):
        self.OrtValueVectorDlPackOrtValue(None, None, "cuda")

    def test_ortmodule_dlpack(self):
        class NeuralNetTanh(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NeuralNetTanh, self).__init__()

                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.tanh = torch.nn.Tanh()

            def forward(self, input1):
                out = self.fc1(input1)
                out = self.tanh(out)
                return out

        def run_step(model, x):
            prediction = model(x)
            loss = prediction.sum()
            loss.backward()
            return prediction, loss

        N, D_in, H, D_out = 120, 1536, 500, 1536
        pt_model = NeuralNetTanh(D_in, H, D_out)
        ort_model = ORTModule(copy.deepcopy(pt_model))

        for step in range(10):
            pt_x = torch.randn(N, D_in, device="cpu", requires_grad=True)
            ort_x = copy.deepcopy(pt_x)
            ort_prediction, ort_loss = run_step(ort_model, ort_x)
            pt_prediction, pt_loss = run_step(pt_model, pt_x)
            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-4)
            _test_helpers.assert_values_are_close(ort_x.grad, pt_x.grad)
            _test_helpers.assert_values_are_close(ort_loss, pt_loss, atol=1e-4)

    def test_bool_input_and_output(self):
        class NeuralNetBoolInputOutput(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super(NeuralNetBoolInputOutput, self).__init__()
                self.fc = torch.nn.Linear(input_size, num_classes)
                self.relu = torch.nn.ReLU()

            def forward(self, condition, x1, x2):
                out1 = self.relu(self.fc(torch.where(condition, x1, x2)))
                out2 = torch.tensor(out1).to(torch.bool)
                return out1, out2

        device = "cpu"
        N, D_in, D_out = 8, 16, 2
        model = NeuralNetBoolInputOutput(D_in, D_out).to(device)
        model = ORTModule(model)
        condition = torch.randint(2, (N, D_in), dtype=torch.bool, device=device)
        x1 = torch.randn(N, D_in, device=device)
        x2 = torch.randn(N, D_in, device=device)
        y1, y2 = model(condition, x1, x2)

        assert y1 is not None
        assert y2 is not None and y2.dtype == torch.bool

    def _ortvalues_to_torch_tensor_ortvaluevector(self, device, tensor_type, new_impl, dtype=np.float32):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=dtype),
        ]
        vect = OrtValueVector()
        ptr = []
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a, device.type if device.type != "ort" else "cpu")
            vect.push_back(ortvalue._ortvalue)
            ptr.append(ortvalue.data_ptr())
        self.assertEqual(len(vect), 2)
        if new_impl == "list":
            raise AssertionError("Conversion from list to torch is not supported anymore.")
        if new_impl:
            tensors = _utils._ortvalues_to_torch_tensor(vect, device)
        else:
            tensors = _ortvalues_to_torch_tensor(vect, device)
        self.assertEqual(len(tensors), len(vect))
        for t in tensors:
            assert isinstance(t, torch.Tensor)
        self.assertEqual(ptr, [t.data_ptr() for t in tensors])
        assert all(map(lambda v: isinstance(v, tensor_type), tensors))

    def test_ortvalues_to_torch_tensor_ortvaluevector_cpu_new(self):
        device = torch.device("cpu")
        self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, True)

    def test_ortvalues_to_torch_tensor_ortvaluevector_cpu_old(self):
        device = torch.device("cpu")
        self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, False)

    def test_ortvalues_to_torch_tensor_ortvaluevector_ort_new(self):
        device = torch.device("ort")
        if hasattr(C, "ort_from_dlpack"):
            self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, True)
        else:
            with self.assertRaises(AttributeError):
                self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, True)

    def test_ortvalues_to_torch_tensor_ortvaluevector_ort_old(self):
        device = torch.device("ort")
        self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, False)

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def test_ortvalues_to_torch_tensor_ortvaluevector_cuda_new(self):
        device = torch.device("cuda:0")
        self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, True)

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def test_ortvalues_to_torch_tensor_ortvaluevector_cuda_old(self):
        device = torch.device("cuda:0")
        self._ortvalues_to_torch_tensor_ortvaluevector(device, torch.Tensor, False)

    def _ortvalues_to_torch_tensor_list(self, device, tensor_type, new_impl):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=np.float32),
        ]
        vect = C.OrtValueVector()
        vect.reserve(len(narrays))
        ptr = []
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a, device.type if device.type != "ort" else "cpu")
            vect.push_back(ortvalue._ortvalue)
            ptr.append(ortvalue.data_ptr())
        self.assertEqual(len(vect), 2)
        if new_impl:
            tensors = _utils._ortvalues_to_torch_tensor(vect, device)
        else:
            tensors = _ortvalues_to_torch_tensor(vect, device)
        self.assertEqual(len(tensors), len(vect))
        self.assertEqual(ptr, [t.data_ptr() for t in tensors])
        assert all(map(lambda v: isinstance(v, tensor_type), tensors))

    def test_ortvalues_to_torch_tensor_list_cpu_new(self):
        device = torch.device("cpu")
        self._ortvalues_to_torch_tensor_list(device, torch.Tensor, True)

    def test_ortvalues_to_torch_tensor_list_cpu_old(self):
        device = torch.device("cpu")
        self._ortvalues_to_torch_tensor_list(device, torch.Tensor, False)

    def test_ortvalues_to_torch_tensor_list_ort_new(self):
        device = torch.device("ort")
        if hasattr(C, "ort_from_dlpack"):
            self._ortvalues_to_torch_tensor_list(device, torch.Tensor, True)
        else:
            with self.assertRaises(AttributeError):
                self._ortvalues_to_torch_tensor_list(device, torch.Tensor, True)

    def test_ortvalues_to_torch_tensor_list_ort_old(self):
        device = torch.device("ort")
        self._ortvalues_to_torch_tensor_list(device, torch.Tensor, False)

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def test_ortvalues_to_torch_tensor_list_cuda_new(self):
        device = torch.device("cuda:0")
        self._ortvalues_to_torch_tensor_list(device, torch.Tensor, True)

    @unittest.skipIf(not has_cuda, reason="No CUDA availabled.")
    def test_ortvalues_to_torch_tensor_list_cuda_old(self):
        device = torch.device("cuda:0")
        self._ortvalues_to_torch_tensor_list(device, torch.Tensor, False)

    def test_element_type(self):
        values = {
            np.bool_: ("tensor(bool)", 9),
            np.int8: ("tensor(int8)", 3),
            np.uint8: ("tensor(uint8)", 2),
            np.int16: ("tensor(int16)", 5),
            np.uint16: ("tensor(uint16)", 4),
            np.int32: ("tensor(int32)", 6),
            np.uint32: ("tensor(uint32)", 12),
            np.int64: ("tensor(int64)", 7),
            np.uint64: ("tensor(uint64)", 13),
            np.float16: ("tensor(float16)", 10),
            np.float32: ("tensor(float)", 1),
            np.float64: ("tensor(double)", 11),
        }
        for dt, expected in values.items():
            numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dt)
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
            stype = ortvalue.data_type()
            self.assertIn(stype, expected)
            proto_type = ortvalue.element_type()
            self.assertIn(proto_type, expected)


if __name__ == "__main__":
    unittest.main()
