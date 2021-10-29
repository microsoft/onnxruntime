#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import copy
import numpy as np
from numpy.testing import assert_almost_equal
import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue as C_OrtValue, OrtValueVector
import torch
from onnxruntime.training.ortmodule import ORTModule


class TestOrtValue(unittest.TestCase):

    def testOrtValueDlPack(self):
        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        assert_almost_equal(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

    def testOrtValueVector(self):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=np.float32)]
        vect = OrtValueVector()
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a)
            vect.push_back(ortvalue._ortvalue)
        self.assertEqual(len(vect), 2)
        for ov, ar in zip(vect, narrays):
            ovar = ov.numpy()
            assert_almost_equal(ar, ovar)

    def testOrtValueVectorDlPack(self):
        narrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[6.0, 7.0], [8.0, 9.0], [1.0, 6.0]], dtype=np.float32)]
        vect = OrtValueVector()
        ptr = []
        for a in narrays:
            ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(a)
            vect.push_back(ortvalue._ortvalue)
            ptr.append(ortvalue.data_ptr())
        self.assertEqual(len(vect), 2)

        def my_to_tensor(dlpack_structure):
            return C_OrtValue.from_dlpack(dlpack_structure, False)

        ortvalues = vect.to_dlpack(my_to_tensor)
        self.assertEqual(len(ortvalues), len(vect))

        ptr2 = []
        for av1, v2 in zip(narrays, ortvalues):
            ptr2.append(v2.data_ptr())
            av2 = v2.numpy()
            assert_almost_equal(av1, av2)
        self.assertEqual(ptr, ptr2)

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

        for step in range(2):
            pt_x = torch.randn(N, D_in, device=device, requires_grad=True)
            ort_x = copy.deepcopy(pt_x)
            ort_prediction, ort_loss = run_step(ort_model, ort_x)
            pt_prediction, pt_loss = run_step(pt_model, pt_x)
            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
            _test_helpers.assert_values_are_close(ort_x.grad, pt_x.grad)
            _test_helpers.assert_values_are_close(ort_loss, pt_loss)


if __name__ == "__main__":
    unittest.main()
