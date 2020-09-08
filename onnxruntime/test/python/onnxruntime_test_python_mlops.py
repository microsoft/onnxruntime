# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import numpy as np
import onnxruntime as onnxrt
import threading
from helper import get_name


class TestInferenceSession(unittest.TestCase):

    def testZipMapStringFloat(self):
        sess = onnxrt.InferenceSession(get_name("zipmap_stringfloat.onnx"))
        x = np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(float)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Z")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(map(string,tensor(float)))')

        output_expected = [{
            'class2': 0.0,
            'class1': 1.0,
            'class3': 3.0
        }, {
            'class2': 23.0,
            'class1': 44.0,
            'class3': 11.0
        }]
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def testZipMapInt64Float(self):
        sess = onnxrt.InferenceSession(get_name("zipmap_int64float.onnx"))
        x = np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(float)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Z")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(map(int64,tensor(float)))')

        output_expected = [{10: 1.0, 20: 0.0, 30: 3.0}, {10: 44.0, 20: 23.0, 30: 11.0}]
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def testDictVectorizer(self):
        sess = onnxrt.InferenceSession(get_name("pipeline_vectorize.onnx"))
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "float_input")
        input_type = str(sess.get_inputs()[0].type)
        self.assertEqual(input_type, "map(int64,tensor(float))")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "variable1")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(float)")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [1, 1])

        # Python type
        x = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        xwrong = x.copy()
        xwrong["a"] = 5.6
        try:
            res = sess.run([output_name], {input_name: xwrong})
        except RuntimeError as e:
            self.assertIn("Unexpected key type  <class 'str'>, it cannot be linked to C type int64_t", str(e))

        # numpy type
        x = {np.int64(k): np.float32(v) for k, v in x.items()}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        x = {np.int64(k): np.float64(v) for k, v in x.items()}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        x = {np.int32(k): np.float64(v) for k, v in x.items()}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testLabelEncoder(self):
        sess = onnxrt.InferenceSession(get_name("LabelEncoder.onnx"))
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "input")
        input_type = str(sess.get_inputs()[0].type)
        self.assertEqual(input_type, "tensor(string)")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [1, 1])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "variable")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, "tensor(int64)")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [1, 1])

        # Array
        x = np.array([['4']])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[3]], dtype=np.int64)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        # Python type
        x = np.array(['4'], ndmin=2)
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([3], ndmin=2, dtype=np.int64)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        x = np.array(['4'], ndmin=2, dtype=np.object)
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([3], ndmin=2, dtype=np.int64)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_run_model_mlnet(self):
        available_providers = onnxrt.get_available_providers()

        # The Windows GPU CI pipeline builds the wheel with both CUDA and DML enabled and ORT does not support cases
        # where one node is asigned to CUDA and one node to DML as it doesn't have the data transfer capabilities to deal with
        # potentially different device memory. Hence, use a session with only DML and CPU (excluding CUDA) for this test as it breaks
        # with both CUDA and DML registered.
        if ('CUDAExecutionProvider' in available_providers and 'DmlExecutionProvider' in available_providers):
            sess = onnxrt.InferenceSession(get_name("mlnet_encoder.onnx"), None, ['DmlExecutionProvider', 'CPUExecutionProvider'])
        else:
            sess = onnxrt.InferenceSession(get_name("mlnet_encoder.onnx"))

        names = [_.name for _ in sess.get_outputs()]
        self.assertEqual(['C00', 'C12'], names)
        c0 = np.array([5.], dtype=np.float32).reshape(1, 1)

        c1 = np.array([b'A\0A\0', b"B\0B\0", b"C\0C\0"], np.void).reshape(1, 3)
        res = sess.run(None, {'C0': c0, 'C1': c1})
        mat = res[1]
        total = mat.sum()
        self.assertEqual(total, 2)
        self.assertEqual(list(mat.ravel()),
                         list(np.array([[[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.]]]).ravel()))

        # In memory, the size of each element is fixed and equal to the
        # longest element. We cannot use bytes because numpy is trimming
        # every final 0 for strings and bytes before creating the array
        # (to save space). It does not have this behaviour for void
        # but as a result, numpy does not know anymore the size
        # of each element, they all have the same size.
        c1 = np.array([b'A\0A\0\0', b"B\0B\0", b"C\0C\0"], np.void).reshape(1, 3)
        res = sess.run(None, {'C0': c0, 'C1': c1})
        mat = res[1]
        total = mat.sum()
        self.assertEqual(total, 0)

if __name__ == '__main__':
    unittest.main()
