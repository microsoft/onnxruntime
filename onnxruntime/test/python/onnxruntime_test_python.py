# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import numpy as np
import onnxruntime as onnxrt
import threading


class TestInferenceSession(unittest.TestCase):

    def get_name(self, name):
        if os.path.exists(name):
            return name
        rel = os.path.join("testdata", name)
        if os.path.exists(rel):
            return rel
        this = os.path.dirname(__file__)
        data = os.path.join(this, "..", "testdata")
        res = os.path.join(data, name)
        if os.path.exists(res):
            return res
        raise FileNotFoundError(
            "Unable to find '{0}' or '{1}' or '{2}'".format(name, rel, res))

    def run_model(self, session_object, run_options):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = session_object.get_inputs()[0].name
        res = session_object.run([], {input_name: x}, run_options=run_options)
        output_expected = np.array(
            [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testModelSerialization(self):
        so = onnxrt.SessionOptions()
        so.log_verbosity_level = 1
        so.logid = "TestModelSerialization"
        so.optimized_model_filepath = "./PythonApiTestOptimizedModel.onnx"
        onnxrt.InferenceSession(self.get_name("mul_1.onnx"), sess_options=so)
        self.assertTrue(os.path.isfile(so.optimized_model_filepath))

    def testGetProviders(self):
        self.assertTrue(
            'CPUExecutionProvider' in onnxrt.get_available_providers())
        self.assertTrue('CPUExecutionProvider' in onnxrt.get_all_providers())
        sess = onnxrt.InferenceSession(self.get_name("mul_1.onnx"))
        self.assertTrue('CPUExecutionProvider' in sess.get_providers())

    def testSetProviders(self):
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            sess = onnxrt.InferenceSession(self.get_name("mul_1.onnx"))
            # confirm that CUDA Provider is in list of registered providers.
            self.assertTrue('CUDAExecutionProvider' in sess.get_providers())
            # reset the session and register only CPU Provider.
            sess.set_providers(['CPUExecutionProvider'])
            # confirm only CPU Provider is registered now.
            self.assertEqual(['CPUExecutionProvider'], sess.get_providers())

    def testInvalidSetProviders(self):
        with self.assertRaises(ValueError) as context:
            sess = onnxrt.InferenceSession(self.get_name("mul_1.onnx"))
            sess.set_providers(['InvalidProvider'])
        self.assertTrue('[\'InvalidProvider\'] does not contain a subset of available providers' in str(
            context.exception))

    def testRunModel(self):
        sess = onnxrt.InferenceSession(self.get_name("mul_1.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 2])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array(
            [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelFromBytes(self):
        with open(self.get_name("mul_1.onnx"), "rb") as f:
            content = f.read()
        sess = onnxrt.InferenceSession(content)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 2])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array(
            [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2(self):
        sess = onnxrt.InferenceSession(self.get_name("matmul_1.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2Contiguous(self):
        sess = onnxrt.InferenceSession(self.get_name("matmul_1.onnx"))
        x = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float32)[:,[1,0]]
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)
        xcontiguous = np.ascontiguousarray(x)
        rescontiguous = sess.run([output_name], {input_name: xcontiguous})
        np.testing.assert_allclose(
            output_expected, rescontiguous[0], rtol=1e-05, atol=1e-08)

    def testRunModelMultipleThreads(self):
        so = onnxrt.SessionOptions()
        so.log_verbosity_level = 1
        so.logid = "MultiThreadsTest"
        sess = onnxrt.InferenceSession(
            self.get_name("mul_1.onnx"), sess_options=so)
        ro1 = onnxrt.RunOptions()
        ro1.logid = "thread1"
        t1 = threading.Thread(target=self.run_model, args=(sess, ro1))
        ro2 = onnxrt.RunOptions()
        ro2.logid = "thread2"
        t2 = threading.Thread(target=self.run_model, args=(sess, ro2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def testRunDevice(self):
        device = onnxrt.get_device()
        self.assertTrue('CPU' in device or 'GPU' in device)

    def testRunModelSymbolicInput(self):
        sess = onnxrt.InferenceSession(self.get_name("matmul_2.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        # Input X has an unknown dimension.
        self.assertEqual(input_shape, ['None', 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        # Output X has an unknown dimension.
        self.assertEqual(output_shape, ['None', 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testBooleanInputs(self):
        sess = onnxrt.InferenceSession(self.get_name("logicaland.onnx"))
        a = np.array([[True, True], [False, False]], dtype=np.bool)
        b = np.array([[True, False], [True, False]], dtype=np.bool)

        # input1:0 is first in the protobuf, and input:0 is second
        # and we maintain the original order.
        a_name = sess.get_inputs()[0].name
        self.assertEqual(a_name, "input1:0")
        a_shape = sess.get_inputs()[0].shape
        self.assertEqual(a_shape, [2, 2])
        a_type = sess.get_inputs()[0].type
        self.assertEqual(a_type, 'tensor(bool)')

        b_name = sess.get_inputs()[1].name
        self.assertEqual(b_name, "input:0")
        b_shape = sess.get_inputs()[1].shape
        self.assertEqual(b_shape, [2, 2])
        b_type = sess.get_inputs()[0].type
        self.assertEqual(b_type, 'tensor(bool)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(bool)')

        output_expected = np.array(
            [[True, False], [False, False]], dtype=np.bool)
        res = sess.run([output_name], {a_name: a, b_name: b})
        np.testing.assert_equal(output_expected, res[0])

    def testStringInput1(self):
        sess = onnxrt.InferenceSession(self.get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'],
                     dtype=np.str).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0])

    def testStringInput2(self):
        sess = onnxrt.InferenceSession(self.get_name("identity_string.onnx"))
        x = np.array(['Olá', '你好', '여보세요', 'hello'],
                     dtype=np.unicode).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0])

    def testInputBytes(self):
        sess = onnxrt.InferenceSession(self.get_name("identity_string.onnx"))
        x = np.array([b'this', b'is', b'identity', b'test']).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0].astype('|S8'))

    def testInputObject(self):
        sess = onnxrt.InferenceSession(self.get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'],
                     object).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0])

    def testInputVoid(self):
        sess = onnxrt.InferenceSession(self.get_name("identity_string.onnx"))
        x = np.array([b'this', b'is', b'identity', b'test'],
                     np.void).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})

        expr = np.array([['this\x00\x00\x00\x00', 'is\x00\x00\x00\x00\x00\x00'],
                         ['identity', 'test\x00\x00\x00\x00']], dtype=object)
        np.testing.assert_equal(expr, res[0])

    def testZipMapStringFloat(self):
        sess = onnxrt.InferenceSession(
            self.get_name("zipmap_stringfloat.onnx"))
        x = np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0],
                     dtype=np.float32).reshape((2, 3))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(float)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Z")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(map(string,tensor(float)))')

        output_expected = [{'class2': 0.0, 'class1': 1.0, 'class3': 3.0},
                           {'class2': 23.0, 'class1': 44.0, 'class3': 11.0}]
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def testZipMapInt64Float(self):
        sess = onnxrt.InferenceSession(self.get_name("zipmap_int64float.onnx"))
        x = np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0],
                     dtype=np.float32).reshape((2, 3))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(float)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Z")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(map(int64,tensor(float)))')

        output_expected = [{10: 1.0, 20: 0.0, 30: 3.0},
                           {10: 44.0, 20: 23.0, 30: 11.0}]
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def testRaiseWrongNumInputs(self):
        with self.assertRaises(ValueError) as context:
            sess = onnxrt.InferenceSession(self.get_name("logicaland.onnx"))
            a = np.array([[True, True], [False, False]], dtype=np.bool)
            res = sess.run([], {'input:0': a})

        self.assertTrue('Model requires 2 inputs' in str(context.exception))

    def testModelMeta(self):
        model_path = "../models/opset8/test_squeezenet/model.onnx"
        if not os.path.exists(model_path):
            return
        sess = onnxrt.InferenceSession(model_path)
        modelmeta = sess.get_modelmeta()
        self.assertEqual('onnx-caffe2', modelmeta.producer_name)
        self.assertEqual('squeezenet_old', modelmeta.graph_name)
        self.assertEqual('', modelmeta.domain)
        self.assertEqual('', modelmeta.description)

    def testProfilerWithSessionOptions(self):
        so = onnxrt.SessionOptions()
        so.enable_profiling = True
        sess = onnxrt.InferenceSession(
            self.get_name("mul_1.onnx"), sess_options=so)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        sess.run([], {'X': x})
        profile_file = sess.end_profiling()

        tags = ['pid', 'dur', 'ts', 'ph', 'X', 'name', 'args']
        with open(profile_file) as f:
            lines = f.readlines()
            self.assertTrue('[' in lines[0])
            for i in range(1, 8):
                for tag in tags:
                    self.assertTrue(tag in lines[i])
            self.assertTrue(']' in lines[8])

    def testDictVectorizer(self):
        sess = onnxrt.InferenceSession(
            self.get_name("pipeline_vectorize.onnx"))
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
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

        xwrong = x.copy()
        xwrong["a"] = 5.6
        try:
            res = sess.run([output_name], {input_name: xwrong})
        except RuntimeError as e:
            self.assertIn(
                "Unexpected key type  <class 'str'>, it cannot be linked to C type int64_t", str(e))

        # numpy type
        x = {np.int64(k): np.float32(v) for k, v in x.items()}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

        x = {np.int64(k): np.float64(v) for k, v in x.items()}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

        x = {np.int32(k): np.float64(v) for k, v in x.items()}
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testLabelEncoder(self):
        sess = onnxrt.InferenceSession(self.get_name("LabelEncoder.onnx"))
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
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

        # Python type
        x = np.array(['4'], ndmin=2)
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([3], ndmin=2, dtype=np.int64)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

        x = np.array(['4'], ndmin=2, dtype=np.object)
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([3], ndmin=2, dtype=np.int64)
        np.testing.assert_allclose(
            output_expected, res[0], rtol=1e-05, atol=1e-08)

    def test_run_model_mlnet(self):
        sess = onnxrt.InferenceSession(self.get_name("mlnet_encoder.onnx"))
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
        c1 = np.array([b'A\0A\0\0', b"B\0B\0", b"C\0C\0"],
                      np.void).reshape(1, 3)
        res = sess.run(None, {'C0': c0, 'C1': c1})
        mat = res[1]
        total = mat.sum()
        self.assertEqual(total, 0)

    def testGraphOptimizationLevel(self):
        opt = onnxrt.SessionOptions()
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC)
            # default should be basic optimization
        opt.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL)
        sess = onnxrt.InferenceSession(self.get_name("logicaland.onnx"), sess_options=opt)
        a = np.array([[True, True], [False, False]], dtype=np.bool)
        b = np.array([[True, False], [True, False]], dtype=np.bool)

        res = sess.run([], {'input1:0': a, 'input:0':b})

    def testSequenceLength(self):
        sess = onnxrt.InferenceSession(self.get_name("sequence_length.onnx"))
        x = [np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3)),
        np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3))]

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'seq(tensor(float))')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(int64)')

        output_expected = np.array(2, dtype=np.int64)
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def testSequenceConstruct(self):
        sess = onnxrt.InferenceSession(
            self.get_name("sequence_construct.onnx"))

        self.assertEqual(sess.get_inputs()[0].type, 'tensor(int64)')
        self.assertEqual(sess.get_inputs()[1].type, 'tensor(int64)')

        self.assertEqual(sess.get_inputs()[0].name, "tensor1")
        self.assertEqual(sess.get_inputs()[1].name, "tensor2")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(tensor(int64))')

        output_expected = [np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
                           np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3))]

        res = sess.run([output_name], {"tensor1": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
                                       "tensor2": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3))})

        np.testing.assert_array_equal(output_expected, res[0])

    def testSequenceInsert(self):
        opt = onnxrt.SessionOptions()
        opt.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        sess = onnxrt.InferenceSession(self.get_name("sequence_insert.onnx"), sess_options=opt)

        self.assertEqual(sess.get_inputs()[0].type, 'seq(tensor(int64))')
        self.assertEqual(sess.get_inputs()[1].type, 'tensor(int64)')

        self.assertEqual(sess.get_inputs()[0].name, "input_seq")
        self.assertEqual(sess.get_inputs()[1].name, "tensor")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(tensor(int64))')

        output_expected = [
            np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3))]
        res = sess.run([output_name], {"tensor": np.array(
            [1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)), "input_seq": []})
        np.testing.assert_array_equal(output_expected, res[0])

    def testOrtExecutionMode(self):
        opt = onnxrt.SessionOptions()
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_SEQUENTIAL)
        opt.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_PARALLEL)

if __name__ == '__main__':
    unittest.main()
