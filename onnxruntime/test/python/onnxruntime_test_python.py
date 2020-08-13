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

    def run_model(self, session_object, run_options):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = session_object.get_inputs()[0].name
        res = session_object.run([], {input_name: x}, run_options=run_options)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testModelSerialization(self):
        so = onnxrt.SessionOptions()
        so.log_verbosity_level = 1
        so.logid = "TestModelSerialization"
        so.optimized_model_filepath = "./PythonApiTestOptimizedModel.onnx"
        onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
        self.assertTrue(os.path.isfile(so.optimized_model_filepath))

    def testGetProviders(self):
        self.assertTrue('CPUExecutionProvider' in onnxrt.get_available_providers())
        # get_all_providers() returns the default EP order from highest to lowest.
        # CPUExecutionProvider should always be last.
        self.assertTrue('CPUExecutionProvider' == onnxrt.get_all_providers()[-1])
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
        self.assertTrue('CPUExecutionProvider' in sess.get_providers())

    def testSetProviders(self):
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
            # confirm that CUDA Provider is in list of registered providers.
            self.assertTrue('CUDAExecutionProvider' in sess.get_providers())
            # reset the session and register only CPU Provider.
            sess.set_providers(['CPUExecutionProvider'])
            # confirm only CPU Provider is registered now.
            self.assertEqual(['CPUExecutionProvider'], sess.get_providers())

    def testSetProvidersWithOptions(self):
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            import sys
            import ctypes
            CUDA_SUCCESS = 0

            def runBaseTest1():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
                self.assertTrue('CUDAExecutionProvider' in sess.get_providers())

                option1 = {'device_id': 0}
                sess.set_providers(['CUDAExecutionProvider'], [option1])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())
                option2 = {'device_id': -1}
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option2])
                sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'], [option1, {}])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())

            def runBaseTest2():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
                self.assertTrue('CUDAExecutionProvider' in sess.get_providers())

                # test get/set of "cuda_mem_limit" configuration.
                options = sess.get_provider_options()
                self.assertTrue('CUDAExecutionProvider' in options)
                option = options['CUDAExecutionProvider']
                self.assertTrue('cuda_mem_limit' in option)
                ori_mem_limit = option['cuda_mem_limit']
                new_mem_limit = int(ori_mem_limit) // 2
                option['cuda_mem_limit'] = new_mem_limit
                sess.set_providers(['CUDAExecutionProvider'], [option])
                options = sess.get_provider_options()
                self.assertEqual(options['CUDAExecutionProvider']['cuda_mem_limit'], str(new_mem_limit))

                option['cuda_mem_limit'] = ori_mem_limit 
                sess.set_providers(['CUDAExecutionProvider'], [option])
                options = sess.get_provider_options()
                self.assertEqual(options['CUDAExecutionProvider']['cuda_mem_limit'], ori_mem_limit)

                option['cuda_mem_limit'] = -1024
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                option['cuda_mem_limit'] = 1024.1024
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                option['cuda_mem_limit'] = 'wrong_value'
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])


                # test get/set of "arena_extend_strategy" configuration.
                options = sess.get_provider_options()
                self.assertTrue('CUDAExecutionProvider' in options)
                option = options['CUDAExecutionProvider']
                self.assertTrue('arena_extend_strategy' in option)
                for strategy in ['kNextPowerOfTwo', 'kSameAsRequested']:
                    option['arena_extend_strategy'] = strategy
                    sess.set_providers(['CUDAExecutionProvider'], [option])
                    options = sess.get_provider_options()
                    self.assertEqual(options['CUDAExecutionProvider']['arena_extend_strategy'], strategy)

                option['arena_extend_strategy'] = 'wrong_value'
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

            def getCudaDeviceCount():
                import ctypes

                num_device = ctypes.c_int()
                result = ctypes.c_int()
                error_str = ctypes.c_char_p()

                result = cuda.cuInit(0)
                result = cuda.cuDeviceGetCount(ctypes.byref(num_device))
                if result != CUDA_SUCCESS:
                    cuda.cuGetErrorString(result, ctypes.byref(error_str))
                    print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
                    return -1

                return num_device.value

            def setDeviceIdTest(i):
                import ctypes
                import onnxruntime as onnxrt

                device = ctypes.c_int()
                result = ctypes.c_int()
                error_str = ctypes.c_char_p()

                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
                option = {'device_id': i}
                sess.set_providers(['CUDAExecutionProvider'], [option])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())
                result = cuda.cuCtxGetDevice(ctypes.byref(device))
                if result != CUDA_SUCCESS:
                    cuda.cuGetErrorString(result, ctypes.byref(error_str))
                    print("cuCtxGetDevice failed with error code %d: %s" % (result, error_str.value.decode()))

                self.assertEqual(result, CUDA_SUCCESS)
                self.assertEqual(i, device.value)

            def runAdvancedTest():
                num_device = getCudaDeviceCount()
                if num_device < 0:
                    return 

                # Configure session to be ready to run on all available cuda devices
                for i in range(num_device):
                    setDeviceIdTest(i)

                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))

                # configure session with not legit option values and that shloud fail
                with self.assertRaises(RuntimeError):
                    option = {'device_id': num_device}
                    sess.set_providers(['CUDAExecutionProvider'], [option])
                    option = {'device_id': 'non_legit_value'}
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                # configure session with not legit option should cause no effect
                option = {'device_id': 0}
                sess.set_providers(['CUDAExecutionProvider'], [option])
                option = {'non_legit_option': num_device}
                sess.set_providers(['CUDAExecutionProvider'], [option])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())



            libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
            for libname in libnames:
                try:
                    cuda = ctypes.CDLL(libname)
                    runBaseTest1()
                    runBaseTest2()
                    runAdvancedTest()

                except OSError:
                    continue
                else:
                    break
            else:
                runBaseTest1()
                runBaseTest2()
                # raise OSError("could not load any of: " + ' '.join(libnames))

    def testInvalidSetProviders(self):
        with self.assertRaises(ValueError) as context:
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
            sess.set_providers(['InvalidProvider'])
        self.assertTrue(
            '[\'InvalidProvider\'] does not contain a subset of available providers' in str(context.exception))

    def testSessionProviders(self):
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            # create session from scratch, but constrain it to only use the CPU.
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=['CPUExecutionProvider'])
            self.assertEqual(['CPUExecutionProvider'], sess.get_providers())

    def testRunModel(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
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
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelFromBytes(self):
        with open(get_name("mul_1.onnx"), "rb") as f:
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
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"))
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2Contiguous(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"))
        x = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float32)[:, [1, 0]]
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
        xcontiguous = np.ascontiguousarray(x)
        rescontiguous = sess.run([output_name], {input_name: xcontiguous})
        np.testing.assert_allclose(output_expected, rescontiguous[0], rtol=1e-05, atol=1e-08)

    def testRunModelMultipleThreads(self):
        so = onnxrt.SessionOptions()
        so.log_verbosity_level = 1
        so.logid = "MultiThreadsTest"
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
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

    def testListAsInput(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x.tolist()})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testStringListAsInput(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'], dtype=np.str).reshape((2, 2))
        x_name = sess.get_inputs()[0].name
        res = sess.run([], {x_name: x.tolist()})
        np.testing.assert_equal(x, res[0])

    def testRunDevice(self):
        device = onnxrt.get_device()
        self.assertTrue('CPU' in device or 'GPU' in device)

    def testRunModelSymbolicInput(self):
        sess = onnxrt.InferenceSession(get_name("matmul_2.onnx"))
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
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testBooleanInputs(self):
        sess = onnxrt.InferenceSession(get_name("logicaland.onnx"))
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

        output_expected = np.array([[True, False], [False, False]], dtype=np.bool)
        res = sess.run([output_name], {a_name: a, b_name: b})
        np.testing.assert_equal(output_expected, res[0])

    def testStringInput1(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'], dtype=np.str).reshape((2, 2))

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
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['Olá', '你好', '여보세요', 'hello'], dtype=np.unicode).reshape((2, 2))

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
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
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
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'], object).reshape((2, 2))

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
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array([b'this', b'is', b'identity', b'test'], np.void).reshape((2, 2))

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

        expr = np.array([['this\x00\x00\x00\x00', 'is\x00\x00\x00\x00\x00\x00'], ['identity', 'test\x00\x00\x00\x00']],
                        dtype=object)
        np.testing.assert_equal(expr, res[0])

    def testRaiseWrongNumInputs(self):
        with self.assertRaises(ValueError) as context:
            sess = onnxrt.InferenceSession(get_name("logicaland.onnx"))
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
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
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

    def testGraphOptimizationLevel(self):
        opt = onnxrt.SessionOptions()
        # default should be all optimizations optimization
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL)
        opt.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
        sess = onnxrt.InferenceSession(get_name("logicaland.onnx"), sess_options=opt)
        a = np.array([[True, True], [False, False]], dtype=np.bool)
        b = np.array([[True, False], [True, False]], dtype=np.bool)

        res = sess.run([], {'input1:0': a, 'input:0': b})

    def testPrePacking(self):
        opt = onnxrt.SessionOptions()
        self.assertTrue(opt.use_prepacking)
        opt.use_prepacking = False
        self.assertFalse(opt.use_prepacking)

    def testSequenceLength(self):
        sess = onnxrt.InferenceSession(get_name("sequence_length.onnx"))
        x = [
            np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3)),
            np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3))
        ]

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
        sess = onnxrt.InferenceSession(get_name("sequence_construct.onnx"))

        self.assertEqual(sess.get_inputs()[0].type, 'tensor(int64)')
        self.assertEqual(sess.get_inputs()[1].type, 'tensor(int64)')

        self.assertEqual(sess.get_inputs()[0].name, "tensor1")
        self.assertEqual(sess.get_inputs()[1].name, "tensor2")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(tensor(int64))')

        output_expected = [
            np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3))
        ]

        res = sess.run(
            [output_name], {
                "tensor1": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
                "tensor2": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3))
            })

        np.testing.assert_array_equal(output_expected, res[0])

    def testSequenceInsert(self):
        opt = onnxrt.SessionOptions()
        opt.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        sess = onnxrt.InferenceSession(get_name("sequence_insert.onnx"), sess_options=opt)

        self.assertEqual(sess.get_inputs()[0].type, 'seq(tensor(int64))')
        self.assertEqual(sess.get_inputs()[1].type, 'tensor(int64)')

        self.assertEqual(sess.get_inputs()[0].name, "input_seq")
        self.assertEqual(sess.get_inputs()[1].name, "tensor")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(tensor(int64))')

        output_expected = [np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3))]
        res = sess.run([output_name], {
            "tensor": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
            "input_seq": []
        })
        np.testing.assert_array_equal(output_expected, res[0])

    def testOrtExecutionMode(self):
        opt = onnxrt.SessionOptions()
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_SEQUENTIAL)
        opt.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_PARALLEL)

    def testLoadingSessionOptionsFromModel(self):
        try:
            os.environ['ORT_LOAD_CONFIG_FROM_MODEL'] = str(1)
            sess = onnxrt.InferenceSession(get_name("model_with_valid_ort_config_json.onnx"))
            session_options = sess.get_session_options()

            self.assertEqual(session_options.inter_op_num_threads, 5)  # from the ORT config

            self.assertEqual(session_options.intra_op_num_threads, 2)  # from the ORT config

            self.assertEqual(session_options.execution_mode,
                             onnxrt.ExecutionMode.ORT_SEQUENTIAL)  # default option (not from the ORT config)

            self.assertEqual(session_options.graph_optimization_level,
                             onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL)  # from the ORT config

            self.assertEqual(session_options.enable_profiling, True)  # from the ORT config

        except Exception:
            raise

        finally:
            # Make sure the usage of the feature is disabled after this test
            os.environ['ORT_LOAD_CONFIG_FROM_MODEL'] = str(0)

    def testSessionOptionsAddFreeDimensionOverrideByDenotation(self):
        so = onnxrt.SessionOptions()
        so.add_free_dimension_override_by_denotation("DATA_BATCH", 3)
        so.add_free_dimension_override_by_denotation("DATA_CHANNEL", 5)
        sess = onnxrt.InferenceSession(get_name("abs_free_dimensions.onnx"), so)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "x")
        input_shape = sess.get_inputs()[0].shape
        # Free dims with denotations - "DATA_BATCH" and "DATA_CHANNEL" have values assigned to them.
        self.assertEqual(input_shape, [3, 5, 5])

    def testSessionOptionsAddFreeDimensionOverrideByName(self):
        so = onnxrt.SessionOptions()
        so.add_free_dimension_override_by_name("Dim1", 4)
        so.add_free_dimension_override_by_name("Dim2", 6)
        sess = onnxrt.InferenceSession(get_name("abs_free_dimensions.onnx"), so)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "x")
        input_shape = sess.get_inputs()[0].shape
        # "Dim1" and "Dim2" have values assigned to them.
        self.assertEqual(input_shape, [4, 6, 5])


if __name__ == '__main__':
    unittest.main()
