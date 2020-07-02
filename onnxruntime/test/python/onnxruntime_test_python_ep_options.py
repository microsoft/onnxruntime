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

            def getCudaDeviceCount(cuda, q):
                import ctypes

                num_device = ctypes.c_int()
                result = ctypes.c_int()
                error_str = ctypes.c_char_p()

                result = cuda.cuInit(0)
                result = cuda.cuDeviceGetCount(ctypes.byref(num_device))
                if result != CUDA_SUCCESS:
                    cuda.cuGetErrorString(result, ctypes.byref(error_str))
                    print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
                    q.put(-1)
                    return

                q.put(num_device.value)

            # This function is suggested to be run on another process everytime it is called, 
            # the reason is to get a fresh CUDA context and avoid potential issues when switching between different devices. 
            def setDeviceIdTest(cuda, i):
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
                    return
                self.assertEqual(i, device.value)

            def runAdvancedTest(cuda, num_device):
                if num_device < 0:
                    return 

                # Configure session to be ready to run on all available cuda devices
                #
                # Note: We run this testcases on differnt child processes due to the resaon that 
                # each ORT process can only be run on one CUDA device at a time.
                for i in range(num_device):
                    p = Process(target=setDeviceIdTest, args=(cuda, i))
                    p.start()
                    p.join()

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
                    from multiprocessing import Process, Queue

                    cuda = ctypes.CDLL(libname)
                    q = Queue()

                    # First, we spawn worker process to query device number.
                    # After the worker process exits, it destroys its CUDA context, 
                    # so parent process can later create CUDA context without errors
                    p = Process(target=getCudaDeviceCount, args=(cuda, q))
                    p.start()
                    p.join()
                    num_device = q.get()

                    runAdvancedTest(cuda, num_device)

                    runBaseTest1()
                    runBaseTest2()

                except OSError:
                    continue
                else:
                    break
            else:
                runBaseTest1()
                runBaseTest2()
                # raise OSError("could not load any of: " + ' '.join(libnames))

if __name__ == '__main__':
    unittest.main()
