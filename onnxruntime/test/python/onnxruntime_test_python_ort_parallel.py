# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ctypes
import sys
import threading
import time
import unittest

import numpy as np
from helper import get_name

import onnxruntime as onnxrt


class ThreadObj:
    def __init__(self, model_path: str, iterations: int, idx: int, num_device: int, provider_options_list: list):
        self.iterations = iterations
        sess_opt = onnxrt.SessionOptions()
        sess_opt.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_opt.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL  # ORT will use thread from inter-op thread pool
        self.inference_session = onnxrt.InferenceSession(model_path, sess_opt, provider_options_list[idx % num_device])
        self.input = {
            "Input3": np.ones([1, 1, 28, 28], np.float32),
        }
        self.idx = idx

    def warmup(self):
        print(f"[THREAD {self.idx}] running warmup")
        self.inference_session.run(None, self.input)
        print(f"[THREAD {self.idx}] warmup done")

    def run(self, thread_times, threads_complete):
        for iter in range(self.iterations):
            print(f"[THREAD {self.idx}] running iteration {iter}")
            thread_times[self.idx] = time.time()
            self.inference_session.run(None, self.input)
            thread_times[self.idx] = time.time()
            print(f"[THREAD {self.idx}] completed iteration {iter}")
        threads_complete[0] += 1


def thread_target(obj, thread_times, threads_complete):
    obj.run(thread_times, threads_complete)


# This unittest class creates 10 threads, each thread creates its own inference session and runs one warmup sequentially.
# Once all threads finish their warmup run, all threads run multiple inference runs concurrently.
class TestParallelRun(unittest.TestCase):
    def test_select_ep_to_run_ort_parallel_execution_mode(self):
        if "TensorrtExecutionProvider" in onnxrt.get_available_providers():
            cuda_lib = self.load_cuda_lib()
            device_cnt = self.cuda_device_count(cuda_lib)
            assert device_cnt > 0
            print(f"Number of GPUs available: {device_cnt}")
            self.run_inference_with_parallel_execution_mode("TensorrtExecutionProvider", device_cnt)
        elif "CUDAExecutionProvider" in onnxrt.get_available_providers():
            cuda_lib = self.load_cuda_lib()
            device_cnt = self.cuda_device_count(cuda_lib)
            assert device_cnt > 0
            print(f"Number of GPUs available: {device_cnt}")
            self.run_inference_with_parallel_execution_mode("CUDAExecutionProvider", device_cnt)

    def load_cuda_lib(self):
        cuda_lib = None
        if sys.platform == "win32":
            cuda_lib = "nvcuda.dll"
        elif sys.platform == "linux":
            cuda_lib = "libcuda.so"
        elif sys.platform == "darwin":
            cuda_lib = "libcuda.dylib"

        if cuda_lib is not None:
            try:
                return ctypes.CDLL(cuda_lib)
            except OSError:
                pass
        return None

    def cuda_device_count(self, cuda_lib):
        if cuda_lib is None:
            return -1
        num_device = ctypes.c_int()
        cuda_lib.cuInit(0)
        result = cuda_lib.cuDeviceGetCount(ctypes.byref(num_device))
        if result != 0:
            error_str = ctypes.c_char_p()
            cuda_lib.cuGetErrorString(result, ctypes.byref(error_str))
            print(f"cuDeviceGetCount failed with error code {result}: {error_str.value.decode()}")
            return -1
        return num_device.value

    def run_inference_with_parallel_execution_mode(self, ep, num_device):
        provider_options = []
        for i in range(num_device):
            option = [
                (
                    ep,
                    {
                        "device_id": i,
                    },
                ),
            ]
            provider_options.append(option)

        model_path = get_name("mnist.onnx")
        iterations = 20
        hang_time = 60

        num_threads = 10
        t_obj_list = []
        thread_list = []

        threads_complete = [0]
        thread_times = [0] * num_threads

        for tidx in range(num_threads):
            obj = ThreadObj(model_path, iterations, tidx, num_device, provider_options)
            t_obj_list.append(obj)
            obj.warmup()

        for t_obj in t_obj_list:
            thread = threading.Thread(
                target=thread_target,
                daemon=True,
                args=(
                    t_obj,
                    thread_times,
                    threads_complete,
                ),
            )
            thread.start()
            thread_list.append(thread)

        time.sleep(5)
        while True:
            for t_time in thread_times:
                if time.time() - t_time < hang_time:
                    continue
                else:
                    print("Hang occured, ending test")
                    exit(1)
            if threads_complete[0] == num_threads:
                break
            time.sleep(5)

        for thread in thread_list:
            thread.join()

        print("All threads completed")


if __name__ == "__main__":
    unittest.main()
