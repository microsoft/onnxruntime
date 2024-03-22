# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import numpy
import torch

import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def numpy_type(torch_type):
    type_map = {
        torch.float32: numpy.float32,
        torch.float16: numpy.float16,
        torch.int32: numpy.int32,
        torch.int64: numpy.int64,
    }
    return type_map[torch_type]


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        choices=["cuda", "rocm", "cpu", None],
        default=None,
        help=(
            "Execution provider to use. By default, a "
            "provider is selected in the priority order "
            "(cuda|rocm, cpu) depending on availability."
        ),
    )
    parser.add_argument(
        "--precision",
        required=False,
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Number format to use",
    )
    parser.add_argument(
        "--profiling",
        required=False,
        type=bool,
        default=False,
        help="If enable profiling",
    )


def provider_name(name):
    provider_map = {
        "cuda": "CUDAExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    return provider_map[name]


def get_default_provider():
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "CUDAExecutionProvider"
    if "ROCMExecutionProvider" in ort.get_available_providers():
        return "ROCMExecutionProvider"
    return "CPUExecutionProvider"


class Benchmark:
    def __init__(self, model, inputs, outputs, args):
        self.provider = get_default_provider() if args.provider is None else provider_name(args.provider)
        logger.info(f"Execution provider: {self.provider}")
        self.profiling = args.profiling
        self.model = model
        logger.info(f"Model: {self.model}")
        self.inputs = inputs
        self.outputs = outputs

    def create_input_output_tensors(self):
        on_gpu = self.provider == "CUDAExecutionProvider" or self.provider == "ROCMExecutionProvider"
        device = "cuda" if on_gpu else "cpu"
        input_tensors = {name: torch.from_numpy(array).to(device) for name, array in self.inputs.items()}
        output_tensors = {name: torch.from_numpy(array).to(device) for name, array in self.outputs.items()}
        return input_tensors, output_tensors

    @classmethod
    def create_io_binding(cls, sess, input_tensors, output_tensors):
        io_binding = sess.io_binding()
        for name, tensor in input_tensors.items():
            io_binding.bind_input(
                name,
                tensor.device.type,
                0,
                numpy_type(tensor.dtype),
                tensor.shape,
                tensor.data_ptr(),
            )
        for name, tensor in output_tensors.items():
            io_binding.bind_output(
                name,
                tensor.device.type,
                0,
                numpy_type(tensor.dtype),
                tensor.shape,
                tensor.data_ptr(),
            )
        return io_binding

    def create_session(self):
        sess_opt = ort.SessionOptions()
        sess_opt.enable_profiling = self.profiling
        sess = ort.InferenceSession(self.model, sess_options=sess_opt, providers=[self.provider])
        return sess

    def benchmark(self):
        sess = self.create_session()
        input_tensors, output_tensors = self.create_input_output_tensors()
        io_binding = self.create_io_binding(sess, input_tensors, output_tensors)

        # warm up
        for _iter in range(10):
            sess.run_with_iobinding(io_binding)

        # measure
        max_iters = 100
        start_time = time.time()
        for _iter in range(max_iters):
            sess.run_with_iobinding(io_binding)

        # time is in milliseconds
        elapsed_time = (time.time() - start_time) * 1000 / max_iters
        return elapsed_time


class BenchmarkOp(ABC):
    def __init__(self, args):
        self.args = args
        self.cases = []

    @classmethod
    @abstractmethod
    def create_inputs_outputs(cls, op_param): ...

    def add_case(self, op_param, model):
        self.cases += [(op_param, model)]

    @abstractmethod
    def create_cases(self): ...

    @classmethod
    @abstractmethod
    def case_profile(cls, op_param, time): ...

    def benchmark(self):
        self.create_cases()
        for op_param, model in self.cases:
            inputs, outputs = self.create_inputs_outputs(op_param)
            bm = Benchmark(model, inputs, outputs, self.args)
            time = bm.benchmark()
            print(self.case_profile(op_param, time))
