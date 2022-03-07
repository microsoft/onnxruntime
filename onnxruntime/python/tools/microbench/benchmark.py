from abc import ABC, abstractmethod
from argparse import ArgumentParser
import time
import numpy
import onnxruntime as ort
import torch


def numpy_type(torch_type):
    type_map = {torch.float32: numpy.float32,
                torch.float16: numpy.float16}
    return type_map[torch_type]


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--provider", required=False, type=str, default="rocm", help="Execution provider to use")
    parser.add_argument("--precision", required=False, type=str, default="fp16", help="Number format to use")
    parser.add_argument('--profiling', type=bool, default=False, help='If enable profiling')


class Benchmark:
    def __init__(self, model, inputs, outputs, args):
        self.provider = args.provider
        self.profiling = args.profiling
        self.model = model
        self.inputs = inputs
        self.outputs = outputs

    def create_input_output_tensors(self):
        device = "cuda"
        input_tensors = {name: torch.from_numpy(array).to(device) for name, array in self.inputs.items()}
        output_tensors = {name: torch.from_numpy(array).to(device) for name, array in self.outputs.items()}
        return input_tensors, output_tensors

    @classmethod
    def create_io_binding(cls, sess, input_tensors, output_tensors):
        io_binding = sess.io_binding()
        for name, tensor in input_tensors.items():
            io_binding.bind_input(name, tensor.device.type, 0, numpy_type(tensor.dtype), tensor.shape, tensor.data_ptr())
        for name, tensor in output_tensors.items():
            io_binding.bind_output(name, tensor.device.type, 0, numpy_type(tensor.dtype), tensor.shape, tensor.data_ptr())
        return io_binding

    def create_session(self):
        sess_opt = ort.SessionOptions()
        sess_opt.enable_profiling = self.profiling
        if self.provider == "rocm":
            execution_provider = ["ROCMExecutionProvider"]
        elif self.provider == "cuda":
            execution_provider = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"The script doesn't support provider type '{self.provider}' yet.")

        sess = ort.InferenceSession(self.model, sess_options=sess_opt, providers=execution_provider)

        if self.provider == "rocm":
            assert 'ROCMExecutionProvider' in sess.get_providers()
        elif self.provider == "cuda":
            assert 'CUDAExecutionProvider' in sess.get_providers()

        return sess

    def benchmark(self):
        sess = self.create_session()
        input_tensors, output_tensors = self.create_input_output_tensors()
        io_binding = self.create_io_binding(sess, input_tensors, output_tensors)
    
        # warm up
        for iter in range(10):
          sess.run_with_iobinding(io_binding)

        # measure
        max_iters = 100
        start_time = time.time()
        for iter in range(max_iters):
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
    def create_inputs_outputs(cls, op_param):
        ...

    def add_case(self, op_param, model):
        self.cases += [(op_param, model)]

    @abstractmethod
    def create_cases(self):
        ...

    @classmethod
    @abstractmethod
    def case_profile(cls, op_param, time):
        ...

    def benchmark(self):
        self.create_cases()
        for op_param, model in self.cases:
            inputs, outputs = self.create_inputs_outputs(op_param)
            bm = Benchmark(model, inputs, outputs, self.args)
            time = bm.benchmark()
            print(self.case_profile(op_param, time))
