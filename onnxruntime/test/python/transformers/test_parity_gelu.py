# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Below are test results for Gelu or FastGelu FP32 kernels using CUDA:

Formula             Input(BeforeCast)  MaxDiff     MaxDiff(Optimized)
0(gelu_python)      FP32               2.38E-07    4.77E-07
0(gelu_python)      FP16               0           6.10E-05
1(gelu)             FP32               4.77E-07    0
1(gelu)             FP16               6.10E-05    0
2(erf_gelu)         FP32               2.38E-07    9.54E-07
2(erf_gelu)         FP16               1.22E-04    1.95E-03
3(gelu_new)         FP32               2.38E-07    2.38E-07
3(gelu_new)         FP16               0           0
4(gelu_fast)        FP32               0           2.38E-07
4(gelu_fast)        FP16               0           3.05E-05
5(openai_gelu)      FP32               0           2.38E-07
5(openai_gelu)      FP16               0           3.05E-05

For comparison, CPU has MaxDiff=4.77E-07 for each formula.
"""

import unittest
import torch
from torch import nn
import numpy
import math
import os


class Gelu(nn.Module):
    def __init__(self, formula=4):
        super().__init__()
        self.formula = formula

    def gelu(self, x):
        if self.formula == 0:
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        elif self.formula == 1:
            return nn.functional.gelu(x)
        elif self.formula == 2:
            # erf_gelu in Megatron: x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))
            return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + 1.0)
        elif self.formula == 3:
            # gelu_new in huggingface transformers
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        elif self.formula == 4:
            # gelu_fast in huggingface transformers with lower precision in a constant (0.7978845608)
            return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
        else:
            # openai_gelu in Megatron
            return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))

    @staticmethod
    def get_fused_op(formula):
        return "Gelu" if formula in [0, 1, 2] else "FastGelu"

    def forward(self, x):
        if x.dtype == torch.float16:
            # This test only evaluates FP32 kernels so add data type cast for input and output.
            fp16_gelu = self.gelu(x.to(torch.float32)).to(torch.float16)
            return (fp16_gelu, )
        else:
            fp32_gelu = self.gelu(x)
            return (fp32_gelu, )


def create_inputs(batch_size=1, sequence_length=1, hidden_size=768, float16=False, device=torch.device('cuda')):
    float_type = torch.float16 if float16 else torch.float32
    input = torch.normal(mean=0.0, std=1.0, size=(batch_size, sequence_length, hidden_size)).to(float_type).to(device)
    return input


def get_output_names():
    outputs = ["output"]
    return outputs


def export_onnx(model, onnx_model_path, float16, hidden_size, device):
    from pathlib import Path
    Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

    input_hidden_states = create_inputs(hidden_size=hidden_size, float16=float16, device=device)
    with torch.no_grad():
        outputs = model(input_hidden_states)

    dynamic_axes = {'input': {0: 'batch_size', 1: 'seq_len'}, "output": {0: 'batch_size', 1: 'seq_len'}}

    torch.onnx.export(model,
                      args=(input_hidden_states),
                      f=onnx_model_path,
                      input_names=['input'],
                      output_names=["output"],
                      dynamic_axes=dynamic_axes,
                      example_outputs=outputs,
                      opset_version=11,
                      do_constant_folding=True)
    print("exported:", onnx_model_path)


def optimize_onnx(input_onnx_path, optimized_onnx_path, expected_gelu_op_type='Gelu'):
    from onnxruntime.transformers.optimizer import optimize_model
    onnx_model = optimize_model(input_onnx_path, model_type='gpt2', opt_level=0)
    assert len(onnx_model.get_nodes_by_op_type(
        expected_gelu_op_type)) == 1, f"Expected {expected_gelu_op_type} node not found in the optimized model"
    onnx_model.save_model_to_file(optimized_onnx_path)


def diff_outputs(torch_outputs, ort_outputs, index):
    """ Returns the maximum difference between PyTorch and OnnxRuntime outputs.
    """
    expected_outputs = torch_outputs[index].cpu().numpy()
    diff = numpy.abs(expected_outputs - ort_outputs[index])
    return numpy.amax(diff)


def compare_outputs(torch_outputs, ort_outputs, atol=1e-06, verbose=True):
    """ Returns True if torch and ORT outputs are close for given thresholds, and False otherwise.
    """
    same = numpy.asarray([
        numpy.allclose(ort_outputs[i], torch_outputs[i].cpu().numpy(), atol=atol, rtol=0)
        for i in range(len(ort_outputs))
    ])

    max_abs_diff = [diff_outputs(torch_outputs, ort_outputs, i) for i in range(len(ort_outputs))]

    is_all_close = same.all()
    if is_all_close:
        for i in numpy.where(numpy.logical_not(same))[0]:
            diff = numpy.fabs(ort_outputs[i] - torch_outputs[i].cpu().numpy())
            idx = numpy.unravel_index(diff.argmax(), diff.shape)
            print(
                f'Output {i}, diff={diff[idx]:.9f} index={idx} ort={ort_outputs[i][idx]:.9f} torch={float(torch_outputs[i][idx]):.9f}'
            )

    return is_all_close, max(max_abs_diff)


def create_ort_session(onnx_model_path, use_gpu=True):
    from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel, __version__ as onnxruntime_version
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.intra_op_num_threads = 2
    sess_options.log_severity_level = 2
    execution_providers = ['CPUExecutionProvider'] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return InferenceSession(onnx_model_path, sess_options, providers=execution_providers)


def onnxruntime_inference(ort_session, input):
    ort_inputs = {'input': numpy.ascontiguousarray(input.cpu().numpy())}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs


def run_parity(model,
               onnx_model_path,
               batch_size,
               hidden_size,
               sequence_length,
               float16,
               device,
               optimized,
               test_cases=100,
               verbose=False):
    print(
        f"optimized={optimized}, onnx_model_path={onnx_model_path}, batch_size={batch_size}, hidden_size={hidden_size}, sequence_length={sequence_length}, float16={float16}, device={device}"
    )
    passed_cases = 0
    max_diffs = []
    printed = False  # print only one sample
    ort_session = create_ort_session(onnx_model_path, device.type == 'cuda')
    for i in range(test_cases):
        input_hidden_states = create_inputs(batch_size, sequence_length, hidden_size, float16, device)

        with torch.no_grad():
            torch_outputs = model(input_hidden_states)

        ort_outputs = onnxruntime_inference(ort_session, input_hidden_states)

        tolerance = 1e-04 if float16 else 1e-06
        is_all_close, max_diff = compare_outputs(torch_outputs, ort_outputs, atol=tolerance)
        max_diffs.append(max_diff)
        if is_all_close:
            passed_cases += 1
        elif verbose and not printed:
            printed = True
            numpy.set_printoptions(precision=10, floatmode='fixed')
            torch.set_printoptions(precision=10)
            print("input", input_hidden_states)
            print("torch_outputs", torch_outputs)
            print("ort_outputs", ort_outputs)

    max_diff = max(max_diffs)
    diff_count = len([i for i in max_diffs if i > 0])
    success_flag = "[FAILED]" if passed_cases < test_cases else "[OK]"
    print(f"{success_flag} Passed_cases={passed_cases}/{test_cases}; Max_diff={max_diff}; Diff_count={diff_count}")
    return test_cases - passed_cases


def run(batch_size, float16, optimized, hidden_size, device, test_cases, formula=0, sequence_length=2):
    test_name = f"batch_size={batch_size}, float16={float16}, optimized={optimized}, hidden_size={hidden_size}, formula={formula}"
    print(f"\nTesting ONNX parity: {test_name}")

    model = Gelu(formula=formula)
    model.eval()
    model.to(device)
    if float16:
        model.half()

    # Do not re-use onnx file from previous test since weights of model are random.
    onnx_model_path = './temp/gelu_{}_{}.onnx'.format(formula, "fp16" if float16 else "fp32")
    export_onnx(model, onnx_model_path, float16, hidden_size, device)

    if optimized:
        optimized_onnx_path = './temp/gelu_{}_opt_{}.onnx'.format(formula, "fp16" if float16 else "fp32")
        optimize_onnx(onnx_model_path, optimized_onnx_path, expected_gelu_op_type=Gelu.get_fused_op(formula))
        onnx_path = optimized_onnx_path
    else:
        onnx_path = onnx_model_path

    num_failure = run_parity(model, onnx_path, batch_size, hidden_size, sequence_length, float16, device, optimized,
                             test_cases)

    # clean up onnx file
    os.remove(onnx_model_path)
    if optimized:
        os.remove(onnx_path)

    return num_failure, test_name


class TestGeluParity(unittest.TestCase):
    def setUp(self):
        self.optimized = True  # Change it to False if you want to test parity of non optimized ONNX
        self.test_cases = 100  # Number of test cases per test run
        self.sequence_length = 2
        self.hidden_size = 768
        self.formula_to_test = [0, 1, 3, 4, 5]  # formula 2 cannot pass precision test.

    def run_test(self, batch_size, float16, optimized, hidden_size, device, formula):
        if float16 and device.type == 'cpu':  # CPU does not support FP16
            return
        num_failure, test_name = run(batch_size, float16, optimized, hidden_size, device, self.test_cases, formula,
                                     self.sequence_length)
        self.assertTrue(num_failure == 0, test_name)

    def run_one(self, optimized, device, hidden_size=768, formula=0):
        for batch_size in [4]:
            self.run_test(batch_size,
                          float16=False,
                          optimized=optimized,
                          hidden_size=hidden_size,
                          device=device,
                          formula=formula)

            self.run_test(batch_size,
                          float16=True,
                          optimized=optimized,
                          hidden_size=hidden_size,
                          device=device,
                          formula=formula)

    def test_cpu(self):
        cpu = torch.device('cpu')
        for i in self.formula_to_test:
            self.run_one(self.optimized, cpu, hidden_size=self.hidden_size, formula=i)

    def test_cuda(self):
        if not torch.cuda.is_available():
            import pytest
            pytest.skip('test requires GPU and torch+cuda')
        else:
            gpu = torch.device('cuda')
            for i in self.formula_to_test:
                self.run_one(self.optimized, gpu, hidden_size=self.hidden_size, formula=i)


if __name__ == '__main__':
    unittest.main()
