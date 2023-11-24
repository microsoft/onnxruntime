# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import argparse
import os
import sys
from pathlib import Path

import numpy
import torch


def parse_arguments(namespace_filter=None):
    parser = argparse.ArgumentParser()

    # useful EPs that don't require the use of optmizer.py
    parser.add_argument(
        "-n",
        "--no_optimize",
        required=False,
        action="store_false",
        default=True,
        dest="optimize",
        help="Turn off onnxruntime optimizers (Default off optimizers ON)",
    )

    # useful for debugging and viewing state during test runs
    parser.add_argument(
        "-l",
        "--log_verbose",
        required=False,
        action="store_true",
        default=False,
        help="Set Onnxruntime log_serverity_level=0 (VERBOSE) ",
    )

    args, remaining_args = parser.parse_known_args(namespace=namespace_filter)

    return args, sys.argv[:1] + remaining_args


def find_transformers_source(sub_dir_paths=[]):  # noqa: B006
    source_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "tools",
        "transformers",
        *sub_dir_paths,
    )
    source_dir = os.path.normpath(source_dir)
    if os.path.exists(source_dir):
        if source_dir not in sys.path:
            sys.path.append(source_dir)
        return True
    return False


def create_inputs(
    batch_size=1,
    sequence_length=1,
    hidden_size=768,
    float16=False,
    device=torch.device("cuda"),  # noqa: B008
):
    float_type = torch.float16 if float16 else torch.float32
    return torch.normal(mean=0.0, std=10.0, size=(batch_size, sequence_length, hidden_size)).to(float_type).to(device)


def export_onnx(model, onnx_model_path, float16, hidden_size, device):
    Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

    input_hidden_states = create_inputs(hidden_size=hidden_size, float16=float16, device=device)
    with torch.no_grad():
        model(input_hidden_states)

    dynamic_axes = {
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"},
    }

    torch.onnx.export(
        model,
        args=(input_hidden_states),
        f=onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
    )
    print("exported:", onnx_model_path)


def optimize_onnx(
    input_onnx_path,
    optimized_onnx_path,
    expected_op=None,
    use_gpu=False,
    opt_level=None,
    verbose=False,
):
    if find_transformers_source():
        from optimizer import optimize_model
    else:
        from onnxruntime.transformers.optimizer import optimize_model

    onnx_model = optimize_model(
        input_onnx_path, model_type="gpt2", use_gpu=use_gpu, opt_level=opt_level, verbose=verbose
    )
    onnx_model.save_model_to_file(optimized_onnx_path)

    if expected_op is not None:
        assert (
            len(onnx_model.get_nodes_by_op_type(expected_op)) == 1
        ), f"Expected {expected_op} node not found in the optimized model {optimized_onnx_path}"


def diff_outputs(torch_outputs, ort_outputs, index):
    """Returns the maximum difference between PyTorch and OnnxRuntime outputs."""
    expected_outputs = torch_outputs[index].cpu().numpy()
    diff = numpy.abs(expected_outputs - ort_outputs[index])
    return numpy.amax(diff)


def compare_outputs(torch_outputs, ort_outputs, atol=1e-06, verbose=True):
    """Compare outputs from PyTorch and OnnxRuntime

    Args:
        torch_outputs (Tuple[Torch.Tensor]): PyTorch model output
        ort_outputs (List[numpy.ndarray]): OnnxRuntime output
        atol (float, optional): Absolute tollerance. Defaults to 1e-06.
        verbose (bool, optional): Print more information. Defaults to True.

    Returns:
        is_all_close(bool): whether all elements are close.
        max_abs_diff(float): maximum absolute difference.
    """
    same = numpy.asarray(
        [
            numpy.allclose(ort_outputs[i], torch_outputs[i].cpu().numpy(), atol=atol, rtol=0)
            for i in range(len(ort_outputs))
        ]
    )

    max_abs_diff = [diff_outputs(torch_outputs, ort_outputs, i) for i in range(len(ort_outputs))]

    is_all_close = same.all()
    if (not is_all_close) and verbose:
        for i in numpy.where(numpy.logical_not(same))[0]:
            diff = numpy.fabs(ort_outputs[i] - torch_outputs[i].cpu().numpy())
            idx = numpy.unravel_index(diff.argmax(), diff.shape)
            print(
                f"Output {i}, diff={diff[idx]:.9f} index={idx} ort={ort_outputs[i][idx]:.9f} torch={float(torch_outputs[i][idx]):.9f}"
            )

    return is_all_close, max(max_abs_diff)


def create_ort_session(onnx_model_path, use_gpu=True, optimized=True, verbose=False):
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.intra_op_num_threads = 2
    sess_options.log_severity_level = 2

    if verbose:
        sess_options.log_severity_level = 0

    execution_providers = []

    if use_gpu:
        if torch.version.cuda:
            execution_providers.append("CUDAExecutionProvider")
        elif torch.version.hip:
            if not optimized:
                execution_providers.append("MIGraphXExecutionProvider")

            execution_providers.append("ROCMExecutionProvider")

    execution_providers.append("CPUExecutionProvider")

    return InferenceSession(onnx_model_path, sess_options, providers=execution_providers)


def onnxruntime_inference(ort_session, input):
    ort_inputs = {"input": numpy.ascontiguousarray(input.cpu().numpy())}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs


def run_parity(
    model,
    onnx_model_path,
    batch_size,
    hidden_size,
    sequence_length,
    float16,
    device,
    optimized,
    test_cases=100,
    verbose=False,
    tolerance=None,
):
    passed_cases = 0
    max_diffs = []
    printed = False  # print only one sample
    ort_session = create_ort_session(onnx_model_path, device.type == "cuda", optimized=optimized, verbose=verbose)
    for _i in range(test_cases):
        input_hidden_states = create_inputs(batch_size, sequence_length, hidden_size, float16, device)

        with torch.no_grad():
            torch_outputs = model(input_hidden_states)

        ort_outputs = onnxruntime_inference(ort_session, input_hidden_states)

        if tolerance is None:
            tolerance = 2e-03 if float16 else 1e-05
        is_all_close, max_diff = compare_outputs(torch_outputs, ort_outputs, atol=tolerance, verbose=verbose)
        max_diffs.append(max_diff)
        if is_all_close:
            passed_cases += 1
        elif verbose and not printed:
            printed = True
            numpy.set_printoptions(precision=10, floatmode="fixed")
            torch.set_printoptions(precision=10)
            print("input", input_hidden_states)
            print("torch_outputs", torch_outputs)
            print("ort_outputs", ort_outputs)

    max_diff = max(max_diffs)
    diff_count = len([i for i in max_diffs if i > 0])
    success_flag = "[FAILED]" if passed_cases < test_cases else "[OK]"
    print(f"{success_flag} Passed_cases={passed_cases}/{test_cases}; Max_diff={max_diff}; Diff_count={diff_count}")
    return test_cases - passed_cases
