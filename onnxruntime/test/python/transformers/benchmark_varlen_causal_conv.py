# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CUDA-event benchmarks for long prefill and short batched causal convolutions."""

import argparse
import statistics

import numpy as np
import torch
from cuda_plugin_ep_helper import resolve_cuda_plugin_ep
from onnx import TensorProto, helper

import onnxruntime as ort


def benchmark_case(batch, length, args):
    device = torch.device("cuda", args.device)
    dtype, element_type = {
        "float32": (torch.float32, np.float32),
        "float16": (torch.float16, np.float16),
        "bfloat16": (torch.bfloat16, TensorProto.BFLOAT16),
    }[args.dtype]
    tokens, channels, kernel_size = batch * length, args.channels, 4
    shapes = {
        "input": [tokens, channels],
        "weight": [channels, 1, kernel_size],
        "cumulative_sequence_length": [batch + 1],
        "bias": [channels],
        "initial_state": [batch, channels, kernel_size - 1],
        "output": [tokens, channels],
        "final_state": [batch, channels, kernel_size - 1],
    }
    onnx_type = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16, "bfloat16": TensorProto.BFLOAT16}[
        args.dtype
    ]
    inputs = list(shapes)[:5]
    outputs = list(shapes)[5:]
    graph = helper.make_graph(
        [helper.make_node("VarlenCausalConvWithState", inputs, outputs, domain="com.microsoft", activation="none")],
        "varlen_conv_benchmark",
        [
            helper.make_tensor_value_info(
                name, TensorProto.INT32 if name == "cumulative_sequence_length" else onnx_type, shapes[name]
            )
            for name in inputs
        ],
        [helper.make_tensor_value_info(name, onnx_type, shapes[name]) for name in outputs],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=10
    )
    stream = torch.cuda.Stream(device=device)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider = resolve_cuda_plugin_ep("CUDAExecutionProvider", default_test_with_cuda_plugin_ep=True)
    session = ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[(provider, {"device_id": str(args.device), "user_compute_stream": str(stream.cuda_stream)})],
    )
    tensors = {
        name: torch.empty(shape, dtype=dtype, device=device)
        for name, shape in shapes.items()
        if name != "cumulative_sequence_length"
    }
    tensors["input"].normal_(0, 0.1)
    tensors["weight"].normal_(0, 0.1)
    tensors["bias"].zero_()
    tensors["initial_state"].zero_()
    tensors["cumulative_sequence_length"] = torch.arange(batch + 1, dtype=torch.int32, device=device) * length
    binding = session.io_binding()
    for name in inputs:
        binding.bind_input(
            name,
            "cuda",
            args.device,
            np.int32 if name == "cumulative_sequence_length" else element_type,
            shapes[name],
            tensors[name].data_ptr(),
        )
    for name in outputs:
        binding.bind_output(name, "cuda", args.device, element_type, shapes[name], tensors[name].data_ptr())
    run_options = ort.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    stream.wait_stream(torch.cuda.current_stream(device))
    for _ in range(args.warmup):
        session.run_with_iobinding(binding, run_options)
    stream.synchronize()

    history = torch.cat(
        [tensors["initial_state"].float(), tensors["input"].reshape(batch, length, channels).transpose(1, 2).float()],
        dim=2,
    )
    expected = torch.zeros((batch, channels, length), device=device)
    for k in range(kernel_size):
        expected += history[:, :, k : k + length] * tensors["weight"][:, 0, k].float().reshape(1, channels, 1)
    tolerance = 0.01 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(
        tensors["output"], expected.transpose(1, 2).reshape(tokens, channels).to(dtype), rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(tensors["final_state"], history[:, :, -3:].to(dtype), rtol=0, atol=0)
    torch.cuda.synchronize(device)

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph, stream=stream):
        for _ in range(args.repeats):
            session.run_with_iobinding(binding, run_options)
    with torch.cuda.stream(stream):
        for _ in range(10):
            cuda_graph.replay()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.rounds)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.rounds)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            for _ in range(args.iterations):
                cuda_graph.replay()
            end.record()
    stream.synchronize()
    times = [
        start.elapsed_time(end) * 1000 / (args.iterations * args.repeats)
        for start, end in zip(starts, ends, strict=True)
    ]
    return statistics.median(times), min(times), max(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if min(args.channels, args.warmup, args.iterations, args.repeats, args.rounds) <= 0:
        parser.error("channels and all iteration counts must be positive")
    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    print(f"# {torch.cuda.get_device_name(args.device)}; {ort.get_build_info()}")
    print("dtype,batch,sequence_length,total_tokens,median_us,min_us,max_us")
    for batch, length in [(1, 2048), (128, 8), (512, 2), (128, 32), (8, 256)]:
        median, minimum, maximum = benchmark_case(batch, length, args)
        print(f"{args.dtype},{batch},{length},{batch * length},{median:.3f},{minimum:.3f},{maximum:.3f}")


if __name__ == "__main__":
    main()
