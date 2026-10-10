# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CUDA-event and host-latency benchmark with byte-identical MatMulNBits weights.

Use a CUDA-enabled source build on PYTHONPATH. For kernel breakdown and dispatch,
run under nsys with CUDA/NVTX tracing and inspect the named benchmark ranges.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


def make_session(
    m,
    k,
    n,
    bits,
    block,
    packed,
    scales,
    dtype,
    mode,
    activation_block,
    runtime_scales,
    weight_only=False,
    runtime_weights=False,
):
    onnx_type = TensorProto.FLOAT16 if dtype == "fp16" else TensorProto.BFLOAT16
    initializers = []
    inputs = [helper.make_tensor_value_info("A", onnx_type, [m, k])]
    if runtime_weights:
        inputs.append(helper.make_tensor_value_info("B", TensorProto.UINT8, list(packed.shape)))
    else:
        initializers.append(numpy_helper.from_array(packed, "B"))
    if runtime_scales:
        inputs.append(helper.make_tensor_value_info("scales", onnx_type, list(scales.shape)))
    else:
        initializers.append(numpy_helper.from_array(scales, "scales"))
    attrs = {"bits": bits, "block_size": block}
    if weight_only:
        attrs.update(K=k, N=n, accuracy_level=0)
    else:
        attrs["activation_quantization"] = mode
        if mode == "dynamic_per_block":
            attrs["activation_block_size"] = activation_block
    node = helper.make_node(
        "MatMulNBits" if weight_only else "DynamicQuantizeMatMulNBits",
        ["A", "B", "scales"],
        ["Y"],
        domain="com.microsoft",
        **attrs,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "benchmark",
            inputs,
            [helper.make_tensor_value_info("Y", onnx_type, [m, n])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("com.microsoft", 1)],
        ir_version=8,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[("CUDAExecutionProvider", {"user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})],
    )


def measure(name, run, warmup, repeat, trials):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    gpu_times, host_times = [], []
    with torch.cuda.nvtx.range(name):
        for _ in range(trials):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            host_start = time.perf_counter()
            for _ in range(repeat):
                run()
            end.record()
            end.synchronize()
            host_times.append((time.perf_counter() - host_start) * 1e6 / repeat)
            gpu_times.append(start.elapsed_time(end) * 1000 / repeat)
    return {
        "cuda_event_median_us": float(np.median(gpu_times)),
        "host_median_us": float(np.median(host_times)),
        "cuda_event_min_us": min(gpu_times),
        "trials": trials,
        "repeat": repeat,
    }


def run_case(args, m, k, n):
    block = args.block_size or max(16, 1 << (k - 1).bit_length())
    if args.bits == 2 and block > 256:
        raise ValueError("INT2 requires --block-size <= 256")
    groups = (k + block - 1) // block
    rng = np.random.default_rng(args.seed)
    packed = rng.integers(0, 256, size=(n, groups, block * args.bits // 8), dtype=np.uint8)
    np_dtype = np.float16 if args.dtype == "fp16" else ml_dtypes.bfloat16
    scales = rng.uniform(0.001, 0.01, size=(n, groups)).astype(np_dtype)
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    a = torch.from_numpy(rng.normal(size=(m, k)).astype(np.float32)).cuda().to(torch_dtype)
    onnx_type = TensorProto.FLOAT16 if args.dtype == "fp16" else TensorProto.BFLOAT16
    activation_groups = (
        (k + args.activation_block_size - 1) // args.activation_block_size if args.mode == "dynamic_per_block" else 1
    )
    row_bytes = k + (0 if args.mode == "dynamic_per_tensor" else activation_groups * 4)
    tile_m = min(m, max(1, (16 * 1024 * 1024) // row_bytes))
    result = {
        "m": m,
        "k": k,
        "n": n,
        "bits": args.bits,
        "block_size": block,
        "dtype": args.dtype,
        "mode": args.mode,
        "activation_block_size": args.activation_block_size,
        "runtime_scales": args.runtime_scales,
        "runtime_weights": args.runtime_weights,
        "packed_weight_bytes": packed.nbytes,
        "weight_padding_bytes": packed.nbytes - (n * k * args.bits + 7) // 8,
        "weight_scale_bytes": scales.nbytes,
        "activation_scratch_target_bytes": 16 * 1024 * 1024,
        "quantized_activation_upper_bound_bytes": tile_m * k,
        "activation_scale_upper_bound_bytes": 4
        if args.mode == "dynamic_per_tensor"
        else tile_m * activation_groups * 4,
        "tensor_reduction_scratch_upper_bound_bytes": 4096 if args.mode == "dynamic_per_tensor" and m * k > 4096 else 0,
        "expanded_weight_scratch_bytes": 0,
        "prepack_bytes": 0,
    }
    scale_tensor = torch.from_numpy(scales.astype(np.float32)).cuda().to(torch_dtype)
    weight_tensor = torch.from_numpy(packed).cuda() if args.runtime_weights else None
    outputs = {}
    for weight_only, label in ((False, "dynamic"), (True, "matmul_nbits")):
        if weight_only and args.skip_matmul_nbits:
            result["matmul_nbits_unavailable"] = "Explicitly disabled with --skip-matmul-nbits"
            continue
        session = make_session(
            m,
            k,
            n,
            args.bits,
            block,
            packed,
            scales,
            args.dtype,
            args.mode,
            args.activation_block_size,
            args.runtime_scales,
            weight_only,
            args.runtime_weights,
        )
        y = torch.empty((m, n), device="cuda", dtype=torch_dtype)
        binding = session.io_binding()
        binding.bind_input("A", "cuda", 0, onnx_type, list(a.shape), a.data_ptr())
        if args.runtime_weights:
            binding.bind_input("B", "cuda", 0, TensorProto.UINT8, list(packed.shape), weight_tensor.data_ptr())
        if args.runtime_scales:
            binding.bind_input("scales", "cuda", 0, onnx_type, list(scales.shape), scale_tensor.data_ptr())
        binding.bind_output("Y", "cuda", 0, onnx_type, list(y.shape), y.data_ptr())
        result[label] = measure(
            f"benchmark/{label}/m{m}_k{k}_n{n}",
            partial(session.run_with_iobinding, binding),
            args.warmup,
            args.repeat,
            args.trials,
        )
        outputs[label] = y.clone()
    pack = 8 // args.bits
    codes = np.stack([(packed >> (args.bits * lane)) & ((1 << args.bits) - 1) for lane in range(pack)], axis=-1)
    codes = (codes.reshape(n, groups * block)[:, :k].astype(np.int16) - (1 << (args.bits - 1))).astype(np.int8)
    weights = torch.from_numpy(codes.copy()).cuda()
    expanded_scales = scale_tensor.float().repeat_interleave(block, dim=1)[:, :k]
    fp_weights = (weights.float() * expanded_scales).to(torch_dtype).T.contiguous()
    y_fp = torch.empty((m, n), dtype=torch_dtype, device="cuda")
    result["fp_gemm"] = measure(
        f"benchmark/fp_gemm/m{m}_k{k}_n{n}",
        lambda: torch.mm(a, fp_weights, out=y_fp),
        args.warmup,
        args.repeat,
        args.trials,
    )
    result["fp_baseline_weight_bytes"] = fp_weights.numel() * fp_weights.element_size()
    if groups == 1 and args.mode != "dynamic_per_block" and m > 16 and m % 8 == 0 and n % 8 == 0 and k % 32 == 0:
        int_weights = weights.T.contiguous()
        accumulators = torch.empty((m, n), dtype=torch.int32, device="cuda")

        def int8_pipeline():
            values = a.float()
            maximum = (
                values.abs().amax() if args.mode == "dynamic_per_tensor" else values.abs().amax(dim=1, keepdim=True)
            )
            activation_scale = torch.where(maximum == 0, 1.0, maximum / 127.0)
            quantized = (values / activation_scale).round().clamp(-127, 127).to(torch.int8)
            torch._int_mm(quantized, int_weights, out=accumulators)
            return ((accumulators.float() * activation_scale) * scale_tensor.float().T).to(torch_dtype)

        result["torch_int8_quantize_gemm_scale"] = measure(
            f"benchmark/torch_int8_pipeline/m{m}_k{k}_n{n}",
            int8_pipeline,
            args.warmup,
            args.repeat,
            args.trials,
        )
        reference_values = a.float().cpu().numpy()
        reference_maximum = (
            np.max(np.abs(reference_values))
            if args.mode == "dynamic_per_tensor"
            else np.max(np.abs(reference_values), axis=1, keepdims=True)
        )
        reference_scale = np.where(reference_maximum == 0, np.float32(1), reference_maximum / np.float32(127)).astype(
            np.float32
        )
        reference_codes = np.clip(np.rint(reference_values / reference_scale), -127, 127).astype(np.int8)
        reference_acc = torch._int_mm(torch.from_numpy(reference_codes).cuda(), int_weights)
        reference_output = (
            (reference_acc.float() * torch.as_tensor(reference_scale, device="cuda")) * scale_tensor.float().T + 0.0
        ).to(torch_dtype)
        torch.testing.assert_close(outputs["dynamic"], reference_output, rtol=0, atol=0)
        baseline_difference = int8_pipeline().float() - outputs["dynamic"].float()
        result["torch_int8_baseline_relative_rmse_vs_dynamic"] = (
            (baseline_difference.square().mean() / outputs["dynamic"].float().square().mean()).sqrt().item()
        )
        result["int8_baseline_expanded_weight_bytes"] = int_weights.numel()
        result["int8_baseline_int32_output_bytes"] = accumulators.numel() * 4
    else:
        result["torch_int8_baseline_unavailable"] = (
            "Requires one interval, M > 16, M/N divisible by 8, and K divisible by 32"
        )
    for label, output in outputs.items():
        difference = output.float() - y_fp.float()
        result[f"{label}_relative_rmse_vs_fp"] = (
            (difference.square().mean() / y_fp.float().square().mean()).sqrt().item()
        )
    if "matmul_nbits" in result:
        result["speedup_vs_matmul_nbits"] = (
            result["matmul_nbits"]["cuda_event_median_us"] / result["dynamic"]["cuda_event_median_us"]
        )
    result["speedup_vs_fp_gemm"] = result["fp_gemm"]["cuda_event_median_us"] / result["dynamic"]["cuda_event_median_us"]
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16, 128, 1536])
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--bits", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--block-size", type=int, default=0, help="0 selects the next power of two >= K")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--mode", choices=["dynamic_per_token", "dynamic_per_tensor", "dynamic_per_block"], default="dynamic_per_token"
    )
    parser.add_argument("--activation-block-size", type=int, default=128)
    parser.add_argument("--runtime-scales", action="store_true")
    parser.add_argument("--runtime-weights", action="store_true", help="Disable constant-weight prepacking")
    parser.add_argument(
        "--skip-matmul-nbits",
        action="store_true",
        help="Explicitly omit the weight-only baseline if this build does not support its geometry",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.k is None) != (args.n is None):
        parser.error("--k and --n must be specified together")
    if any(m <= 0 for m in args.m) or min(args.warmup, args.repeat, args.trials, args.activation_block_size) <= 0:
        parser.error("Rows, warmup, repeat, trials, and activation block size must be positive")
    if args.k is not None and min(args.k, args.n) <= 0:
        parser.error("K and N must be positive")
    if args.block_size and (args.block_size < 16 or args.block_size & (args.block_size - 1)):
        parser.error("--block-size must be a power of two >= 16")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("A CUDA-enabled ONNX Runtime build is required")
    cases = [(args.k, args.n)] if args.k is not None else [(2048, 6144), (6144, 2048), (2048, 4096), (2048, 2048)]
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "onnxruntime": ort.__version__,
        "warmup": args.warmup,
        "timing": "Same user compute stream; CUDA events include inter-launch host gaps",
        "baselines": "FP GEMM uses decoded/scaled weights; MatMulNBits reuses byte-identical B/scales; no accuracy equivalence claimed",
        "matmul_nbits_baseline": "explicitly disabled" if args.skip_matmul_nbits else "enabled",
    }
    print(json.dumps(metadata), flush=True)
    results = [run_case(args, m, k, n) for k, n in cases for m in args.m]
    if args.output:
        args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
