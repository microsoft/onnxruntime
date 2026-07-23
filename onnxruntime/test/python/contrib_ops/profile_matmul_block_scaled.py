# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Accuracy and latency harness for the CUDA MatMulBlockQuantizedFp4Weight contrib op.

The script builds a single-node com.microsoft contrib-op model, binds CUDA tensors with
I/O binding, compares the output with an FP32 dequantized reference, and prints one JSON
record per case. It is intended for opt-in Blackwell profiling, not for normal CI.

Examples:
  python profile_matmul_block_scaled.py --suite smoke
  python profile_matmul_block_scaled.py --op fp4 --activation-dtype bf16 --m 1 --n 4096 --k 4096 --bias
  python profile_matmul_block_scaled.py --op fp4 --m 16 --n 11008 --k 4096 --repeat 200

For kernel-level evidence, wrap a representative case with nsys:
  nsys profile -t cuda,nvtx -o block_scaled --export=sqlite \
      python profile_matmul_block_scaled.py --op fp4 --m 16 --n 4096 --k 4096
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from onnx import TensorProto, helper

import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail

try:
    import nvtx

    _HAS_NVTX = True
except ImportError:
    nvtx = None
    _HAS_NVTX = False


RESULT_PREFIX = "MATMUL_BLOCK_SCALED_RESULT "

_TORCH_TO_ONNX = {
    torch.float16: TensorProto.FLOAT16,
    torch.bfloat16: TensorProto.BFLOAT16,
}
_FP4_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


@dataclass(frozen=True)
class Case:
    op: str
    m: int
    n: int
    k: int
    activation_dtype: str
    block_size: int | None = None
    bias: bool = False
    seed: int = 0


def _nvtx_range(name: str, color: str = "green"):
    if not _HAS_NVTX:
        return nullcontext()
    return nvtx.annotate(name, color=color)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness.")
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build.")


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _onnx_dtype(name: str) -> int:
    return _TORCH_TO_ONNX[_torch_dtype(name)]


def _raw_uint8(tensor: torch.Tensor) -> bytes:
    return np.ascontiguousarray(tensor.detach().view(torch.uint8).cpu().numpy()).tobytes()


def _make_float_initializer(name: str, tensor: torch.Tensor, onnx_dtype: int):
    if onnx_dtype == TensorProto.FLOAT:
        values = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))
        return helper.make_tensor(name, onnx_dtype, list(tensor.shape), values.tobytes(), raw=True)
    if onnx_dtype == TensorProto.FLOAT16:
        values = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float16))
        return helper.make_tensor(name, onnx_dtype, list(tensor.shape), values.tobytes(), raw=True)
    if onnx_dtype == TensorProto.BFLOAT16:
        values = tensor.detach().to(torch.float32).flatten().cpu().tolist()
        return helper.make_tensor(name, onnx_dtype, list(tensor.shape), values, raw=False)
    raise ValueError(f"Unsupported initializer dtype: {onnx_dtype}")


def _make_session(model: bytes) -> onnxruntime.InferenceSession:
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.log_severity_level = 3
    try:
        return onnxruntime.InferenceSession(model, session_options, providers=["CUDAExecutionProvider"])
    except OrtFail as error:
        if "MatMulBlockScaled" in str(error) and "not a registered" in str(error):
            raise RuntimeError(
                "The active onnxruntime package does not register the MatMulBlockScaled contrib ops. "
                "Build and install this branch's CUDA wheel before running the harness."
            ) from error
        raise


def _model_bytes(nodes, graph_inputs, graph_outputs, initializers, name: str) -> bytes:
    graph = helper.make_graph(nodes, name, graph_inputs, graph_outputs, initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)],
    )
    return model.SerializeToString()


def _quantize_fp4(weight: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[1] % 2 != 0:
        raise ValueError("FP4 packed weight requires even K.")

    n, k = weight.shape
    k_blocks = math.ceil(k / block_size)
    padded_k = k_blocks * block_size
    padded = torch.nn.functional.pad(weight.float(), (0, padded_k - k))
    blocks = padded.reshape(n, k_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    scale = torch.clamp(max_abs / 6.0, min=1.0 / 1024.0).to(torch.float8_e4m3fn)
    scale_f32 = scale.float()

    scaled = blocks / scale_f32.unsqueeze(-1)
    values = _FP4_POS_VALUES.to(device=weight.device)
    flat_abs = scaled.abs().reshape(-1, 1).clamp(max=6.0)
    nearest = torch.abs(flat_abs - values.reshape(1, -1)).argmin(dim=1).reshape_as(scaled).to(torch.uint8)
    codes = nearest | ((scaled < 0).to(torch.uint8) << 3)
    codes = codes.reshape(n, padded_k)[:, :k].contiguous()

    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low | (high << 4)).contiguous()

    quantized_values = values[nearest.long()].reshape_as(scaled) * torch.where(scaled < 0, -1.0, 1.0)
    dequantized = (quantized_values * scale_f32.unsqueeze(-1)).reshape(n, padded_k)[:, :k].contiguous()
    return packed, scale.view(torch.uint8).contiguous(), dequantized


def _make_fp4_model(case: Case, b_packed: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None) -> bytes:
    activation_onnx_type = _onnx_dtype(case.activation_dtype)
    block_size = case.block_size or 16
    inputs = ["A", "B", "weight_scale", "weight_scale_2"]
    initializers = [
        helper.make_tensor("B", TensorProto.UINT8, [case.n, case.k // 2], _raw_uint8(b_packed), raw=True),
        helper.make_tensor(
            "weight_scale",
            TensorProto.UINT8,
            [case.n, math.ceil(case.k / block_size)],
            _raw_uint8(weight_scale),
            raw=True,
        ),
        helper.make_tensor("weight_scale_2", TensorProto.FLOAT, [1], [1.0], raw=False),
    ]
    if bias is not None:
        inputs.extend(["", "bias"])
        initializers.append(_make_float_initializer("bias", bias, activation_onnx_type))

    node = helper.make_node(
        "MatMulBlockQuantizedFp4Weight",
        inputs,
        ["Y"],
        domain="com.microsoft",
        K=case.k,
        N=case.n,
        block_size=block_size,
    )
    graph_inputs = [helper.make_tensor_value_info("A", activation_onnx_type, [case.m, case.k])]
    graph_outputs = [helper.make_tensor_value_info("Y", activation_onnx_type, [case.m, case.n])]
    return _model_bytes([node], graph_inputs, graph_outputs, initializers, "MatMulBlockQuantizedFp4Weight_Profile")


def _fp4_reference(a: torch.Tensor, b_dequantized: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    result = a.float() @ b_dequantized.float().T
    if bias is not None:
        result += bias.float().reshape(1, -1)
    return result.to(a.dtype).float()


def _fp4_native_sm120_enabled() -> bool:
    return os.environ.get("ORT_MATMUL_BLOCK_SCALED_FP4_NATIVE_SM120", "").lower() in {"1", "true", "yes", "on"}


def _fp4_native_sm120_supported(case: Case) -> bool:
    block_size = case.block_size or 16
    return (
        _fp4_native_sm120_enabled()
        and case.m > 8
        and block_size == 16
        and case.k % 32 == 0
        and case.n % 32 == 0
        and case.activation_dtype in {"fp16", "bf16"}
    )


def _fp4_expected_path(case: Case) -> str:
    block_size = case.block_size or 16
    if case.m > 0 and case.m <= 8 and block_size == 16 and case.k % 32 == 0:
        return "fp4_gemv"
    if _fp4_native_sm120_supported(case):
        return "sm120_native_fp4_gemm"
    return "fp4_dequant_cublas"


def _make_inputs(case: Case) -> tuple[bytes, torch.Tensor, torch.Tensor, str]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(case.seed)
    block_size = case.block_size or 16

    activation_dtype = _torch_dtype(case.activation_dtype)
    a = (torch.randn((case.m, case.k), generator=generator, device="cuda") * 0.75).to(activation_dtype).contiguous()
    weight = torch.randn((case.n, case.k), generator=generator, device="cuda", dtype=torch.float32) * 0.75
    b_packed, weight_scale, b_dequantized = _quantize_fp4(weight, block_size)
    bias = None
    if case.bias:
        bias = (torch.randn((case.n,), generator=generator, device="cuda") * 0.25).to(activation_dtype).contiguous()
    model = _make_fp4_model(case, b_packed, weight_scale, bias)
    if _fp4_native_sm120_supported(case):
        _, _, a_dequantized = _quantize_fp4(a.float(), block_size)
        reference = _fp4_reference(a_dequantized.to(activation_dtype), b_dequantized, bias)
    else:
        reference = _fp4_reference(a, b_dequantized, bias)
    return model, a, reference, _fp4_expected_path(case)


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = actual.float() - expected.float()
    abs_diff = diff.abs()
    rel_diff = abs_diff / torch.clamp(expected.float().abs(), min=1.0e-6)
    return {
        "max_abs_error": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "max_rel_error": float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0,
        "max_expected_abs": float(expected.float().abs().max().item()) if expected.numel() else 0.0,
    }


def _run_timed(
    session: onnxruntime.InferenceSession, a: torch.Tensor, y: torch.Tensor, warmup: int, repeat: int
) -> list[float]:
    io_binding = session.io_binding()
    io_binding.bind_input("A", "cuda", 0, _TORCH_TO_ONNX[a.dtype], list(a.shape), a.data_ptr())
    io_binding.bind_output("Y", "cuda", 0, _TORCH_TO_ONNX[y.dtype], list(y.shape), y.data_ptr())

    with _nvtx_range("warmup", "yellow"):
        for _ in range(warmup):
            session.run_with_iobinding(io_binding)
    torch.cuda.synchronize()

    times_ms = []
    with _nvtx_range("benchmark", "green"):
        for _ in range(repeat):
            start = time.perf_counter()
            session.run_with_iobinding(io_binding)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def _summarize_times(times_ms: list[float]) -> dict[str, float]:
    sorted_times = sorted(times_ms)
    p90_index = min(len(sorted_times) - 1, math.ceil(0.90 * len(sorted_times)) - 1)
    p99_index = min(len(sorted_times) - 1, math.ceil(0.99 * len(sorted_times)) - 1)
    return {
        "mean_ms": statistics.fmean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "p90_ms": sorted_times[p90_index],
        "p99_ms": sorted_times[p99_index],
        "min_ms": sorted_times[0],
    }


def run_case(case: Case, warmup: int, repeat: int, atol: float, rtol: float) -> dict[str, Any]:
    model, a, reference, expected_path = _make_inputs(case)
    output_dtype = _torch_dtype(case.activation_dtype)
    y = torch.empty((case.m, case.n), dtype=output_dtype, device="cuda")
    session = _make_session(model)
    times_ms = _run_timed(session, a, y, warmup, repeat)

    metrics = _error_metrics(y, reference)
    threshold = atol + rtol * metrics["max_expected_abs"]
    passed = metrics["max_abs_error"] <= threshold
    flops = 2.0 * case.m * case.n * case.k
    timing = _summarize_times(times_ms)
    result = {
        "op": case.op,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "block_size": case.block_size or 16,
        "activation_dtype": case.activation_dtype,
        "bias": case.bias,
        "expected_path": expected_path,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "tflops": flops / (timing["mean_ms"] * 1.0e-3) / 1.0e12,
        **timing,
        **metrics,
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return result


def _default_cases(args) -> list[Case]:
    if args.m is not None and args.n is not None and args.k is not None:
        return [
            Case(
                op=args.op,
                m=args.m,
                n=args.n,
                k=args.k,
                activation_dtype=args.activation_dtype,
                block_size=args.block_size,
                bias=args.bias,
                seed=args.seed,
            )
        ]

    cases = []
    if args.suite == "smoke":
        cases.extend(
            [
                Case("fp4", 1, 80, 256, "fp16", bias=True, seed=args.seed + 3),
                Case("fp4", 32, 128, 256, "bf16", bias=False, seed=args.seed + 4),
            ]
        )
        return cases

    matrix_ms = [1, 2, 4, 8] if args.suite == "decode" else [16, 32, 64, 128]
    matrix_shapes = [(4096, 4096), (4096, 11008)]
    for k, n in matrix_shapes:
        cases.extend(
            Case("fp4", m, n, k, args.activation_dtype, bias=args.bias, seed=args.seed + 100 + m) for m in matrix_ms
        )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the CUDA block-scaled FP4 MatMul contrib op")
    parser.add_argument("--op", choices=["fp4"], default="fp4")
    parser.add_argument("--suite", choices=["smoke", "decode", "prefill"], default="smoke")
    parser.add_argument("--m", type=int, help="M rows for single-case mode")
    parser.add_argument("--n", type=int, help="N columns for single-case mode")
    parser.add_argument("--k", type=int, help="K reduction dimension for single-case mode")
    parser.add_argument("--block-size", type=int, help="Override block_size attribute")
    parser.add_argument("--activation-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--bias", action="store_true", help="Enable FP4 bias")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2.0)
    parser.add_argument("--rtol", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_cuda()
    single_case_args = [args.m is not None, args.n is not None, args.k is not None]
    if any(single_case_args) and not all(single_case_args):
        raise ValueError("Single-case mode requires all of --m, --n and --k.")

    results = [run_case(case, args.warmup, args.repeat, args.atol, args.rtol) for case in _default_cases(args)]
    failures = [result for result in results if not result["passed"]]
    if failures:
        raise SystemExit(f"{len(failures)} case(s) failed accuracy checks")


if __name__ == "__main__":
    main()
