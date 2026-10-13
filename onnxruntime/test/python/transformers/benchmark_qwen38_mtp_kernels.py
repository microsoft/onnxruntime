import argparse
import functools
import json
import os
import re
import sys
from pathlib import Path

import cupy
import numpy

ROOT = Path(__file__).resolve().parents[3]


@functools.cache
def compile_device_code(code, name):
    cuda_path = os.environ.get("CUDA_PATH") or cupy.cuda.get_cuda_path()
    options = ("--std=c++17", f"-I{cuda_path}/include", f"-I{cuda_path}/include/cccl")
    for arch in filter(None, os.environ.get("ORT_KERNEL_PROBE_ARCHS", "").split(",")):
        cupy.cuda.compiler.compile_using_nvrtc(code, options=options, arch=arch, name_expressions=(name,))
    module = cupy.RawModule(code=code, options=options, name_expressions=(name,))
    return module.get_function(name)


def device_function(source, name):
    match = re.search(r"__device__[^\n]*\b" + name + r"\(", source)
    if match is None:
        raise ValueError(f"Missing device function: {name}")
    body = source.index("{", match.end())
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end]


def graph_latency(kernel, grid, block, args, iterations=100):
    cupy.cuda.Stream.null.synchronize()
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        for _ in range(10):
            kernel(grid, block, args)
    stream.synchronize()
    stream.begin_capture()
    with stream:
        for _ in range(iterations):
            kernel(grid, block, args)
    graph = stream.end_capture()
    start = cupy.cuda.Event()
    stop = cupy.cuda.Event()
    samples = []
    for _ in range(5):
        start.record(stream)
        graph.launch(stream)
        stop.record(stream)
        stop.synchronize()
        samples.append(cupy.cuda.get_elapsed_time(start, stop) / iterations)
    return float(numpy.median(samples))


def key_data(values, element_type):
    if element_type == "float":
        return cupy.asarray(values)
    if element_type == "half":
        return cupy.asarray(values.astype(numpy.float16))
    bits = values.view(numpy.uint32)
    rounded = bits + numpy.uint32(0x7FFF) + ((bits >> 16) & 1)
    return cupy.asarray((rounded >> 16).astype(numpy.uint16))


def main():
    parser = argparse.ArgumentParser(description="Standalone NVRTC parity/timing probe; does not rebuild ORT.")
    parser.add_argument("--baseline-stdin", action="store_true", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--sanitizer-case", action="store_true")
    parser.add_argument("--kernel", choices=("nvfp4", "nvfp4_reuse"), default="nvfp4")
    args = parser.parse_args()
    baseline = sys.stdin.read()
    if args.kernel == "nvfp4_reuse":
        run_nvfp4_reuse(baseline, args.check_only, args.sanitizer_case)
        run_nvfp4_reuse(baseline, args.check_only, args.sanitizer_case, map_sources=False)
    else:
        run_nvfp4(baseline, args.check_only, args.sanitizer_case)


def compile_nvfp4(source, element_type, fused, bias, wide, reuse=False):
    common = (ROOT / "contrib_ops/cuda/llm/moe_gemm/common.h").read_text()
    details = (ROOT / "contrib_ops/cuda/llm/fpA_intB_gemv/details.h").read_text()
    interleaved = (ROOT / "contrib_ops/cuda/llm/cutlass_extensions/interleaved_numeric_conversion.h").read_text()
    uninterleave = device_function(interleaved.replace("CUTLASS_DEVICE", "__device__"), "fp4_e2m1x8_uninterleave")
    activation = common[common.index("struct ActivationParams") : common.index("  ActivationParams()")]
    activation += "};"
    enum = common[common.index("enum class ActivationType") : common.index("};") + 2]
    converter_start = details.index("template <typename AType, bool PairInterleaved = false>")
    converter_end = details.index("\n};", converter_start) + 3
    pair_start = source.index("template <typename T, bool FusedSwiGlu, bool EnableBias>\n__device__")
    pair_end = source.index("// NVFP4 schema weights", pair_start)
    kernel_start = source.index("__global__ void MoeGemvFp4RawKPackedKernel")
    kernel_start = source.rindex("template <", 0, kernel_start)
    kernel_end = source.index("template <typename T, bool FusedSwiGlu>\nvoid LaunchMoeGemvFp4RawNPacked", kernel_start)
    prefix = (
        """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
namespace std {
using cuda::std::is_same_v;
using cuda::std::conditional_t;
using cuda::std::numeric_limits;
}
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
namespace cutlass_kernels {
"""
        + enum
        + activation
        + "\n}\nnamespace cutlass { namespace detail {\n"
        + uninterleave
    )
    prefix += "\n}}\nnamespace fiv {\n" + details[converter_start:converter_end] + "\n}\n"
    body = (
        device_function(source, "DecodeE4M3Fn") + "\n" + source[pair_start:pair_end] + source[kernel_start:kernel_end]
    )
    flags = f"{str(fused).lower()},{str(bias).lower()}"
    lanes = ",4" if wide else ""
    if reuse:
        lanes = ",8,true"
    name = f"MoeGemvFp4RawKPackedKernel<{element_type},{flags}{lanes}>"
    activation_dtype = numpy.dtype(
        [
            ("activation_type", numpy.int32),
            ("swiglu_alpha", numpy.uint64),
            ("swiglu_beta", numpy.uint64),
            ("swiglu_limit", numpy.uint64),
            ("alpha", numpy.float32),
            ("beta", numpy.float32),
            ("limit", numpy.float32),
            ("swiglu_fusion", numpy.int32),
        ],
        align=True,
    )
    return compile_device_code(prefix + body, name), activation_dtype


def run_nvfp4_reuse(baseline, check_only, sanitizer_case=False, map_sources=True):
    candidate = (ROOT / "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu").read_text()
    random = numpy.random.default_rng(51)
    checks = 0
    for element_type in ("half", "__nv_bfloat16"):
        for rows in (70,) if sanitizer_case else (70, 80):
            for group in (5,) if sanitizer_case else (1, 2, 3, 4, 5, 8):
                for with_bias in (False, True):
                    old_kernel, activation_dtype = compile_nvfp4(baseline, element_type, True, with_bias, False)
                    new_kernel, _ = compile_nvfp4(candidate, element_type, True, with_bias, False, reuse=True)
                    experts, columns, reduction = 512, 1280, 2560
                    tokens = rows // 10 if map_sources else rows
                    identifiers = numpy.arange(rows, dtype=numpy.int32) // group
                    counts = numpy.bincount(identifiers, minlength=experts)
                    offsets = cupy.asarray(numpy.concatenate(([0], numpy.cumsum(counts))).astype(numpy.int64))
                    mapped_experts = cupy.asarray(identifiers)
                    mapped_rows = cupy.arange(rows, dtype=cupy.int32) % tokens if map_sources else numpy.uint64(0)
                    act = key_data(
                        random.standard_normal((tokens, reduction), dtype=numpy.float32) * 0.05, element_type
                    )
                    weight = cupy.asarray(
                        random.integers(0, 256, (experts, columns, reduction // 2), dtype=numpy.uint8)
                    )
                    scales = cupy.asarray(
                        random.integers(16, 64, (experts, columns, reduction // 16), dtype=numpy.uint8)
                    )
                    globals_ = cupy.full(experts, 0.125, dtype=cupy.float32)
                    biases = key_data(
                        random.standard_normal((experts, columns), dtype=numpy.float32) * 0.01, element_type
                    )
                    output_dtype = cupy.float16 if element_type == "half" else cupy.uint16
                    expected = cupy.full((rows, columns // 2), 0xDEAD, dtype=output_dtype)
                    actual = cupy.full_like(expected, 0xBEEF)
                    activation = numpy.zeros((), dtype=activation_dtype)
                    activation["alpha"] = 1.0
                    activation["limit"] = numpy.inf
                    head = (act, weight, scales, globals_, biases if with_bias else numpy.uint64(0))
                    tail = (
                        offsets,
                        mapped_experts,
                        numpy.int32(experts),
                        numpy.int64(columns * reduction // 2),
                        numpy.int64(columns * reduction // 16),
                        numpy.int32(columns),
                        numpy.int32(reduction),
                        activation[()],
                        mapped_rows,
                        numpy.int32(tokens),
                    )
                    old_args = (*head, expected, *tail)
                    new_args = (*head, actual, *tail)
                    grid = (rows, columns // 16)
                    old_kernel(grid, (128,), old_args)
                    new_kernel(grid, (128,), new_args)
                    cupy.testing.assert_array_equal(expected, actual)
                    checks += 1
                    if not check_only and not with_bias:
                        old_ms = graph_latency(old_kernel, grid, (128,), old_args)
                        new_ms = graph_latency(new_kernel, grid, (128,), new_args)
                        print(
                            json.dumps(
                                {
                                    "kernel": "nvfp4_fc1_expert_reuse",
                                    "dtype": element_type,
                                    "expanded_rows": rows,
                                    "rows_per_expert": group,
                                    "mapped_sources": map_sources,
                                    "baseline_ms": old_ms,
                                    "candidate_ms": new_ms,
                                    "speedup": old_ms / new_ms,
                                }
                            ),
                            flush=True,
                        )
    print(f"FC1 expert-reuse exact-output checks passed: {checks}, mapped sources: {map_sources}", flush=True)


def output_float(values, element_type):
    result = cupy.asnumpy(values)
    if element_type == "half":
        return result.astype(numpy.float32)
    return (result.astype(numpy.uint32) << 16).view(numpy.float32)


def run_nvfp4(baseline, check_only, sanitizer_case=False):
    source_path = ROOT / "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu"
    candidate = source_path.read_text()
    random = numpy.random.default_rng(37)
    wide = "int KLanes" in candidate
    checks = 0
    for element_type in ("half", "__nv_bfloat16"):
        for rows in (80,) if sanitizer_case else (10, 20, 70, 80):
            shapes = (
                ((514, 528, True),) if sanitizer_case else ((1280, 2560, True), (2560, 640, False), (514, 528, True))
            )
            for columns, reduction, fused in shapes:
                for with_bias in (False, True):
                    old_kernel, activation_dtype = compile_nvfp4(baseline, element_type, fused, with_bias, False)
                    use_wide = wide and reduction <= 1024
                    new_kernel, _ = compile_nvfp4(candidate, element_type, fused, with_bias, use_wide)
                    experts = 512
                    tokens = rows // 10
                    act = key_data(
                        random.standard_normal((tokens, reduction), dtype=numpy.float32) * 0.05, element_type
                    )
                    weight = cupy.asarray(
                        random.integers(0, 256, (experts, columns, reduction // 2), dtype=numpy.uint8)
                    )
                    scales = cupy.asarray(
                        random.integers(16, 64, (experts, columns, reduction // 16), dtype=numpy.uint8)
                    )
                    globals_ = cupy.full(experts, 0.125, dtype=cupy.float32)
                    biases = key_data(
                        random.standard_normal((experts, columns), dtype=numpy.float32) * 0.01, element_type
                    )
                    mapped_experts = cupy.asarray(random.integers(0, experts, rows, dtype=numpy.int32))
                    mapped_rows = cupy.asarray(random.integers(0, tokens, rows, dtype=numpy.int32))
                    offsets = cupy.zeros(experts + 1, dtype=cupy.int64)
                    output_columns = columns // 2 if fused else columns
                    output_dtype = cupy.float16 if element_type == "half" else cupy.uint16
                    expected = cupy.empty((rows, output_columns), dtype=output_dtype)
                    actual = cupy.empty_like(expected)
                    activation = numpy.zeros((), dtype=activation_dtype)
                    activation["alpha"] = 1.0
                    activation["limit"] = numpy.inf
                    common = (act, weight, scales, globals_, biases if with_bias else numpy.uint64(0))
                    tail = (
                        offsets,
                        mapped_experts,
                        numpy.int32(experts),
                        numpy.int64(columns * reduction // 2),
                        numpy.int64(columns * reduction // 16),
                        numpy.int32(columns),
                        numpy.int32(reduction),
                        activation[()],
                        mapped_rows,
                        numpy.int32(tokens),
                    )
                    old_args = (*common, expected, *tail)
                    new_args = (*common, actual, *tail)
                    old_grid = (rows, (columns + 15) // 16)
                    tile_columns = 32 if use_wide else 16
                    new_grid = (rows, (columns + tile_columns - 1) // tile_columns)
                    old_kernel(old_grid, (128,), old_args)
                    new_kernel(new_grid, (128,), new_args)
                    numpy.testing.assert_allclose(
                        output_float(actual, element_type),
                        output_float(expected, element_type),
                        rtol=0.002 if element_type == "half" else 0.01,
                        atol=0.0005,
                    )
                    checks += 1
                    if not check_only and not with_bias and columns != 514:
                        old_ms = graph_latency(old_kernel, old_grid, (128,), old_args)
                        new_ms = graph_latency(new_kernel, new_grid, (128,), new_args)
                        print(
                            json.dumps(
                                {
                                    "kernel": "nvfp4_fc1" if fused else "nvfp4_fc2",
                                    "dtype": element_type,
                                    "expanded_rows": rows,
                                    "baseline_ms": old_ms,
                                    "candidate_ms": new_ms,
                                    "speedup": old_ms / new_ms,
                                }
                            ),
                            flush=True,
                        )
    print(f"NVFP4 parity checks passed: {checks}", flush=True)


if __name__ == "__main__":
    main()
