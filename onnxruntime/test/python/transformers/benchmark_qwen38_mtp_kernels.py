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
QSA_SOURCE = ROOT / "contrib_ops/cuda/sparse/packed_sparse_attention_indexer_impl.cu"


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


def qsa_prefix(source):
    header = (QSA_SOURCE.parent / "packed_sparse_attention_indexer_impl.h").read_text()
    math_source = (QSA_SOURCE.parent / "sparse_attention_indexer_device_math.cuh").read_text()
    topk_source = (ROOT / "core/providers/cuda/cu_inc/topk_warp_sort.cuh").read_text()
    params = header[header.index("struct PackedSparseAttentionIndexerParams") : header.index("};") + 2]
    fields = re.findall(r"\b(int|bool|float) (\w+) = [^;]+;", params)
    types = {"int": numpy.int32, "bool": numpy.bool_, "float": numpy.float32}
    params_dtype = numpy.dtype([(name, types[kind]) for kind, name in fields], align=True)
    helpers = source[
        source.index("template <typename T>\nstruct alignas") : source.index(
            "template <typename T>\n__global__ void QsaUpdateStateKernel"
        )
    ]
    prefix = """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
namespace std { using cuda::std::numeric_limits; }
#ifndef UINT_MAX
#define UINT_MAX 0xffffffffu
#endif
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef int int32_t;
typedef unsigned int uint32_t;
template<typename T> __device__ float to_float(T value) { return static_cast<float>(value); }
"""
    prefix += params + f"\nstatic_assert(sizeof(PackedSparseAttentionIndexerParams) == {params_dtype.itemsize});\n"
    prefix += """
constexpr int kHierarchicalTileBlocks = 8, kQwenNumHeads = 4, kQwenHeadSize = 128;
constexpr int kWarpSize = 32, kQwenCompressRatio = 4;
namespace psai { constexpr int kKeyStateLength = 0; }
namespace topk { constexpr uint64_t kPaddingSortKey = 0;
"""
    prefix += device_function(topk_source, "WarpReduceSum") + "\n"
    prefix += device_function(topk_source, "PackStableSortKey") + "\n}\n"
    prefix += helpers + "\n" + device_function(math_source, "SaiCausalThreshold") + "\n"
    return prefix, params_dtype


def compile_qsa(source, element_type, paired=False):
    prefix, params_dtype = qsa_prefix(source)
    body = source[
        source.index("template <typename T>\n__global__ void QsaScoreTileTopKKernel") : source.index(
            "__device__ __forceinline__ uint64_t QsaMergeRank"
        )
    ]
    name = f"{'QsaPairedScoreTileTopKKernel' if paired else 'QsaScoreTileTopKKernel'}<{element_type}>"
    return compile_device_code(prefix + body, name), params_dtype


def compile_qsa_emit(source):
    prefix, _ = qsa_prefix(source)
    topk_source = (ROOT / "core/providers/cuda/cu_inc/topk_warp_sort.cuh").read_text()
    prefix += "\nnamespace topk {\n" + device_function(topk_source, "UnpackStableSortIndex") + "\n}\n"
    start = source.index("__global__ void QsaEmitHierarchicalTopKKernel")
    end = source.index("// One block per query token.", start)
    return compile_device_code(prefix + source[start:end], "QsaEmitHierarchicalTopKKernel")


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


def compile_qsa_merge(source):
    prefix, _ = qsa_prefix(source)
    body = source[
        source.index("__device__ __forceinline__ uint64_t QsaMergeRank") : source.index(
            "__global__ void QsaEmitHierarchicalTopKKernel"
        )
    ]
    code = prefix + "\nconstexpr int kBoundedTopKMax = 512;\n" + body
    merge = compile_device_code(code, "QsaMergeTileTopKKernel")
    compact = compile_device_code(code, "QsaCompactTileTopKKernel") if "QsaCompactTileTopKKernel" in body else None
    return compact, merge


def merge_operations(kernels, data, rows, stride):
    compact, merge = kernels
    scratch = cupy.full(data.size, 0xBEEF, dtype=cupy.uint64)
    alternate = cupy.full_like(scratch, 0xDEAD)
    operations = []
    current = data
    width = 8
    if compact is not None and rows >= 4:
        chunks = (stride + 2047) // 2048
        compact_stride = chunks * 512
        operations.append(
            (compact, (chunks, rows), (256,), (current, scratch, numpy.int32(stride), numpy.int32(compact_stride)))
        )
        current, scratch = scratch, alternate
        stride = compact_stride
        width = 512
    while width < stride:
        pairs = (stride + 2 * width - 1) // (2 * width)
        operations.append(
            (
                merge,
                (rows * pairs,),
                (256,),
                (current, scratch, numpy.int32(stride), numpy.int32(width), numpy.int32(rows)),
            )
        )
        current, scratch = scratch, alternate if current is data else current
        width *= 2
    return operations, current[: rows * stride].reshape(rows, stride)[:, :512]


def launch_operations(operations):
    for kernel, grid, block, arguments in operations:
        kernel(grid, block, arguments)


def sequence_latency(operations):
    def launch(_grid, _block, _arguments):
        launch_operations(operations)

    return graph_latency(launch, (), (), ())


def run_qsa_merge(baseline, check_only, sanitizer_case=False):
    old_kernels = compile_qsa_merge(baseline)
    new_kernels = compile_qsa_merge(QSA_SOURCE.read_text())
    random = numpy.random.default_rng(17)
    checks = 0
    for rows in (8,) if sanitizer_case else (1, 2, 8):
        for stride in (4104,) if sanitizer_case else (2048, 4104, 65536):
            for visible in (9, stride) if sanitizer_case else (0, 1, 9, 1025, stride):
                for tied in (False, True):
                    values = numpy.zeros((rows, stride), dtype=numpy.uint64)
                    if tied:
                        values[:, :visible] = numpy.uint64(1 << 63) + numpy.arange(
                            visible, 0, -1, dtype=numpy.int64
                        ).astype(numpy.uint64)
                    else:
                        values[:, :visible] = random.integers(1, 2**63, (rows, visible), dtype=numpy.uint64)
                    values = numpy.sort(values.reshape(rows, -1, 8), axis=2)[:, :, ::-1].copy().reshape(rows, stride)
                    data = cupy.asarray(values)
                    expected = cupy.sort(data, axis=1)[:, ::-1][:, :512]
                    old_ops, old_result = merge_operations(old_kernels, data, rows, stride)
                    new_ops, new_result = merge_operations(new_kernels, data, rows, stride)
                    launch_operations(old_ops)
                    launch_operations(new_ops)
                    cupy.testing.assert_array_equal(expected, old_result)
                    cupy.testing.assert_array_equal(expected, new_result)
                    checks += 1
                    if not check_only and rows in (2, 8) and stride == 65536 and visible in (1025, stride) and not tied:
                        old_ms = sequence_latency(old_ops)
                        new_ms = sequence_latency(new_ops)
                        print(
                            json.dumps(
                                {
                                    "kernel": "qsa_merge_hierarchy",
                                    "rows": rows,
                                    "visible_blocks": visible,
                                    "baseline_ms": old_ms,
                                    "candidate_ms": new_ms,
                                    "speedup": old_ms / new_ms,
                                }
                            ),
                            flush=True,
                        )
    print(f"QSA compact-merge oracle checks passed: {checks}", flush=True)


def run_qsa(baseline, check_only, sanitizer_case=False):
    candidate = QSA_SOURCE.read_text()
    random = numpy.random.default_rng(42)
    checks = 0
    old_mergers = compile_qsa_merge(baseline)
    new_mergers = compile_qsa_merge(candidate)
    old_emit = compile_qsa_emit(baseline)
    new_emit = compile_qsa_emit(candidate)
    for element_type in ("float", "half", "__nv_bfloat16"):
        old_kernel, params_dtype = compile_qsa(baseline, element_type)
        new_kernel, _ = compile_qsa(candidate, element_type)
        paired_kernel, _ = compile_qsa(candidate, element_type, paired=True)
        for rows in (8,) if sanitizer_case else (1, 2, 5, 7, 8):
            for capacity in (4099,) if sanitizer_case else (2048, 4099, 65536):
                keys = key_data(random.standard_normal((capacity, 128), dtype=numpy.float32), element_type)
                queries = cupy.asarray(random.standard_normal((rows, 4, 128), dtype=numpy.float32))
                for visible in (9,) if sanitizer_case else (0, 1, 9, 1024, capacity):
                    params = numpy.zeros((), dtype=params_dtype)
                    for name, value in {
                        "batch_size": 1,
                        "total_tokens": rows,
                        "num_heads": 4,
                        "head_size": 128,
                        "compress_ratio": 4,
                        "state_capacity": capacity,
                        "capacity": 2051,
                        "block_topk": 512,
                        "scale": 0.0883883461356163,
                    }.items():
                        params[name] = value
                    cumulative = cupy.asarray([0, rows], dtype=cupy.int32)
                    past = cupy.asarray([visible * 4], dtype=cupy.int32)
                    positions = cupy.zeros(rows, dtype=cupy.int64)
                    lengths = cupy.asarray([[visible, 0]], dtype=cupy.int32)
                    overflow = cupy.zeros(1, dtype=cupy.int32)
                    tiles = (capacity + 7) // 8
                    expected = cupy.full((rows, tiles * 8), 0xDEAD, dtype=cupy.uint64)
                    actual = cupy.full_like(expected, 0xBEEF)
                    common = (keys, queries, cumulative, past, positions, lengths, overflow)
                    old_args = (*common, expected, numpy.int32(tiles), params[()])
                    new_args = (*common, actual, numpy.int32(tiles), params[()])
                    old_grid = (
                        (min(tiles, 128), rows)
                        if "const int token = static_cast<int>(blockIdx.y)" in baseline
                        else (rows * tiles,)
                    )
                    use_paired = rows >= 4
                    candidate_kernel = paired_kernel if use_paired else new_kernel
                    new_grid = (min(tiles, 128), (rows + 1) // 2 if use_paired else rows)
                    new_block = (256,) if use_paired else (1024,)
                    old_kernel(old_grid, (1024,), old_args)
                    candidate_kernel(new_grid, new_block, new_args)
                    cupy.testing.assert_array_equal(expected, actual)
                    old_merge_ops, old_result = merge_operations(old_mergers, expected, rows, tiles * 8)
                    new_merge_ops, new_result = merge_operations(new_mergers, actual, rows, tiles * 8)
                    expected_indices = cupy.full((rows, 2051), -77, dtype=cupy.int32)
                    actual_indices = cupy.full_like(expected_indices, -88)
                    expected_counts = cupy.full(rows, -77, dtype=cupy.int32)
                    actual_counts = cupy.full_like(expected_counts, -88)
                    emit_common = (cumulative, past, positions, lengths, overflow)
                    old_emit_args = (
                        old_result,
                        *emit_common,
                        numpy.int32(old_result.strides[0] // 8),
                        expected_indices,
                        expected_counts,
                        params[()],
                    )
                    new_emit_args = (
                        new_result,
                        *emit_common,
                        numpy.int32(new_result.strides[0] // 8),
                        actual_indices,
                        actual_counts,
                        params[()],
                    )
                    old_ops = [
                        (old_kernel, old_grid, (1024,), old_args),
                        *old_merge_ops,
                        (old_emit, (rows,), (128,), old_emit_args),
                    ]
                    new_ops = [
                        (candidate_kernel, new_grid, new_block, new_args),
                        *new_merge_ops,
                        (new_emit, (rows,), (128,), new_emit_args),
                    ]
                    launch_operations(old_ops)
                    launch_operations(new_ops)
                    cupy.testing.assert_array_equal(expected_indices, actual_indices)
                    cupy.testing.assert_array_equal(expected_counts, actual_counts)
                    checks += 1
                    if not check_only and rows in (2, 8) and capacity == 65536 and visible in (1024, capacity):
                        old_ms = graph_latency(old_kernel, old_grid, (1024,), old_args)
                        new_ms = graph_latency(candidate_kernel, new_grid, new_block, new_args)
                        print(
                            json.dumps(
                                {
                                    "kernel": "qsa_score",
                                    "dtype": element_type,
                                    "rows": rows,
                                    "visible_blocks": visible,
                                    "baseline_ms": old_ms,
                                    "candidate_ms": new_ms,
                                    "speedup": old_ms / new_ms,
                                }
                            ),
                            flush=True,
                        )
                        old_pipeline_ms = sequence_latency(old_ops)
                        new_pipeline_ms = sequence_latency(new_ops)
                        print(
                            json.dumps(
                                {
                                    "kernel": "qsa_score_merge_emit",
                                    "dtype": element_type,
                                    "rows": rows,
                                    "visible_blocks": visible,
                                    "baseline_ms": old_pipeline_ms,
                                    "candidate_ms": new_pipeline_ms,
                                    "speedup": old_pipeline_ms / new_pipeline_ms,
                                }
                            ),
                            flush=True,
                        )
        checks += qsa_ragged_replay(old_kernel, new_kernel, params_dtype, element_type, random)
        checks += qsa_ragged_replay(
            old_kernel, paired_kernel, params_dtype, element_type, random, paired=True, baseline=baseline
        )
    print(f"QSA exact-key checks passed: {checks}", flush=True)


def qsa_ragged_replay(old_kernel, new_kernel, params_dtype, element_type, random, paired=False, baseline=None):
    rows, capacity, batches = 8, 4099, 1 if paired else 3
    params = numpy.zeros((), dtype=params_dtype)
    for name, value in {
        "batch_size": batches,
        "total_tokens": rows,
        "num_heads": 4,
        "head_size": 128,
        "compress_ratio": 4,
        "state_capacity": capacity,
        "capacity": 2051,
        "block_topk": 512,
        "scale": 0.0883883461356163,
    }.items():
        params[name] = value
    keys = key_data(random.standard_normal((batches, capacity, 128), dtype=numpy.float32), element_type)
    query = cupy.asarray(random.standard_normal((rows, 4, 128), dtype=numpy.float32))
    cumulative = cupy.asarray([0, rows] if paired else [0, 0, 7, 8], dtype=cupy.int32)
    past = cupy.zeros(batches, dtype=cupy.int32)
    positions = cupy.zeros(rows, dtype=cupy.int64)
    lengths = cupy.zeros((batches, 2), dtype=cupy.int32)
    overflow = cupy.zeros(batches, dtype=cupy.int32)
    tiles = (capacity + 7) // 8
    expected = cupy.empty((rows, tiles * 8), dtype=cupy.uint64)
    actual = cupy.empty_like(expected)
    common = (keys, query, cumulative, past, positions, lengths, overflow)
    old_args = (*common, expected, numpy.int32(tiles), params[()])
    new_args = (*common, actual, numpy.int32(tiles), params[()])
    old_ops = [(old_kernel, (min(tiles, 128), rows), (1024,), old_args)]
    new_ops = [
        (new_kernel, (min(tiles, 128), (rows + 1) // 2 if paired else rows), (256,) if paired else (1024,), new_args)
    ]
    if paired:
        old_merge_ops, old_result = merge_operations(compile_qsa_merge(baseline), expected, rows, tiles * 8)
        new_merge_ops, new_result = merge_operations(compile_qsa_merge(QSA_SOURCE.read_text()), actual, rows, tiles * 8)
        expected_indices = cupy.empty((rows, 2051), dtype=cupy.int32)
        actual_indices = cupy.empty_like(expected_indices)
        expected_counts = cupy.empty(rows, dtype=cupy.int32)
        actual_counts = cupy.empty_like(expected_counts)
        emit_common = (cumulative, past, positions, lengths, overflow)
        old_ops += [
            *old_merge_ops,
            (
                compile_qsa_emit(baseline),
                (rows,),
                (128,),
                (
                    old_result,
                    *emit_common,
                    numpy.int32(old_result.strides[0] // 8),
                    expected_indices,
                    expected_counts,
                    params[()],
                ),
            ),
        ]
        new_ops += [
            *new_merge_ops,
            (
                compile_qsa_emit(QSA_SOURCE.read_text()),
                (rows,),
                (128,),
                (
                    new_result,
                    *emit_common,
                    numpy.int32(new_result.strides[0] // 8),
                    actual_indices,
                    actual_counts,
                    params[()],
                ),
            ),
        ]
    cupy.cuda.Stream.null.synchronize()
    stream = cupy.cuda.Stream(non_blocking=True)
    stream.begin_capture()
    with stream:
        launch_operations(old_ops)
        launch_operations(new_ops)
    graph = stream.end_capture()
    checks = 0
    overflow_cases = ((0,), (1,)) if paired else ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1))
    for visible in (0, 1, 9, 1025, capacity):
        for overflow_values in overflow_cases:
            lengths.set(
                numpy.array([[visible, 0]] if paired else [[0, 0], [visible, 0], [visible // 2, 0]], dtype=numpy.int32)
            )
            past.set(numpy.array([visible * 4] if paired else [0, visible * 4, (visible // 2) * 4], dtype=numpy.int32))
            overflow.set(numpy.array(overflow_values, dtype=numpy.int32))
            expected.fill(0xDEAD)
            actual.fill(0xBEEF)
            cupy.cuda.Stream.null.synchronize()
            graph.launch(stream)
            stream.synchronize()
            cupy.testing.assert_array_equal(expected, actual)
            if paired:
                cupy.testing.assert_array_equal(expected_indices, actual_indices)
                cupy.testing.assert_array_equal(expected_counts, actual_counts)
            checks += 1
    return checks


def main():
    parser = argparse.ArgumentParser(description="Standalone NVRTC parity/timing probe; does not rebuild ORT.")
    parser.add_argument("--baseline-stdin", action="store_true", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--sanitizer-case", action="store_true")
    parser.add_argument("--kernel", choices=("qsa", "qsa_merge", "nvfp4", "nvfp4_reuse"), default="qsa")
    args = parser.parse_args()
    baseline = sys.stdin.read()
    if args.kernel == "qsa":
        run_qsa(baseline, args.check_only, args.sanitizer_case)
    elif args.kernel == "qsa_merge":
        run_qsa_merge(baseline, args.check_only, args.sanitizer_case)
    elif args.kernel == "nvfp4_reuse":
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
