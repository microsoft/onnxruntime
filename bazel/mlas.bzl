# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native x86-64 MLAS targets, with ISA flags confined to dispatched kernels."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

_LIB = "onnxruntime/core/mlas/lib/"

def _library(name, srcs, copts = []):
    cc_library(
        name = name,
        srcs = srcs,
        copts = copts,
        linkstatic = True,
        includes = [_LIB[:-1], "onnxruntime/core/mlas/inc"],
        deps = [":headers"],
    )

def mlas_libraries():
    """Declare the Linux x86-64 baseline and ISA-specific MLAS libraries."""
    _library(
        "mlas_sse2",
        [_LIB + "qgemm_kernel_sse.cpp"] + native.glob([
            _LIB + "x86_64/*Sse2.S",
            _LIB + "x86_64/cvtfp16a.S",
        ]),
        ["-msse2"],
    )
    _library(
        "mlas_avx",
        native.glob([
            _LIB + "x86_64/*Avx.S",
            _LIB + "intrinsics/avx/*.cpp",
        ]),
        ["-mavx"],
    )
    _library(
        "mlas_avx2",
        native.glob([
            _LIB + "x86_64/*Avx2.S",
            _LIB + "x86_64/*Fma3.S",
            _LIB + "x86_64/QgemvU8S8KernelAvxVnni.S",
            _LIB + "x86_64/cvtfp16Avx.S",
            _LIB + "intrinsics/avx2/*.cpp",
        ]) + [
            _LIB + name
            for name in [
                "sqnbitgemm_kernel_avx2.cpp",
                "sqnbitgemm_lut_kernel_avx2.cpp",
                "rotary_embedding_kernel_avx2.cpp",
                "qkv_quant_kernel_avx2.cpp",
                "layernorm_kernel_avx2.cpp",
            ]
        ],
        ["-mavx2", "-mfma", "-mf16c", "-ffp-contract=off"],
    )
    _library(
        "mlas_avx2vnni",
        [_LIB + "sqnbitgemm_kernel_avx2vnni.cpp"],
        ["-mavx2", "-mfma", "-mf16c", "-mavxvnni"],
    )
    _library(
        "mlas_avx512f",
        native.glob([
            _LIB + "x86_64/*Avx512F.S",
            _LIB + "intrinsics/avx512/*.cpp",
        ]) + [_LIB + "linear_attention_kernel_avx512f.cpp"],
        ["-mavx512f"],
    )
    _library(
        "mlas_avx512",
        native.glob([
            _LIB + "x86_64/*Avx512Core.S",
            _LIB + "x86_64/*Avx512Vnni.S",
            _LIB + "x86_64/*Amx*.S",
        ]) + [
            _LIB + name
            for name in [
                "sqnbitgemm_kernel_avx512.cpp",
                "sqnbitgemm_kernel_avx512vnni.cpp",
                "sqnbitgemm_kernel_avx512vnni_2bit.cpp",
                "qkv_quant_kernel_avx512vnni.cpp",
                "q4gemm_avx512.cpp",
                "qgemm_kernel_amx.cpp",
            ]
        ],
        ["-mfma", "-mf16c", "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mavx512vnni"],
    )
    cc_library(
        name = "mlas",
        linkstatic = True,
        includes = [_LIB[:-1], "onnxruntime/core/mlas/inc"],
        srcs = [
            _LIB + name + ".cpp"
            for name in [
                "platform",
                "threading",
                "sgemm",
                "halfgemm",
                "halfconv",
                "qgemm",
                "qdwconv",
                "convolve",
                "sconv_nchw_depthwise_multiplier_greater_than_1",
                "convsym",
                "pooling",
                "transpose",
                "reorder",
                "snchwc",
                "activate",
                "logistic",
                "tanh",
                "eltwise",
                "erf",
                "silu",
                "gelu",
                "compute",
                "dequantize",
                "quantize",
                "qgemm_kernel_default",
                "qladd",
                "qlmul",
                "qpostprocessor",
                "qlgavgpool",
                "qdwconv_kernelsize",
                "qnbitgemm",
                "qlutgemm",
                "flashattn",
                "flashattn_qkv",
                "flashattn_gqa",
                "qkv_quant",
                "linear_attention",
                "cast",
                "layernorm",
                "rotary_embedding",
                "saturation_check",
                "q4_dq",
                "q4gemm",
                "activate_fp16",
                "dwconv",
                "dgemm",
                "pooling_fp16",
                "qgemm_kernel_avx2",
                # This pack helper is used before ISA dispatch, including on AVX2-only hosts.
                "sqnbitgemm_kernel_avx512_2bit",
            ]
        ],
        deps = [
            ":headers",
            ":mlas_sse2",
            ":mlas_avx",
            ":mlas_avx2",
            ":mlas_avx2vnni",
            ":mlas_avx512f",
            ":mlas_avx512",
        ],
    )
