# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native desktop MLAS targets, with ISA flags confined to dispatched kernels."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

_LIB = "onnxruntime/core/mlas/lib/"

def _library(name, srcs, copts = [], windows_copts = []):
    cc_library(
        name = name,
        srcs = srcs,
        copts = select({
            "//bazel:windows": windows_copts,
            "//conditions:default": copts,
        }),
        linkstatic = True,
        includes = [_LIB[:-1], _LIB + "amd64", "onnxruntime/core/mlas/inc"],
        target_compatible_with = ["@platforms//cpu:x86_64"],
        deps = [":headers"],
    )

def mlas_libraries():
    """Declare native desktop MLAS kernels with per-ISA compiler options."""
    cc_library(
        name = "mlas_neon",
        srcs = [
            _LIB + "aarch64/" + name + ".S"
            for name in [
                "ConvSymS8KernelDot",
                "ConvSymS8KernelDotLd64",
                "ConvSymU8KernelDot",
                "ConvSymS8KernelNeon",
                "ConvSymU8KernelNeon",
                "DepthwiseQConvSymS8KernelNeon",
                "DepthwiseQConvSymU8KernelNeon",
                "DepthwiseQConvKernelSize9Neon",
                "QgemmU8X8KernelNeon",
                "QgemmS8S8KernelNeon",
                "QgemmU8X8KernelUdot",
                "QgemmS8S8KernelSdot",
                "SgemmKernelNeon",
                "SgemvKernelNeon",
                "SymQgemmS8KernelNeon",
                "SymQgemmS8KernelSdot",
                "SymQgemmS8KernelSdotLd64",
            ]
        ] + [
            _LIB + name + ".cpp"
            for name in [
                "qgemm_kernel_neon",
                "qgemm_kernel_udot",
                "qgemm_kernel_sdot",
                "qnbitgemm_kernel_neon",
                "sqnbitgemm_kernel_neon_fp32",
                "rotary_embedding_kernel_neon",
                "qkv_quant_kernel_neon",
                "hgemm_kernel_neon",
                "softmax_kernel_neon",
                "eltwise_kernel_neon",
                "sconv_nchw_depthwise_multiplier_1",
                "linear_attention_kernel_neon",
            ]
        ],
        includes = [_LIB[:-1], "onnxruntime/core/mlas/inc"],
        linkstatic = True,
        target_compatible_with = ["@platforms//cpu:aarch64"],
        deps = [":headers", ":mlas_neon_dot", ":mlas_neon_i8mm", ":mlas_neon_bf16"],
    )
    for suffix, features, sources in [
        ("dot", "dotprod", ["sqnbitgemm_kernel_neon_int8.cpp", "sqnbitgemm_kernel_neon_int8_2bit.cpp"]),
        ("i8mm", "i8mm", ["sqnbitgemm_kernel_neon_int8_i8mm.cpp"]),
        ("bf16", "bf16", ["aarch64/SbgemmKernelNeon.S", "sbgemm_kernel_neon.cpp"]),
    ]:
        cc_library(
            name = "mlas_neon_" + suffix,
            srcs = [_LIB + source for source in sources],
            copts = ["-march=armv8.2-a+" + features],
            includes = [_LIB[:-1], "onnxruntime/core/mlas/inc"],
            linkstatic = True,
            target_compatible_with = ["@platforms//cpu:aarch64"],
            deps = [":headers"],
        )
    _library(
        "mlas_windows_asm",
        select({
            "//bazel:windows": native.glob([_LIB + "amd64/*.asm"]),
            "//conditions:default": [],
        }),
    )
    _library(
        "mlas_sse2",
        [_LIB + "qgemm_kernel_sse.cpp"] + select({
            "//bazel:windows": [_LIB + "qgemm_kernel_sse41.cpp"],
            "//conditions:default": native.glob([
                _LIB + "x86_64/*Sse2.S",
                _LIB + "x86_64/cvtfp16a.S",
            ]),
        }),
        ["-msse2"],
    )
    _library(
        "mlas_avx",
        native.glob([_LIB + "intrinsics/avx/*.cpp"]) + select({
            "//bazel:windows": [],
            "//conditions:default": native.glob([
                _LIB + "x86_64/*Avx.S",
            ]),
        }),
        ["-mavx"],
        ["/arch:AVX"],
    )
    _library(
        "mlas_avx2",
        native.glob([_LIB + "intrinsics/avx2/*.cpp"]) + select({
            "//bazel:windows": [],
            "//conditions:default": native.glob([
                _LIB + "x86_64/*Avx2.S",
                _LIB + "x86_64/*Fma3.S",
                _LIB + "x86_64/QgemvU8S8KernelAvxVnni.S",
                _LIB + "x86_64/cvtfp16Avx.S",
            ]),
        }) + [
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
        ["/arch:AVX2"],
    )
    _library(
        "mlas_avx2vnni",
        [_LIB + "sqnbitgemm_kernel_avx2vnni.cpp"],
        ["-mavx2", "-mfma", "-mf16c", "-mavxvnni"],
        ["/arch:AVX2"],
    )
    _library(
        "mlas_avx512f",
        native.glob([_LIB + "intrinsics/avx512/*.cpp"]) + select({
            "//bazel:windows": [],
            "//conditions:default": native.glob([
                _LIB + "x86_64/*Avx512F.S",
            ]),
        }) + [_LIB + "linear_attention_kernel_avx512f.cpp"],
        ["-mavx512f"],
        ["/arch:AVX512"],
    )
    _library(
        "mlas_avx512",
        select({
            "//bazel:windows": [],
            "//conditions:default": native.glob([
                _LIB + "x86_64/*Avx512Core.S",
                _LIB + "x86_64/*Avx512Vnni.S",
                _LIB + "x86_64/*Amx*.S",
            ]),
        }) + [
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
        ["/arch:AVX512"],
    )
    cc_library(
        name = "mlas",
        # ISA archives call baseline helpers after this archive has been scanned on ELF linkers.
        alwayslink = True,
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
                # This pack helper is used before ISA dispatch, including on AVX2-only hosts.
                "sqnbitgemm_kernel_avx512_2bit",
            ]
        ] + select({
            "//bazel:x86_64": [_LIB + "qgemm_kernel_avx2.cpp"],
            "//conditions:default": [],
        }),
        deps = [":headers"] + select({
            "//bazel:aarch64": [":mlas_neon"],
            "//conditions:default": [
                ":mlas_sse2",
                ":mlas_avx",
                ":mlas_avx2",
                ":mlas_avx2vnni",
                ":mlas_avx512f",
                ":mlas_avx512",
                ":mlas_windows_asm",
            ],
        }),
    )
