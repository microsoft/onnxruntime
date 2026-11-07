# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native CUDA compilation and shared-provider targets."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load(":headers_only.bzl", "headers_only")

def cuda_libraries():
    """Declare optional CUDA provider artifacts without invoking another build system."""
    cuda_compatible = select({
        "//bazel:cuda_enabled": ["@platforms//os:linux", "@platforms//cpu:x86_64"],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    headers_only(
        name = "cuda_host_headers",
        deps = [":headers"],
    )
    cc_import(
        name = "providers_shared",
        shared_library = ":libonnxruntime_providers_shared.so",
        target_compatible_with = cuda_compatible,
    )
    cc_library(
        name = "cuda_headers",
        hdrs = native.glob([
            "onnxruntime/**/*.cuh",
        ]),
        defines = ["USE_CUDA", "ORT_ENABLE_STREAM", "NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING"],
        target_compatible_with = cuda_compatible,
        deps = [
            ":cuda_host_headers",
            "@cuda//:cuda_runtime",
            "@cuda//:cuda",
            "@cuda//:cublas",
            "@cuda//:curand",
            "@cuda//:cuda_headers",
            "@cuda//:cub",
            "@cuda//:thrust",
            "@ort_cudnn//:headers",
            "@ort_cudnn_frontend//:headers",
            "@ort_cutlass//:headers",
        ],
    )
    cuda_library(
        name = "cuda_kernels",
        srcs = native.glob([
            "onnxruntime/core/providers/cuda/**/*.cu",
        ], exclude = [
            "onnxruntime/core/providers/cuda/plugin/**",
        ]) + [
            "onnxruntime/contrib_ops/cuda/bert/attention_kv_cache.cu",
            "onnxruntime/contrib_ops/cuda/bert/attention_transpose.cu",
            "onnxruntime/contrib_ops/cuda/bert/unfused_attention.cu",
        ],
        copts = ["-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
        deps = [":cuda_headers"],
        target_compatible_with = cuda_compatible,
        tags = ["manual"],
    )
    cc_library(
        name = "cuda_provider",
        srcs = native.glob([
            "onnxruntime/core/providers/cuda/**/*.cc",
            "onnxruntime/core/providers/shared_library/*.cc",
        ], exclude = [
            "onnxruntime/core/providers/cuda/cuda_pch.cc",
            "onnxruntime/core/providers/cuda/plugin/**",
        ]) + [
            "onnxruntime/contrib_ops/cuda/bert/attention_kernel_options.cc",
            "onnxruntime/contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.cc",
        ],
        deps = [
            ":cuda_headers",
            ":cuda_kernels",
            ":providers_shared",
            "@abseil-cpp//absl/container:flat_hash_map",
            "@abseil-cpp//absl/strings",
            "@abseil-cpp//absl/synchronization",
        ],
        tags = ["manual"],
        target_compatible_with = cuda_compatible,
        alwayslink = True,
    )
    cc_binary(
        name = "libonnxruntime_providers_cuda.so",
        additional_linker_inputs = ["//bazel:cuda_provider.lds"],
        linkshared = True,
        linkopts = [
            "-ldl",
            "-pthread",
            "-Wl,-rpath,'$$ORIGIN'",
            "-Wl,--version-script=$(location //bazel:cuda_provider.lds)",
            "-Wl,--no-undefined",
            "-Wl,--gc-sections",
        ],
        deps = [":cuda_provider"],
        target_compatible_with = cuda_compatible,
        tags = ["manual"],
    )
