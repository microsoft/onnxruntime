# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native targets for the existing C++ CPU test infrastructure."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//bazel:shared_tests.bzl", "shared_api_tests")

def core_tests():
    shared_api_tests()
    test_copts = select({
        "//bazel:windows": ["/utf-8"],
        "//conditions:default": [],
    })
    cc_library(
        name = "test_support",
        testonly = True,
        srcs = native.glob([
            "onnxruntime/test/util/*.cc",
            "onnxruntime/test/unittest_util/*.cc",
        ]) + [
            "onnxruntime/test/common/cuda_op_test_utils.cc",
            "onnxruntime/test/common/tensor_op_test_utils.cc",
            "onnxruntime/test/framework/dummy_provider.cc",
            "onnxruntime/test/internal_testing_ep/internal_testing_ep_static_kernels.cc",
            "onnxruntime/test/internal_testing_ep/internal_testing_execution_provider.cc",
            "onnxruntime/test/onnx/callback.cc",
            "onnxruntime/test/onnx/heap_buffer.cc",
            "onnxruntime/test/onnx/tensorprotoutils.cc",
            "onnxruntime/test/providers/compare_provider_test_utils.cc",
            "onnxruntime/test/providers/kernel_compute_test_utils.cc",
        ],
        includes = ["onnxruntime/test/util/include"],
        copts = test_copts,
        deps = [
            ":runtime",
            "@googletest//:gtest",
        ],
    )
    cc_library(
        name = "ort_test_main",
        testonly = True,
        srcs = ["onnxruntime/test/unittest_main/test_main.cc"],
        local_defines = ["main=OrtTestMain"],
        copts = test_copts,
        deps = [":test_support"],
    )
    session_srcs = [
        "onnxruntime/test/framework/inference_session_test.cc",
        "onnxruntime/test/framework/moe_offload_session_policy_test.cc",
        "onnxruntime/test/framework/more_session_option_test.cc",
        "onnxruntime/test/framework/session_state_test.cc",
    ]
    suites = {
        "framework_test": native.glob(
            ["onnxruntime/test/framework/*.cc"],
            exclude = session_srcs + ["onnxruntime/test/framework/dummy_provider.cc"],
        ),
        "session_test": session_srcs,
        "graph_test": native.glob(["onnxruntime/test/ir/*.cc"]),
        "common_runtime_test": native.glob(
            ["onnxruntime/test/common/*.cc"],
            exclude = [
                "onnxruntime/test/common/cuda_op_test_utils.cc",
                "onnxruntime/test/common/dnnl_op_test_utils.cc",
                "onnxruntime/test/common/narrow_test.cc",
                "onnxruntime/test/common/safeint_test.cc",
                "onnxruntime/test/common/semver_test.cc",
                "onnxruntime/test/common/span_utils_test.cc",
                "onnxruntime/test/common/string_utils_test.cc",
                "onnxruntime/test/common/tensor_op_test_utils.cc",
            ],
        ),
        "logging_test": native.glob(["onnxruntime/test/common/logging/*.cc"]),
        "platform_test": native.glob(["onnxruntime/test/platform/*.cc"]) + select({
            "//bazel:windows": native.glob([
                "onnxruntime/test/platform/windows/*.cc",
                "onnxruntime/test/platform/windows/logging/*.cc",
            ]),
            "//conditions:default": [],
        }),
        "provider_infrastructure_test": [
            "onnxruntime/test/providers/get_execution_providers_test.cc",
            "onnxruntime/test/providers/io_binding_test.cc",
            "onnxruntime/test/providers/memcpy_test.cc",
            "onnxruntime/test/providers/op_kernel_type_control_test.cc",
            "onnxruntime/test/providers/partitioning_utils_test.cc",
        ],
        "runtime_optimizer_test": native.glob(["onnxruntime/test/optimizer/runtime_optimization/*.cc"]),
        "flatbuffers_test": native.glob(["onnxruntime/test/flatbuffers/*.cc"]),
        "quantization_test": native.glob(["onnxruntime/test/quantization/*.cc"]),
        "lora_test": native.glob(["onnxruntime/test/lora/*.cc"]),
        "opaque_api_test": native.glob(["onnxruntime/test/opaque_api/*.cc"]),
        "ep_graph_test": native.glob(["onnxruntime/test/ep_graph/*.cc"]),
        "internal_testing_ep_test": [
            "onnxruntime/test/internal_testing_ep/internal_testing_tests.cc",
            "onnxruntime/test/internal_testing_ep/internal_testing_partitioning_tests.cc",
        ],
        "optimizer_test": native.glob(
            ["onnxruntime/test/optimizer/*.cc"],
            exclude = ["onnxruntime/test/optimizer/gqa_value_layout_transformer_test.cc"],
        ) + select({
            "//bazel:contrib_enabled": ["onnxruntime/test/optimizer/gqa_value_layout_transformer_test.cc"],
            "//conditions:default": [],
        }),
        "cpu_provider_test": native.glob(
            ["onnxruntime/test/providers/cpu/**/*.cc"],
            # This runner needs the separately downloaded ONNX model corpus.
            exclude = ["onnxruntime/test/providers/cpu/model_tests.cc"],
        ),
    }
    for name, srcs in suites.items():
        cc_test(
            name = name,
            srcs = srcs + ["tools/bazel/test_main.cc"],
            data = native.glob(["onnxruntime/test/testdata/**"]),
            copts = test_copts,
            # ORT's test environment already owns the process default logger.
            local_defines = ["SKIP_DEFAULT_LOGGER_TESTS"] if name == "logging_test" else [],
            deps = [":ort_test_main"],
            linkstatic = True,
            size = "large",
            timeout = "long",
        )
    for name, srcs in {
        "contrib_sparse_test": ["onnxruntime/test/contrib_ops/math/matmul_sparse_test.cc"],
        "contrib_quantization_test": ["onnxruntime/test/contrib_ops/dynamic_quantize_matmul_test.cc"],
        "contrib_attention_test": [
            "onnxruntime/test/contrib_ops/linear_attention_gates_op_test.cc",
            "onnxruntime/test/contrib_ops/linear_attention_op_test.cc",
        ],
    }.items():
        cc_test(
            name = name,
            srcs = srcs + ["tools/bazel/test_main.cc"],
            data = native.glob(["onnxruntime/test/testdata/**"]),
            copts = test_copts,
            deps = [":ort_test_main"],
            target_compatible_with = select({
                "//bazel:contrib_enabled": [],
                "//conditions:default": ["@platforms//:incompatible"],
            }),
            linkstatic = True,
            size = "large",
            timeout = "long",
        )
    native.test_suite(
        name = "core_tests",
        tests = [":" + name for name in suites],
    )
