# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Existing public API tests linked against the native shared runtime."""

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//bazel:loader_tests.bzl", "loader_tests")
load("//bazel:platforms.bzl", "native_shared_library")
load("//bazel:shared_library_bundle.bzl", "shared_library_bundle")

def shared_api_tests():
    cc_library(
        name = "shared_api_test_common",
        testonly = True,
        srcs = [
            "onnxruntime/core/common/helper.cc",
            "onnxruntime/core/common/status.cc",
        ] + select({
            "//bazel:windows": ["onnxruntime/core/platform/windows/stacktrace.cc"],
            "//conditions:default": ["onnxruntime/core/platform/posix/stacktrace.cc"],
        }),
        deps = [
            ":headers",
            "@abseil-cpp//absl/debugging:stacktrace",
            "@abseil-cpp//absl/debugging:symbolize",
        ],
    )
    cc_library(
        name = "shared_api_test_support",
        testonly = True,
        srcs = [
            "onnxruntime/test/shared_lib/custom_op_utils.cc",
            "onnxruntime/test/shared_lib/utils.cc",
            "onnxruntime/test/util/file_util.cc",
            "onnxruntime/test/util/test_allocator.cc",
        ],
        includes = ["onnxruntime/test/util/include"],
        deps = [
            ":shared_api_test_common",
            "@googletest//:gtest",
        ] + select({
            "//bazel:linux": [":shared_api_test_runtime"],
            "//conditions:default": [":runtime_shared"],
        }),
    )
    native_shared_library(
        name = "custom_op_test_library",
        linux = "libcustom_op_library.so",
        windows = "custom_op_library.dll",
        macos = "libcustom_op_library.dylib",
        testonly = True,
        srcs = [
            "onnxruntime/test/testdata/custom_op_library/custom_op_library.cc",
            "onnxruntime/test/testdata/custom_op_library/cpu/cpu_ops.cc",
        ],
        win_def_file = select({
            "//bazel:windows": "onnxruntime/test/testdata/custom_op_library/custom_op_library.def",
            "//conditions:default": None,
        }),
        deps = [":shared_api_test_common"],
    )
    native.filegroup(
        name = "custom_op_test_interface",
        testonly = True,
        srcs = [":custom_op_test_library"],
        output_group = "interface_library",
    )
    cc_import(
        name = "custom_op_test_import",
        testonly = True,
        shared_library = ":custom_op_test_library",
        interface_library = select({
            "//bazel:windows": ":custom_op_test_interface",
            "//conditions:default": None,
        }),
    )
    plugin_libraries = []
    for plugin, exports in [
        ("example_plugin_ep", "example_plugin_ep_library.def"),
        ("example_plugin_ep_kernel_registry", "ep_lib.def"),
        ("example_plugin_ep_virt_gpu", "ep_lib.def"),
    ]:
        directory = "onnxruntime/test/autoep/library/" + plugin
        native_shared_library(
            name = plugin,
            linux = "lib" + plugin + ".so",
            windows = plugin + ".dll",
            macos = "lib" + plugin + ".dylib",
            testonly = True,
            srcs = native.glob([directory + "/**/*.cc"]),
            win_def_file = select({
                "//bazel:windows": directory + "/" + exports,
                "//conditions:default": None,
            }),
            deps = [":shared_api_test_common"],
        )
        plugin_libraries.append(":" + plugin)
    shared_library_bundle(
        name = "shared_api_test_runtime",
        testonly = True,
        runtime = ":runtime_library",
        plugins = plugin_libraries + select({
            "//bazel:linux_x86_64_cuda": [
                ":libonnxruntime_providers_shared.so",
                ":libonnxruntime_providers_cuda.so",
            ],
            "//conditions:default": [],
        }),
        target_compatible_with = ["@platforms//os:linux"],
    )
    cc_library(
        name = "global_threadpool_test_main",
        testonly = True,
        srcs = ["onnxruntime/test/global_thread_pools/test_main.cc"],
        local_defines = ["main=OrtTestMain", "USE_ONNXRUNTIME_DLL"],
        deps = [":shared_api_test_support"],
    )
    for name, srcs, main in [
        ("global_threadpool_test", ["onnxruntime/test/global_thread_pools/test_inference.cc"], ":global_threadpool_test_main"),
        ("custom_op_registration_test", [
            "onnxruntime/test/custom_op_registration/test_registercustomops.cc",
            "tools/bazel/shared_api_test_main.cc",
        ], ":shared_api_test_support"),
        ("plugin_api_test", [
            "onnxruntime/test/autoep/test_autoep_utils.cc",
            "onnxruntime/test/autoep/test_execution.cc",
            "tools/bazel/shared_api_test_main.cc",
        ], ":shared_api_test_support"),
        ("shared_api_test", [
            "onnxruntime/test/shared_lib/test_allocator.cc",
            "onnxruntime/test/shared_lib/test_data_copy.cc",
            "onnxruntime/test/shared_lib/test_env_creation.cc",
            "onnxruntime/test/shared_lib/test_ep_context_data_api.cc",
            "onnxruntime/test/shared_lib/test_experimental_api.cc",
            "onnxruntime/test/shared_lib/test_inference.cc",
            "onnxruntime/test/shared_lib/test_model_builder_api.cc",
            "onnxruntime/test/shared_lib/test_model_loading.cc",
            "onnxruntime/test/shared_lib/test_nontensor_types.cc",
            "onnxruntime/test/shared_lib/test_ort_format_models.cc",
            "onnxruntime/test/shared_lib/test_run_options.cc",
            "onnxruntime/test/shared_lib/test_session_options.cc",
            "onnxruntime/test/shared_lib/test_version.cc",
            "tools/bazel/shared_api_test_main.cc",
        ], ":shared_api_test_support"),
    ]:
        libraries = plugin_libraries if name == "plugin_api_test" else (
            [":custom_op_test_library"] if name in ["shared_api_test", "custom_op_registration_test"] else []
        )
        cc_test(
            name = name,
            srcs = srcs + ["tools/bazel/test_main.cc"],
            data = native.glob(["onnxruntime/test/testdata/**"]) + libraries,
            args = [
                "--ort_test_library=$(rootpath %s)" % library
                for library in libraries
            ],
            deps = [main] + ([":custom_op_test_import"] if name == "custom_op_registration_test" else []),
            env = select({
                "//bazel:windows": {},
                "//bazel:macos": {"DYLD_LIBRARY_PATH": "."},
                "//conditions:default": {"LD_LIBRARY_PATH": "."},
            }),
            linkstatic = True,
            size = "large",
            timeout = "long",
        )
    native.test_suite(
        name = "shared_api_tests",
        tests = [
            ":custom_op_registration_test",
            ":global_threadpool_test",
            ":plugin_api_test",
            ":shared_api_test",
        ],
    )
    loader_tests(plugin_libraries)
