# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native language bindings sharing the Bazel-built runtime."""

load("@aspect_rules_js//js:defs.bzl", "js_test")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_dotnet//dotnet:defs.bzl", "csharp_library", "csharp_test")
load("@rules_go//go:def.bzl", "go_library", "go_test")
load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("@rules_python//python:defs.bzl", "py_test")
load(":jni_headers.bzl", "jni_headers")

def language_bindings():
    """Declare language bindings using native Bazel compilation actions."""
    cuda_defines = select({
        "//bazel:cuda_enabled": ["USE_CUDA"],
        "//conditions:default": [],
    })
    cc_library(
        name = "api_headers",
        hdrs = native.glob(["include/onnxruntime/**/*.h"]),
        includes = ["include", "include/onnxruntime", "include/onnxruntime/core/session"],
    )
    cc_import(
        name = "runtime_shared",
        shared_library = ":libonnxruntime.so",
    )
    go_library(
        name = "go_binding",
        srcs = native.glob(
            ["go/onnxruntime/*.go", "go/onnxruntime/*.c", "go/onnxruntime/*.h"],
            exclude = ["go/onnxruntime/*_test.go"],
        ),
        cgo = True,
        clinkopts = ["-ldl"],
        importpath = "github.com/microsoft/onnxruntime/go/onnxruntime",
    )
    go_test(
        name = "go_test",
        srcs = native.glob(
            ["go/onnxruntime/*_test.go"],
            exclude = ["go/onnxruntime/qwen3_test.go"],
        ),
        data = [":libonnxruntime.so", "go/onnxruntime/ort.go"] + native.glob(["go/testdata/**"]),
        embed = [":go_binding"],
        env = {"ORT_LIB_PATH": "../../$(rootpath :libonnxruntime.so)"},
        rundir = "go/onnxruntime",
    )
    java_library(
        name = "java_binding",
        srcs = native.glob([
            "java/src/main/java/**/*.java",
            "java/src/main/jvm/**/*.java",
        ]),
    )
    jni_headers(
        name = "java_native_headers",
        java = ":java_binding",
    )
    cc_binary(
        name = "libonnxruntime4j_jni.so",
        srcs = native.glob(
            ["java/src/main/native/*.c"],
            exclude = ["java/src/main/native/*Training*.c"],
        ) + ["java/src/main/native/OrtJniUtil.h"],
        includes = ["java/src/main/native"],
        local_defines = cuda_defines,
        linkopts = ["-Wl,-rpath,'$$ORIGIN'"],
        linkshared = True,
        deps = [
            ":api_headers",
            ":generated_headers",
            ":java_native_headers",
            ":runtime_shared",
            "@bazel_tools//tools/jdk:jni",
        ],
    )
    java_test(
        name = "java_smoke_test",
        srcs = ["tools/bazel/RuntimeSmokeTest.java"],
        args = [
            "$(rootpath go/testdata/add_f32.onnx)",
            "$(rootpath :libonnxruntime.so)",
            "$(rootpath :libonnxruntime4j_jni.so)",
        ],
        data = [
            "go/testdata/add_f32.onnx",
            ":libonnxruntime.so",
            ":libonnxruntime4j_jni.so",
        ],
        main_class = "RuntimeSmokeTest",
        use_testrunner = False,
        deps = [":java_binding"],
    )
    cc_binary(
        name = "onnxruntime_pybind11_state.so",
        srcs = native.glob(
            ["onnxruntime/python/*.cc"],
            exclude = ["onnxruntime/python/onnxruntime_pybind_cuda_quant.cc"],
        ),
        additional_linker_inputs = ["//bazel:python.lds"],
        copts = ["-fvisibility=hidden"],
        data = select({
            "//bazel:cuda_enabled": [
                ":libonnxruntime_providers_cuda.so",
                ":libonnxruntime_providers_shared.so",
            ],
            "//conditions:default": [],
        }),
        local_defines = cuda_defines,
        linkopts = ["-Wl,--version-script=$(location //bazel:python.lds)", "-Wl,-Bsymbolic"],
        linkshared = True,
        linkstatic = True,
        deps = [
            ":runtime",
            "@ort_pybind11//:headers",
            "@ort_pip//numpy:numpy_headers",
            "@rules_python//python/cc:current_py_cc_headers",
        ] + select({
            "//bazel:cuda_enabled": [":cuda_headers"],
            "//conditions:default": [],
        }),
    )
    py_test(
        name = "python_smoke_test",
        srcs = ["tools/bazel/python_smoke_test.py"],
        main = "tools/bazel/python_smoke_test.py",
        args = ["$(rootpath go/testdata/add_f32.onnx)"],
        data = [":onnxruntime_pybind11_state.so", "go/testdata/add_f32.onnx"],
        deps = ["@ort_pip//numpy"],
    )
    cc_binary(
        name = "libonnxruntime_binding.so",
        srcs = native.glob(["js/node/src/*.cc", "js/node/src/*.h"]),
        local_defines = ["NAPI_VERSION=6", "ORT_API_MANUAL_INIT"] + cuda_defines,
        linkopts = ["-Wl,-rpath,'$$ORIGIN'"],
        linkshared = True,
        deps = [
            ":api_headers",
            ":runtime_shared",
            "@ort_node_headers//:headers",
            "@ort_node_addon_api//:headers",
        ],
    )
    copy_file(
        name = "node_binding",
        src = ":libonnxruntime_binding.so",
        out = "onnxruntime_binding.node",
    )
    js_test(
        name = "node_smoke_test",
        entry_point = "tools/bazel/node_smoke_test.cjs",
        args = [
            "$(rootpath :onnxruntime_binding.node)",
            "$(rootpath go/testdata/add_f32.onnx)",
        ],
        data = [
            ":onnxruntime_binding.node",
            ":libonnxruntime.so",
            "go/testdata/add_f32.onnx",
        ],
    )
    csharp_library(
        name = "csharp_binding",
        srcs = native.glob(["csharp/src/Microsoft.ML.OnnxRuntime/**/*.shared.cs"]),
        allow_unsafe_blocks = True,
        keyfile = "csharp/OnnxRuntime.snk",
        out = "Microsoft.ML.OnnxRuntime",
        target_frameworks = ["net8.0"],
        deps = ["@ort_dotnet_tensors//:tensors"],
    )
    csharp_test(
        name = "csharp_smoke_test",
        srcs = ["tools/bazel/RuntimeSmokeTest.cs"],
        args = [
            "$(rootpath go/testdata/add_f32.onnx)",
            "$(rootpath :libonnxruntime.so)",
        ],
        data = ["go/testdata/add_f32.onnx", ":libonnxruntime.so"],
        target_frameworks = ["net10.0"],
        deps = [":csharp_binding"],
    )
