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
load(":platforms.bzl", "ORIGIN_RPATH", "desktop_shared_library")

def language_bindings():
    """Declare language bindings using native Bazel compilation actions."""
    cc_import(
        name = "node_import_library",
        interface_library = "@ort_node_import_library//file",
        system_provided = True,
        target_compatible_with = ["@platforms//os:windows"],
    )
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
        shared_library = ":runtime_library",
    )
    go_library(
        name = "go_binding",
        srcs = native.glob(
            ["go/onnxruntime/*.go", "go/onnxruntime/*.c", "go/onnxruntime/*.h"],
            exclude = ["go/onnxruntime/*_test.go"],
        ),
        cgo = True,
        clinkopts = select({
            "//bazel:linux": ["-ldl"],
            "//conditions:default": [],
        }),
        importpath = "github.com/microsoft/onnxruntime/go/onnxruntime",
    )
    go_test(
        name = "go_test",
        srcs = native.glob(
            ["go/onnxruntime/*_test.go"],
            exclude = ["go/onnxruntime/qwen3_test.go"],
        ),
        data = [":runtime_library", "go/onnxruntime/ort.go"] + native.glob(["go/testdata/**"]),
        embed = [":go_binding"],
        env = {"ORT_LIB_PATH": "../../$(rootpath :runtime_library)"},
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
    desktop_shared_library(
        name = "java_native_library",
        linux = "libonnxruntime4j_jni.so",
        windows = "onnxruntime4j_jni.dll",
        macos = "libonnxruntime4j_jni.dylib",
        srcs = native.glob(
            ["java/src/main/native/*.c"],
            exclude = ["java/src/main/native/*Training*.c"],
        ) + ["java/src/main/native/OrtJniUtil.h"],
        includes = ["java/src/main/native"],
        local_defines = cuda_defines,
        linkopts = ORIGIN_RPATH,
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
            "$(rootpath :runtime_library)",
            "$(rootpath :java_native_library)",
        ],
        data = [
            "go/testdata/add_f32.onnx",
            ":runtime_library",
            ":java_native_library",
        ],
        main_class = "RuntimeSmokeTest",
        use_testrunner = False,
        deps = [":java_binding"],
    )
    desktop_shared_library(
        name = "python_native_library",
        linux = "onnxruntime_pybind11_state.so",
        windows = "onnxruntime_pybind11_state.dll",
        macos = "onnxruntime_pybind11_state.dylib",
        srcs = native.glob(
            ["onnxruntime/python/*.cc"],
            exclude = ["onnxruntime/python/onnxruntime_pybind_cuda_quant.cc"],
        ),
        additional_linker_inputs = ["//bazel:python.lds"],
        copts = select({
            "//bazel:windows": [],
            "//conditions:default": ["-fvisibility=hidden"],
        }),
        data = select({
            "//bazel:cuda_enabled": [
                ":libonnxruntime_providers_cuda.so",
                ":libonnxruntime_providers_shared.so",
            ],
            "//conditions:default": [],
        }),
        local_defines = cuda_defines,
        linkopts = select({
            "//bazel:windows": ["/EXPORT:PyInit_onnxruntime_pybind11_state"],
            "//bazel:macos": ["-Wl,-exported_symbol,_PyInit_onnxruntime_pybind11_state", "-undefined", "dynamic_lookup"],
            "//conditions:default": ["-Wl,--version-script=$(location //bazel:python.lds)", "-Wl,-Bsymbolic"],
        }),
        linkstatic = True,
        deps = [
            ":runtime",
            "@ort_pybind11//:headers",
            "@ort_pip//numpy:numpy_headers",
            "@rules_python//python/cc:current_py_cc_headers",
        ] + select({
            "//bazel:windows": ["@rules_python//python/cc:current_py_cc_libs"],
            "//conditions:default": [],
        }) + select({
            "//bazel:cuda_enabled": [":cuda_headers"],
            "//conditions:default": [],
        }),
    )
    copy_file(
        name = "python_windows_extension",
        src = ":onnxruntime_pybind11_state.dll",
        out = "onnxruntime_pybind11_state.pyd",
        target_compatible_with = ["@platforms//os:windows"],
    )
    copy_file(
        name = "python_macos_extension",
        src = ":onnxruntime_pybind11_state.dylib",
        out = "python_macos/onnxruntime_pybind11_state.so",
        target_compatible_with = ["@platforms//os:macos"],
    )
    py_test(
        name = "python_smoke_test",
        srcs = ["tools/bazel/python_smoke_test.py"],
        main = "tools/bazel/python_smoke_test.py",
        args = ["$(rootpath go/testdata/add_f32.onnx)"],
        data = ["go/testdata/add_f32.onnx"] + select({
            "//bazel:windows": [":python_windows_extension"],
            "//bazel:macos": [":python_macos_extension"],
            "//conditions:default": [":onnxruntime_pybind11_state.so"],
        }),
        imports = select({
            "//bazel:macos": ["python_macos"],
            "//conditions:default": [],
        }),
        deps = ["@ort_pip//numpy"],
    )
    desktop_shared_library(
        name = "node_native_library",
        linux = "libonnxruntime_binding.so",
        windows = "onnxruntime_binding.dll",
        macos = "libonnxruntime_binding.dylib",
        srcs = native.glob(["js/node/src/*.cc", "js/node/src/*.h"]),
        local_defines = ["NAPI_VERSION=6", "ORT_API_MANUAL_INIT"] + cuda_defines,
        linkopts = ORIGIN_RPATH + select({
            "//bazel:macos": ["-undefined", "dynamic_lookup"],
            "//conditions:default": [],
        }),
        deps = [
            ":api_headers",
            ":runtime_shared",
            "@ort_node_headers//:headers",
            "@ort_node_addon_api//:headers",
        ] + select({
            "//bazel:windows": [":node_import_library"],
            "//conditions:default": [],
        }),
    )
    copy_file(
        name = "node_binding",
        src = ":node_native_library",
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
            ":runtime_library",
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
            "$(rootpath :runtime_library)",
        ],
        data = ["go/testdata/add_f32.onnx", ":runtime_library"],
        target_frameworks = ["net10.0"],
        deps = [":csharp_binding"],
    )
