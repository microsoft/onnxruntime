# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Existing internal loader tests with isolated executable/library directories."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//bazel:platforms.bzl", "desktop_shared_library")

def loader_tests(plugin_libraries):
    runtime_path_directory = "onnxruntime/test/shared_lib/runtime_path_test_shared_library"
    desktop_shared_library(
        name = "runtime_path_test_library",
        linux = "libonnxruntime_runtime_path_test_shared_library.so",
        windows = "onnxruntime_runtime_path_test_shared_library.dll",
        macos = "libonnxruntime_runtime_path_test_shared_library.dylib",
        testonly = True,
        srcs = [runtime_path_directory + "/runtime_path_test_shared_library.cc"],
        win_def_file = select({
            "//bazel:windows": runtime_path_directory + "/runtime_path_test_shared_library.def",
            "//conditions:default": None,
        }),
        additional_linker_inputs = [runtime_path_directory + "/runtime_path_test_shared_library.lds"],
        linkopts = select({
            "//bazel:windows": [],
            "//bazel:macos": ["-Wl,-exported_symbol,_OrtTestGetSharedLibraryRuntimePath"],
            "//conditions:default": [
                "-Wl,--version-script=$(location " + runtime_path_directory + "/runtime_path_test_shared_library.lds)",
            ],
        }),
        # Env::GetRuntimePath must execute inside this module, not the runtime DLL.
        deps = [":runtime"],
    )
    native.filegroup(
        name = "runtime_path_test_interface",
        testonly = True,
        srcs = [":runtime_path_test_library"],
        output_group = "interface_library",
    )
    cc_import(
        name = "runtime_path_test_import",
        testonly = True,
        shared_library = ":runtime_path_test_library",
        interface_library = select({
            "//bazel:windows": ":runtime_path_test_interface",
            "//conditions:default": None,
        }),
    )
    cc_test(
        name = "runtime_path_test",
        srcs = [
            "onnxruntime/test/shared_lib/test_runtime_path.cc",
            "tools/bazel/shared_api_test_main.cc",
            "tools/bazel/test_main.cc",
        ],
        data = native.glob(["onnxruntime/test/testdata/**"]),
        deps = [
            ":runtime_path_test_import",
            ":shared_api_test_support",
        ],
        linkstatic = True,
        size = "large",
    )
    for kind in ["bridge", "probe"]:
        fixtures = []
        defines = []
        for variant in (["valid", "missing_export"] if kind == "bridge" else ["shared", "valid", "missing_export"]):
            name = "onnxruntime_" + kind + "_" + variant + "_fixture"
            desktop_shared_library(
                name = name,
                linux = "lib" + name + ".so",
                windows = name + ".dll",
                macos = "lib" + name + ".dylib",
                testonly = True,
                srcs = ["onnxruntime/test/shared_lib/" + (
                    "provider_bridge_test_library.cc" if kind == "bridge" else "optional_provider_probe_test_library.cc"
                )],
                local_defines = (
                    ["ORT_TEST_PROVIDER_SET_HOST"] if variant == "valid" else []
                ) if kind == "bridge" else ["ORT_TEST_OPTIONAL_PROVIDER_" + variant],
                deps = [":shared_api_test_common"],
            )
            fixtures.append(":" + name)
            define = (
                "ORT_PROVIDER_BRIDGE_" + variant.upper() + "_TEST_LIBRARY" if kind == "bridge" else "ORT_OPTIONAL_PROBE_" + variant + "_LIBRARY"
            )
            defines += select({
                "//bazel:windows": [define + '=\\"' + name + '.dll\\"'],
                "//bazel:macos": [define + '=\\"lib' + name + '.dylib\\"'],
                "//conditions:default": [define + '=\\"lib' + name + '.so\\"'],
            })
        name = "provider_bridge_test" if kind == "bridge" else "optional_provider_probe_test"
        cc_binary(
            name = name + "_binary",
            testonly = True,
            srcs = [
                "onnxruntime/test/shared_lib/" + (
                    "provider_bridge_test.cc" if kind == "bridge" else "optional_provider_probe_test.cc"
                ),
                "tools/bazel/test_main.cc",
            ],
            local_defines = defines,
            deps = [":ort_test_main"],
            linkstatic = True,
        )
        py_test(
            name = name,
            srcs = ["tools/bazel/isolated_binary_test.py"],
            main = "tools/bazel/isolated_binary_test.py",
            args = ["$(rootpath :" + name + "_binary)"] + [
                ("--library=" if kind == "bridge" else "--fixture=") + "$(rootpath " + fixture + ")"
                for fixture in fixtures
            ],
            data = [":" + name + "_binary"] + fixtures + native.glob(["onnxruntime/test/testdata/**"]),
            size = "large",
        )
    cc_binary(
        name = "runtime_dlopen_test_binary",
        testonly = True,
        srcs = ["onnxruntime/test/shared_lib/dlopen_main.cc"],
        deps = [":api_headers"],
        target_compatible_with = ["@platforms//os:windows"],
    )
    py_test(
        name = "runtime_dlopen_test",
        srcs = ["tools/bazel/isolated_binary_test.py"],
        main = "tools/bazel/isolated_binary_test.py",
        args = [
            "$(rootpath :runtime_dlopen_test_binary)",
            "--library=$(rootpath :runtime_library)",
            "--testdata=$(rootpath onnxruntime/test/testdata/matmul_1.onnx)",
        ],
        data = [
            ":runtime_dlopen_test_binary",
            ":runtime_library",
            "onnxruntime/test/testdata/matmul_1.onnx",
        ],
        target_compatible_with = ["@platforms//os:windows"],
    )
    cc_binary(
        name = "autoep_test",
        testonly = True,
        srcs = native.glob(["onnxruntime/test/autoep/*.cc"]) + [
            "onnxruntime/test/shared_lib/utils.cc",
            "tools/bazel/test_main.cc",
        ],
        data = plugin_libraries + native.glob(["onnxruntime/test/testdata/**"]),
        deps = [":ort_test_main"],
        copts = select({
            "//bazel:windows": ["/utf-8"],
            "//conditions:default": [],
        }),
        linkstatic = True,
    )
    py_test(
        name = "autoep_tests",
        srcs = ["tools/bazel/isolated_binary_test.py"],
        main = "tools/bazel/isolated_binary_test.py",
        args = ["$(rootpath :autoep_test)"] + [
            "--library=$(rootpath %s)" % library
            for library in plugin_libraries
        ],
        data = [":autoep_test"] + plugin_libraries + native.glob(["onnxruntime/test/testdata/**"]),
        env = select({
            "//bazel:windows": {},
            "//bazel:macos": {"DYLD_LIBRARY_PATH": "."},
            "//conditions:default": {"LD_LIBRARY_PATH": "."},
        }),
        tags = ["manual"],
        size = "large",
        timeout = "long",
    )
