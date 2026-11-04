# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native CPython wheel packaging without invoking setup.py or build.py."""

load("@rules_python//python:packaging.bzl", "py_wheel_rule")
load("@rules_python//python:py_test.bzl", "py_test")

def python_wheel_package():
    """Package the source-assembled CPU inference API and native extension."""
    compatible = select({
        "//bazel:windows_x86_64": [],
        "//bazel:linux_x86_64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
    py_wheel_rule(
        name = "python_wheel",
        distribution = "onnxruntime",
        version = "1.32.0",
        abi = "cp312",
        python_tag = "cp312",
        platform = select({
            "//bazel:windows": "win_amd64",
            "//conditions:default": "linux_x86_64",
        }),
        python_requires = ">=3.12,<3.13",
        requires = ["numpy==2.4.0"],
        license = "MIT",
        extra_distinfo_files = {
            "LICENSE": "LICENSE",
            "ThirdPartyNotices.txt": "ThirdPartyNotices.txt",
        },
        strip_path_prefixes = ["python_runtime/"],
        target_compatible_with = compatible,
        deps = [":python_runtime_files"],
    )
    py_test(
        name = "python_wheel_test",
        srcs = ["tools/bazel/python_wheel_test.py"],
        main = "tools/bazel/python_wheel_test.py",
        args = [
            "$(rootpath :python_wheel)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath go/testdata/add_f32.onnx)",
        ],
        data = [
            ":python_wheel",
            "VERSION_NUMBER",
            "go/testdata/add_f32.onnx",
        ],
        target_compatible_with = compatible,
        deps = ["@ort_pip//numpy"],
    )
