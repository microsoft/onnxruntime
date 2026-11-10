# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Redistributable SDK built from the native runtime, not a prebuilt download."""

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
load("@rules_python//python:py_test.bzl", "py_test")

def runtime_sdk_package():
    """Package the selected native runtime and standalone public API headers."""
    pkg_files(
        name = "runtime_sdk_headers",
        srcs = native.glob(
            ["include/onnxruntime/core/session/*.h"],
            exclude = ["include/onnxruntime/core/session/environment.h"],
        ),
        prefix = "include",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "runtime_sdk_library",
        srcs = [
            ":runtime_release_library",
            ":providers_shared_release_library",
        ],
        attributes = pkg_attributes(mode = "0755"),
        prefix = select({
            "//bazel:windows": "bin",
            "//conditions:default": "lib",
        }),
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "runtime_sdk_import_library",
        srcs = [":runtime_interface_library"],
        prefix = "lib",
        renames = {":runtime_interface_library": "onnxruntime.lib"},
        target_compatible_with = ["@platforms//os:windows"],
    )
    pkg_files(
        name = "runtime_sdk_cuda_libraries",
        srcs = [
            ":cuda_provider_release_library",
        ],
        attributes = pkg_attributes(mode = "0755"),
        prefix = "lib",
        strip_prefix = strip_prefix.files_only(),
        target_compatible_with = select({
            "//bazel:cuda_enabled": ["@platforms//os:linux"],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    pkg_files(
        name = "runtime_sdk_notices",
        srcs = [
            "LICENSE",
            "ThirdPartyNotices.txt",
            "VERSION_NUMBER",
        ],
    )
    pkg_files(
        name = "runtime_sdk_example",
        srcs = ["tools/bazel/sdk_inference_example.cc"],
        prefix = "examples",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "runtime_sdk_example_model",
        srcs = ["go/testdata/add_f32.onnx"],
        prefix = "examples",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_zip(
        name = "runtime_sdk_archive",
        srcs = [
            ":runtime_sdk_headers",
            ":runtime_sdk_library",
            ":runtime_sdk_notices",
            ":runtime_sdk_example",
            ":runtime_sdk_example_model",
        ] + select({
            "//bazel:windows": [":runtime_sdk_import_library"],
            "//conditions:default": [],
        }) + select({
            "//bazel:cuda_enabled": [":runtime_sdk_cuda_libraries"],
            "//conditions:default": [],
        }),
        out = "onnxruntime-sdk.zip",
        package_dir = "onnxruntime",
    )
    py_test(
        name = "runtime_sdk_archive_test",
        srcs = ["tools/bazel/sdk_archive_test.py"],
        main = "tools/bazel/sdk_archive_test.py",
        args = [
            "$(rootpath :runtime_sdk_archive)",
            "$(rootpath VERSION_NUMBER)",
        ] + select({
            "//bazel:cuda_enabled": ["cuda"],
            "//conditions:default": ["cpu"],
        }),
        data = [
            ":runtime_sdk_archive",
            "VERSION_NUMBER",
        ],
    )
