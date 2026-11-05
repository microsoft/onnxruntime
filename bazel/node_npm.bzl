# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Production-only source-native npm distribution and offline consumer test."""

load("@aspect_rules_js//js:defs.bzl", "js_binary", "js_run_binary", "js_test")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

def node_npm_package():
    """Assemble native JS, compiler-produced declarations, and bundled common API."""
    compatible = select({
        "//bazel:windows": [],
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:x86_64": [],
        "//bazel:aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
    js_binary(
        name = "node_npm_assembler",
        entry_point = "tools/bazel/node_npm_package.cjs",
        data = ["//bazel/node:compiler"],
    )
    js_run_binary(
        name = "node_npm_tree",
        tool = ":node_npm_assembler",
        srcs = native.glob(["js/common/lib/*.ts", "js/node/lib/*.ts"]) + [
            "js/node/package.json",
            "js/tsconfig.json",
            "VERSION_NUMBER",
            "LICENSE",
            "ThirdPartyNotices.txt",
            ":onnxruntime_binding.node",
            ":runtime_library",
            ":providers_shared_library",
        ],
        out_dirs = ["node_npm_tree"],
        args = [
            "node_npm_tree",
            "$(rootpath :onnxruntime_binding.node)",
            "$(rootpath :runtime_library)",
            "$(rootpath :providers_shared_library)",
        ] + select({
            "//bazel:windows": ["win32"],
            "//bazel:macos": ["darwin"],
            "//conditions:default": ["linux"],
        }) + select({
            "//bazel:aarch64": ["arm64"],
            "//conditions:default": ["x64"],
        }),
        target_compatible_with = compatible,
    )
    pkg_tar(
        name = "node_npm",
        srcs = [":node_npm_tree"],
        out = "onnxruntime-node-bazel.tgz",
        extension = "tar.gz",
        strip_prefix = "node_npm_tree",
        target_compatible_with = compatible,
    )
    js_test(
        name = "node_npm_test",
        entry_point = "tools/bazel/node_npm_test.cjs",
        include_npm = True,
        args = [
            "$(rootpath :node_npm)",
            "$(rootpath go/testdata/add_f32.onnx)",
            "$(rootpath VERSION_NUMBER)",
        ],
        data = [
            ":node_npm",
            "//bazel/node:compiler",
            "go/testdata/add_f32.onnx",
            "VERSION_NUMBER",
        ],
        target_compatible_with = compatible,
    )
