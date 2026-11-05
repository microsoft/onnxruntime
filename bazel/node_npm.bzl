# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Production-only source-native npm distributions and offline consumer tests."""

load("@aspect_rules_js//js:defs.bzl", "js_binary", "js_run_binary", "js_test")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load(":js_env.bzl", "WINDOWS_JS_ENV")

def node_npm_package():
    """Assemble common and native Node packages with compiler-produced declarations."""
    compatible = select({
        "//bazel:windows": [],
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:x86_64": [],
        "//bazel:aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    js_binary(
        name = "node_npm_assembler",
        env = WINDOWS_JS_ENV,
        entry_point = "tools/bazel/node_npm_package.cjs",
        data = native.glob(["js/common/lib/*.ts", "js/node/lib/*.ts"]) + [
            "//bazel/node:compiler",
            "js/common/package.json",
            "js/common/README.md",
            "js/node/package.json",
            "js/tsconfig.json",
            "VERSION_NUMBER",
            "LICENSE",
            "ThirdPartyNotices.txt",
        ],
    )
    js_run_binary(
        name = "common_npm_tree",
        tool = ":node_npm_assembler",
        srcs = native.glob(["js/common/lib/*.ts"]) + [
            "js/common/package.json",
            "js/common/README.md",
            "js/tsconfig.json",
            "VERSION_NUMBER",
            "LICENSE",
            "ThirdPartyNotices.txt",
        ],
        out_dirs = ["common_npm_tree"],
        args = [
            "common",
            "common_npm_tree",
        ],
    )
    pkg_tar(
        name = "common_npm",
        srcs = [":common_npm_tree"],
        out = "onnxruntime-common-bazel.tgz",
        extension = "tar.gz",
        strip_prefix = "common_npm_tree",
    )
    js_test(
        name = "common_npm_test",
        env = WINDOWS_JS_ENV,
        entry_point = "tools/bazel/common_npm_test.cjs",
        include_npm = True,
        args = [
            "$(rootpath :common_npm)",
            "$(rootpath VERSION_NUMBER)",
        ],
        data = [
            ":common_npm",
            "//bazel/node:compiler",
            "VERSION_NUMBER",
        ],
    )
    js_run_binary(
        name = "node_npm_tree",
        tool = ":node_npm_assembler",
        srcs = native.glob(["js/common/lib/*.ts", "js/node/lib/*.ts"]) + [
            "js/common/package.json",
            "js/common/README.md",
            "js/node/package.json",
            "js/tsconfig.json",
            "VERSION_NUMBER",
            "LICENSE",
            "ThirdPartyNotices.txt",
            ":node_native_release_library",
            ":runtime_release_library",
            ":providers_shared_release_library",
        ] + select({
            "//bazel:cuda_enabled": [":cuda_provider_release_library"],
            "//conditions:default": [],
        }),
        out_dirs = ["node_npm_tree"],
        args = [
            "node",
            "node_npm_tree",
            "$(rootpath :node_native_release_library)",
            "$(rootpath :runtime_release_library)",
            "$(rootpath :providers_shared_release_library)",
        ] + select({
            "//bazel:windows": ["win32"],
            "//bazel:macos": ["darwin"],
            "//conditions:default": ["linux"],
        }) + select({
            "//bazel:aarch64": ["arm64"],
            "//conditions:default": ["x64"],
        }) + select({
            "//bazel:cuda_enabled": [
                "cuda",
                "$(rootpath :cuda_provider_release_library)",
            ],
            "//conditions:default": ["cpu"],
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
        env = WINDOWS_JS_ENV,
        entry_point = "tools/bazel/node_npm_test.cjs",
        include_npm = True,
        args = [
            "$(rootpath :node_npm)",
            "$(rootpath :common_npm)",
            "$(rootpath go/testdata/add_f32.onnx)",
            "$(rootpath VERSION_NUMBER)",
        ],
        data = [
            ":node_npm",
            ":common_npm",
            "//bazel/node:compiler",
            "go/testdata/add_f32.onnx",
            "VERSION_NUMBER",
        ],
        target_compatible_with = compatible,
    )
