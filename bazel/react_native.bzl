# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Official React Native prefab SDKs for the source-built Android JSI bridge."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _sdk_impl(module_ctx):
    http_archive(
        name = "ort_react_android",
        urls = ["https://repo.maven.apache.org/maven2/com/facebook/react/react-android/0.76.9/react-android-0.76.9-release.aar"],
        sha256 = "b6853fe90a3ede4c59196dd05f2e5a66b6677e273f433d647f8175423513d62f",
        type = "zip",
        build_file_content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_import(name = "jsi_binary", shared_library = "jni/arm64-v8a/libjsi.so")
cc_import(name = "react_binary", shared_library = "jni/arm64-v8a/libreactnative.so")
cc_library(
    name = "sdk",
    hdrs = glob(["prefab/modules/jsi/include/**/*.h", "prefab/modules/reactnative/include/**/*.h"]),
    includes = ["prefab/modules/jsi/include", "prefab/modules/reactnative/include"],
    deps = [":jsi_binary", ":react_binary"],
    target_compatible_with = ["@platforms//os:android", "@platforms//cpu:aarch64"],
)
exports_files(["classes.jar", "jni/arm64-v8a/libc++_shared.so", "jni/arm64-v8a/libjsi.so", "jni/arm64-v8a/libreactnative.so"])
""",
    )
    http_archive(
        name = "ort_fbjni",
        urls = ["https://repo.maven.apache.org/maven2/com/facebook/fbjni/fbjni/0.6.0/fbjni-0.6.0.aar"],
        sha256 = "e87165d3c0f81371fa03fa7cae50ee1576c550329516bb1f4978f419b4b07d4c",
        type = "zip",
        build_file_content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_import(name = "binary", shared_library = "jni/arm64-v8a/libfbjni.so")
cc_library(
    name = "sdk",
    hdrs = glob(["prefab/modules/fbjni/include/**/*.h"]),
    includes = ["prefab/modules/fbjni/include"],
    deps = [":binary"],
    target_compatible_with = ["@platforms//os:android", "@platforms//cpu:aarch64"],
)
exports_files(["classes.jar", "jni/arm64-v8a/libfbjni.so"])
""",
    )
    return module_ctx.extension_metadata(reproducible = True)

react_native_sdk = module_extension(implementation = _sdk_impl)
