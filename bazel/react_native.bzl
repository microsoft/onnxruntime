# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Official React Native prefab SDKs for the source-built Android JSI bridge."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _sdk_impl(module_ctx):
    http_archive(
        name = "ort_react_android",
        urls = ["https://repo.maven.apache.org/maven2/com/facebook/react/react-android/0.73.11/react-android-0.73.11-release.aar"],
        sha256 = "e0b9d7ce648fd3d84344641266e1a8ca652587775b43de028a0f68ce8c42c80e",
        type = "zip",
        build_file_content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_import(name = "jsi_binary", shared_library = "jni/arm64-v8a/libjsi.so")
cc_import(name = "react_binary", shared_library = "jni/arm64-v8a/libreact_nativemodule_core.so")
cc_import(name = "turbo_binary", shared_library = "jni/arm64-v8a/libturbomodulejsijni.so")
cc_library(
    name = "sdk",
    hdrs = glob(["prefab/modules/jsi/include/**/*.h", "prefab/modules/react_nativemodule_core/include/**/*.h", "prefab/modules/turbomodulejsijni/include/**/*.h"]),
    includes = ["prefab/modules/jsi/include", "prefab/modules/react_nativemodule_core/include", "prefab/modules/turbomodulejsijni/include"],
    deps = [":jsi_binary", ":react_binary", ":turbo_binary"],
    target_compatible_with = ["@platforms//os:android", "@platforms//cpu:aarch64"],
)
exports_files(["classes.jar", "jni/arm64-v8a/libc++_shared.so", "jni/arm64-v8a/libjsi.so"])
""",
    )
    http_archive(
        name = "ort_fbjni",
        urls = ["https://repo.maven.apache.org/maven2/com/facebook/fbjni/fbjni/0.5.1/fbjni-0.5.1.aar"],
        sha256 = "e10fe7fc9ced989c41b05ce0468a4ebbc5d1d38616edfe8a016aa07fb8ba4ede",
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

def _ndk_shared_impl(ctx):
    root = ctx.os.environ.get("ANDROID_NDK_HOME")
    if not root:
        fail("Android JSI requires NDK r27d. Set ANDROID_NDK_HOME to the same NDK used by --config=android_arm64.")
    properties = ctx.read(ctx.path(root + "/source.properties"))
    if "27.3.13750724" not in properties:
        fail("Android JSI requires NDK r27d (27.3.13750724); update ANDROID_NDK_HOME to that SDK.")
    host = "windows-x86_64" if "windows" in ctx.os.name else ("darwin-x86_64" if "mac" in ctx.os.name else "linux-x86_64")
    library = ctx.path(root + "/toolchains/llvm/prebuilt/" + host + "/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so")
    if not library.exists:
        fail("NDK r27d ARM64 shared C++ runtime is missing at %s; reinstall the complete SDK." % library)
    ctx.symlink(library, "libc++_shared.so")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
cc_import(name = "runtime", shared_library = "libc++_shared.so", visibility = ["//visibility:public"])
exports_files(["libc++_shared.so"])
""")

ndk_shared_runtime = repository_rule(
    implementation = _ndk_shared_impl,
    environ = ["ANDROID_NDK_HOME"],
    local = True,
)
