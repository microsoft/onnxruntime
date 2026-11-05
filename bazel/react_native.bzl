# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Official React Native prefab SDKs for the source-built Android JSI bridge."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file", "http_jar")

def _sdk_impl(module_ctx):
    for name, url, sha, filename in [
        ("ort_react_license", "https://raw.githubusercontent.com/facebook/react-native/v0.73.11/LICENSE", "da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93", "ReactNative-LICENSE"),
        ("ort_fbjni_license", "https://raw.githubusercontent.com/facebookincubator/fbjni/v0.5.1/LICENSE", "f62d190c08e2b300b45780f0b3cabcacc78d0e373f7a0de79d31f82b084d35ea", "fbjni-LICENSE"),
    ]:
        http_file(name = name, urls = [url], sha256 = sha, downloaded_file_path = filename)
    http_archive(
        name = "ort_react_android",
        urls = ["https://repo.maven.apache.org/maven2/com/facebook/react/react-android/0.73.11/react-android-0.73.11-release.aar"],
        sha256 = "e0b9d7ce648fd3d84344641266e1a8ca652587775b43de028a0f68ce8c42c80e",
        type = "zip",
        build_file_content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_java//java:defs.bzl", "java_import")
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
java_import(name = "java", jars = ["classes.jar"])
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
load("@rules_java//java:defs.bzl", "java_import")
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
java_import(name = "java", jars = ["classes.jar"])
""",
    )
    http_archive(
        name = "ort_android_platform",
        urls = ["https://dl.google.com/android/repository/platform-35_r02.zip"],
        sha256 = "0988cacad01b38a18a47bac14a0695f246bc76c1b06c0eeb8eb0dc825ab0c8e0",
        strip_prefix = "android-35",
        build_file_content = """
load("@rules_java//java:defs.bzl", "java_import")
java_import(name = "java", jars = ["android.jar"], visibility = ["//visibility:public"])
""",
    )
    for name, url, sha in [
        ("ort_androidx_annotation", "https://dl.google.com/dl/android/maven2/androidx/annotation/annotation-jvm/1.7.1/annotation-jvm-1.7.1.jar", "e36b8e4b8393a4adc74e3d4ab22ad5a36396f0cea2e40b5734eae14937dfd224"),
        ("ort_react_kotlin", "https://repo.maven.apache.org/maven2/org/jetbrains/kotlin/kotlin-stdlib/1.8.0/kotlin-stdlib-1.8.0.jar", "c77bef8774640b9fb9d6e217459ff220dae59878beb7d2e4b430506feffc654e"),
    ]:
        http_jar(
            name = name,
            urls = [url],
            sha256 = sha,
        )
    http_archive(
        name = "ort_react_headers",
        urls = ["https://github.com/facebook/react-native/archive/refs/tags/v0.73.11.tar.gz"],
        sha256 = "fd3271ff56d13469bfcb8d84429728e2eaa5f48c138188dc30b1c1ead790428e",
        strip_prefix = "react-native-0.73.11/packages/react-native",
        build_file_content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "base",
    hdrs = glob(["React/Base/*.h"]),
    strip_include_prefix = "React/Base",
    include_prefix = "React",
    target_compatible_with = ["@platforms//os:ios"],
)
cc_library(
    name = "sdk",
    hdrs = glob(["ReactCommon/**/*.h"]),
    includes = [
        "ReactCommon",
        "ReactCommon/jsi",
        "ReactCommon/callinvoker",
        "ReactCommon/react/nativemodule/core",
        "ReactCommon/react/nativemodule/core/platform/ios",
    ],
    deps = [":base"],
    target_compatible_with = ["@platforms//os:ios"],
)
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
    ctx.symlink(ctx.path(root + "/NOTICE.toolchain"), "NDK-NOTICE")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
cc_import(name = "runtime", shared_library = "libc++_shared.so", visibility = ["//visibility:public"])
exports_files(["libc++_shared.so", "NDK-NOTICE"])
""")

ndk_shared_runtime = repository_rule(
    implementation = _ndk_shared_impl,
    environ = ["ANDROID_NDK_HOME"],
    local = True,
)
