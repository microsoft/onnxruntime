# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Android source-built runtime, JNI binding, and single-ABI AAR packaging."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("@rules_java//java/common:java_info.bzl", "JavaInfo")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("//bazel:java_release.bzl", "java_documentation", "java_maven_repository")
load("//bazel:jni_headers.bzl", "jni_headers")
load("//bazel:release_native_binary.bzl", "release_native_binary")

_ANDROID = ["@platforms//os:android", "@platforms//cpu:aarch64"]

def _aar_impl(ctx):
    jar = ctx.attr.java[JavaInfo].java_outputs[0].class_jar
    out = ctx.actions.declare_file("onnxruntime-android-arm64.aar")
    args = ctx.actions.args()
    args.add(out)
    args.add(jar)
    args.add(ctx.file.runtime)
    args.add(ctx.file.jni)
    args.add_all(ctx.files.assets)
    ctx.actions.run(
        executable = ctx.executable._packager,
        arguments = [args],
        inputs = [jar, ctx.file.runtime, ctx.file.jni] + ctx.files.assets,
        outputs = [out],
        mnemonic = "AndroidAar",
    )
    return [DefaultInfo(files = depset([out]))]

_aar = rule(
    implementation = _aar_impl,
    attrs = {
        "java": attr.label(providers = [JavaInfo], mandatory = True),
        "runtime": attr.label(allow_single_file = True, mandatory = True),
        "jni": attr.label(allow_single_file = True, mandatory = True),
        "assets": attr.label_list(allow_files = True),
        "_packager": attr.label(
            default = "//:android_aar_packager",
            executable = True,
            cfg = "exec",
        ),
    },
)

def android_bindings():
    """Declare Android targets; cross-compile with --config=android_arm64."""
    java_library(
        name = "android_java",
        javacopts = ["--release", "8"],
        srcs = native.glob([
            "java/src/main/java/**/*.java",
            "java/src/main/android/**/*.java",
        ]),
    )
    jni_headers(
        name = "android_jni_headers",
        java = ":android_java",
    )
    java_library(
        name = "android_java_util_classes",
        srcs = ["java/src/test/java/ai/onnxruntime/UtilTest.java"],
        deps = [":android_java", ":java_junit"],
    )
    java_test(
        name = "android_java_util_test",
        srcs = ["tools/bazel/JavaTestRunner.java"],
        main_class = "JavaTestRunner",
        use_testrunner = False,
        args = ["-", "-", "java/src/test/resources", "ai.onnxruntime.UtilTest"],
        data = native.glob(["java/src/test/resources/**"]),
        runtime_deps = [":android_java_util_classes"],
        deps = [":java_junit"],
    )
    cc_binary(
        name = "android/libonnxruntime.so",
        additional_linker_inputs = ["//bazel:onnxruntime.lds"],
        linkopts = [
            "-Wl,--version-script=$(location //bazel:onnxruntime.lds)",
            "-Wl,-soname,libonnxruntime.so",
            "-Wl,-z,max-page-size=16384",
            "-Wl,--no-undefined",
        ],
        linkshared = True,
        linkstatic = True,
        target_compatible_with = _ANDROID,
        deps = [":runtime"],
    )
    cc_import(
        name = "android_runtime_shared",
        shared_library = ":android/libonnxruntime.so",
        target_compatible_with = _ANDROID,
    )
    cc_binary(
        name = "android/libonnxruntime4j_jni.so",
        srcs = native.glob(
            ["java/src/main/native/*.c"],
            exclude = ["java/src/main/native/*Training*.c"],
        ) + ["java/src/main/native/OrtJniUtil.h"],
        includes = ["java/src/main/native"],
        linkopts = [
            "-Wl,-soname,libonnxruntime4j_jni.so",
            "-Wl,-z,max-page-size=16384",
            "-Wl,--no-undefined",
        ],
        linkshared = True,
        linkstatic = True,
        target_compatible_with = _ANDROID,
        # jni.h comes from the NDK sysroot, not a host JDK's jni_md.h.
        deps = [":api_headers", ":generated_headers", ":android_jni_headers", ":android_runtime_shared"],
    )
    release_native_binary(
        name = "android_runtime_release_library",
        src = ":android/libonnxruntime.so",
        target_compatible_with = _ANDROID,
    )
    release_native_binary(
        name = "android_jni_release_library",
        src = ":android/libonnxruntime4j_jni.so",
        target_compatible_with = _ANDROID,
    )
    py_binary(
        name = "android_aar_packager",
        srcs = ["tools/bazel/android_aar.py"],
        main = "tools/bazel/android_aar.py",
        python_version = "PY3",
    )
    py_test(
        name = "android_aar_packager_test",
        srcs = ["tools/bazel/android_aar_test.py", "tools/bazel/android_aar.py"],
        main = "tools/bazel/android_aar_test.py",
        imports = ["tools/bazel"],
    )
    _aar(
        name = "android_aar",
        java = ":android_java",
        runtime = ":android_runtime_release_library",
        jni = ":android_jni_release_library",
        assets = [
            "LICENSE",
            "ThirdPartyNotices.txt",
            "VERSION_NUMBER",
        ] + native.glob([
            "include/onnxruntime/core/session/*.h",
        ], exclude = ["include/onnxruntime/core/session/environment.h"]),
        target_compatible_with = _ANDROID,
    )
    java_documentation(
        name = "android_javadoc_jar",
        library = ":android_java",
        tool = ":java_release_tool",
        tags = ["manual"],
    )
    java_maven_repository(
        name = "android_maven",
        library = ":android_java",
        aar = ":android_aar",
        javadoc = ":android_javadoc_jar",
        version = "VERSION_NUMBER",
        license = "LICENSE",
        notices = "ThirdPartyNotices.txt",
        classifier = "android-arm64",
        tool = ":java_release_tool",
        tags = ["manual"],
        target_compatible_with = _ANDROID,
    )
