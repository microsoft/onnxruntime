# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Standalone Java distribution with source-built native libraries as resources."""

load("@rules_java//java:defs.bzl", "java_import", "java_test")

def java_native_package():
    """Assemble a single-platform CPU JAR using the existing JNI resource loader."""
    compatible = select({
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//bazel:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:aarch64": [],
        "//bazel:x86_64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
    resource_root = "ai/onnxruntime/native/" + select({
        "//bazel:windows": "win-",
        "//bazel:macos": "osx-",
        "//conditions:default": "linux-",
    }) + select({
        "//bazel:aarch64": "aarch64/",
        "//conditions:default": "x64/",
    })
    resources = ""
    for label, filenames in [
        (":runtime_library", ["onnxruntime.dll", "libonnxruntime.dylib", "libonnxruntime.so"]),
        (":java_native_library", ["onnxruntime4j_jni.dll", "libonnxruntime4j_jni.dylib", "libonnxruntime4j_jni.so"]),
        (":providers_shared_library", ["onnxruntime_providers_shared.dll", "libonnxruntime_providers_shared.dylib", "libonnxruntime_providers_shared.so"]),
    ]:
        resources = resources + " \"$(location " + label + "):" + resource_root + select({
            "//bazel:windows": filenames[0],
            "//bazel:macos": filenames[1],
            "//conditions:default": filenames[2],
        }) + "\""
    native.genrule(
        name = "java_native_jar",
        srcs = [
            ":java_binding",
            ":java_native_library",
            ":providers_shared_library",
            ":runtime_library",
            "LICENSE",
            "ThirdPartyNotices.txt",
            "VERSION_NUMBER",
        ],
        outs = ["onnxruntime-native.jar"],
        cmd = "\"$(location @bazel_tools//tools/jdk:singlejar)\" --output \"$@\" --normalize --compression" +
              " --sources \"$(location :java_binding)\" --resources" + resources +
              " LICENSE:META-INF/LICENSE ThirdPartyNotices.txt:META-INF/ThirdPartyNotices.txt" +
              " VERSION_NUMBER:META-INF/onnxruntime/VERSION_NUMBER",
        target_compatible_with = compatible,
        tools = ["@bazel_tools//tools/jdk:singlejar"],
    )
    java_import(
        name = "java_native_distribution",
        jars = [":java_native_jar"],
        target_compatible_with = compatible,
    )
    java_test(
        name = "java_native_jar_test",
        srcs = ["tools/bazel/RuntimeSmokeTest.java"],
        args = ["$(rootpath go/testdata/add_f32.onnx)"],
        data = ["go/testdata/add_f32.onnx"],
        jvm_flags = ["-Djava.library.path="],
        main_class = "RuntimeSmokeTest",
        target_compatible_with = compatible,
        use_testrunner = False,
        deps = [":java_native_distribution"],
    )
