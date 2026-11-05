# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Standalone Java distribution with source-built native libraries as resources."""

load("@rules_java//java:defs.bzl", "java_import", "java_library", "java_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")
load(":java_release.bzl", "java_documentation", "java_maven_repository")

def java_native_package():
    """Assemble a single-platform JAR using the existing JNI resource loader."""
    compatible = select({
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//bazel:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:aarch64": [],
        "//bazel:x86_64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    variant = select({
        "//bazel:cuda_enabled": "cuda",
        "//conditions:default": "cpu",
    })
    resource_root = "ai/onnxruntime/native/" + select({
        "//bazel:windows": "win-",
        "//bazel:macos": "osx-",
        "//conditions:default": "linux-",
    }) + select({
        "//bazel:aarch64": "aarch64/",
        "//conditions:default": "x64/",
    })
    classifier = select({
        "//bazel:windows": "win-",
        "//bazel:macos": "osx-",
        "//conditions:default": "linux-",
    }) + select({
        "//bazel:aarch64": "aarch64",
        "//conditions:default": "x64",
    })
    resources = ""
    for label, filenames in [
        (":runtime_release_library", ["onnxruntime.dll", "libonnxruntime.dylib", "libonnxruntime.so"]),
        (":java_native_release_library", ["onnxruntime4j_jni.dll", "libonnxruntime4j_jni.dylib", "libonnxruntime4j_jni.so"]),
        (":providers_shared_release_library", ["onnxruntime_providers_shared.dll", "libonnxruntime_providers_shared.dylib", "libonnxruntime_providers_shared.so"]),
    ]:
        resources = resources + " \"$(location " + label + "):" + resource_root + select({
            "//bazel:windows": filenames[0],
            "//bazel:macos": filenames[1],
            "//conditions:default": filenames[2],
        }) + "\""
    resources += select({
        "//bazel:cuda_enabled": ' "$(location :cuda_provider_release_library):ai/onnxruntime/native/linux-x64/libonnxruntime_providers_cuda.so"',
        "//conditions:default": "",
    })
    native.genrule(
        name = "java_native_jar",
        srcs = [
            ":java_binding",
            ":java_native_release_library",
            ":providers_shared_release_library",
            ":runtime_release_library",
            "LICENSE",
            "ThirdPartyNotices.txt",
            "VERSION_NUMBER",
        ] + select({
            "//bazel:cuda_enabled": [":cuda_provider_release_library"],
            "//conditions:default": [],
        }),
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
    py_binary(
        name = "java_release_tool",
        srcs = ["tools/bazel/java_release.py"],
        main = "tools/bazel/java_release.py",
    )
    java_documentation(
        name = "java_javadoc_jar",
        library = ":java_binding",
        tool = ":java_release_tool",
    )
    java_maven_repository(
        name = "java_maven",
        library = ":java_binding",
        native_jar = ":java_native_jar",
        javadoc = ":java_javadoc_jar",
        version = "VERSION_NUMBER",
        license = "LICENSE",
        notices = "ThirdPartyNotices.txt",
        classifier = classifier,
        variant = variant,
        tool = ":java_release_tool",
        target_compatible_with = compatible,
    )
    py_test(
        name = "java_release_tool_test",
        imports = ["tools/bazel"],
        srcs = [
            "tools/bazel/java_release.py",
            "tools/bazel/java_release_test.py",
        ],
        main = "tools/bazel/java_release_test.py",
    )
    java_library(
        name = "maven_resolution_mojo_classes",
        srcs = [
            "tools/bazel/MavenResolutionMojo.java",
            "tools/bazel/RuntimeSmokeTest.java",
        ],
        javacopts = ["--release", "8"],
        deps = [
            ":java_binding",
            "@ort_maven//:maven_libraries",
        ],
    )
    native.genrule(
        name = "maven_resolution_mojo",
        srcs = [
            ":maven_resolution_mojo_classes",
            "tools/bazel/maven-plugin.xml",
        ],
        outs = ["maven-resolution-probe-1.0.0.jar"],
        cmd = "\"$(location @bazel_tools//tools/jdk:singlejar)\" --output \"$@\" --normalize --compression" +
              " --sources \"$(location :maven_resolution_mojo_classes)\"" +
              " --resources \"$(location tools/bazel/maven-plugin.xml):META-INF/maven/plugin.xml\"",
        tools = ["@bazel_tools//tools/jdk:singlejar"],
    )
    java_test(
        name = "java_maven_copy_test",
        srcs = [
            "tools/bazel/MavenFileCopyTest.java",
            "tools/bazel/MavenPackageSmokeTest.java",
        ],
        main_class = "MavenFileCopyTest",
        use_testrunner = False,
    )
    java_test(
        name = "java_maven_test",
        srcs = [
            "tools/bazel/MavenPackageSmokeTest.java",
            "tools/bazel/RuntimeSmokeTest.java",
        ],
        args = [
            "$(rootpath :java_maven)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath @ort_maven//:home_marker)",
            "$(rootpath :maven_resolution_mojo)",
            "$(rootpath tools/bazel/maven-resolution-probe.pom)",
        ] + select({
            "//bazel:windows": ["win-"],
            "//bazel:macos": ["osx-"],
            "//conditions:default": ["linux-"],
        }) + select({
            "//bazel:aarch64": ["aarch64"],
            "//conditions:default": ["x64"],
        }) + select({
            "//bazel:cuda_enabled": ["cuda"],
            "//conditions:default": ["cpu"],
        }) + ["$(rootpath go/testdata/add_f32.onnx)"],
        data = [
            ":java_maven",
            ":maven_resolution_mojo",
            "VERSION_NUMBER",
            "go/testdata/add_f32.onnx",
            "tools/bazel/maven-resolution-probe.pom",
            "@ort_maven//:distribution",
            "@ort_maven//:home_marker",
        ],
        jvm_flags = ["-Djava.library.path="],
        main_class = "MavenPackageSmokeTest",
        target_compatible_with = compatible,
        use_testrunner = False,
        deps = [":java_binding"],
    )
