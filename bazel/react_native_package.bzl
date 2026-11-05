# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""AAR and Metro-source npm packaging for the source-built React Native bridge."""

load("@aspect_rules_js//js:defs.bzl", "js_binary", "js_run_binary")
load("@rules_java//java/common:java_info.bzl", "JavaInfo")

def react_native_javascript():
    js_binary(
        name = "react_native_package_consumer",
        entry_point = "tools/bazel/react_native_package_consumer.cjs",
        data = ["//bazel/node:compiler", "VERSION_NUMBER"],
        include_npm = True,
    )
    js_binary(
        name = "react_native_common_compiler",
        entry_point = "tools/bazel/react_native_common.cjs",
        data = ["//bazel/node:compiler"],
    )
    js_run_binary(
        name = "react_native_common",
        tool = ":react_native_common_compiler",
        srcs = native.glob(["js/common/lib/*.ts"]) + ["js/tsconfig.json"],
        out_dirs = ["react_native_common"],
        args = ["react_native_common"],
    )

def _package_impl(ctx):
    jar = ctx.attr.java[JavaInfo].java_outputs[0].class_jar
    aar = ctx.actions.declare_file("onnxruntime-react-native-android-arm64.aar")
    npm = ctx.actions.declare_file("onnxruntime-react-native-android-arm64.tgz")
    args = ctx.actions.args()
    args.add("--aar", aar)
    args.add("--npm", npm)
    args.add("--classes", jar)
    args.add("--version", ctx.file.version)
    args.add_all(ctx.files.native, before_each = "--native")
    args.add_all(ctx.files.frontend, before_each = "--frontend")
    args.add_all(ctx.files.common, before_each = "--common")
    args.add_all(ctx.files.notices, before_each = "--notice")
    ctx.actions.run(
        executable = ctx.executable.packager,
        arguments = [args],
        inputs = [jar, ctx.file.version] + ctx.files.native + ctx.files.frontend + ctx.files.common + ctx.files.notices,
        outputs = [aar, npm],
        mnemonic = "ReactNativePackage",
    )
    return [
        DefaultInfo(files = depset([aar, npm])),
        OutputGroupInfo(aar = depset([aar]), npm = depset([npm])),
    ]

react_native_package = rule(
    implementation = _package_impl,
    attrs = {
        "java": attr.label(providers = [JavaInfo], mandatory = True),
        "native": attr.label_list(allow_files = True),
        "frontend": attr.label_list(allow_files = True),
        "common": attr.label_list(allow_files = True, cfg = "exec"),
        "notices": attr.label_list(allow_files = True),
        "version": attr.label(allow_single_file = True),
        "packager": attr.label(executable = True, cfg = "exec", mandatory = True),
    },
)
