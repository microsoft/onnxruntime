# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Maven repository artifacts and real Javadoc from the native Java sources."""

load("@rules_java//java/common:java_common.bzl", "java_common")
load("@rules_java//java/common:java_info.bzl", "JavaInfo")

def _java_documentation_impl(ctx):
    jdk = ctx.attr._jdk[java_common.JavaRuntimeInfo]
    sources = ctx.attr.library[JavaInfo].source_jars
    output = ctx.actions.declare_file(ctx.label.name + ".jar")
    args = ctx.actions.args()
    args.add("javadoc")
    args.add("--java-home", jdk.java_home)
    args.add("--output", output)
    args.add_all(sources, before_each = "--source")
    ctx.actions.run(
        executable = ctx.executable.tool,
        arguments = [args],
        inputs = depset(sources, transitive = [jdk.files]),
        tools = [ctx.attr.tool[DefaultInfo].files_to_run],
        outputs = [output],
        mnemonic = "OnnxRuntimeJavadoc",
    )
    return [DefaultInfo(files = depset([output]))]

java_documentation = rule(
    implementation = _java_documentation_impl,
    attrs = {
        "library": attr.label(mandatory = True, providers = [JavaInfo]),
        "tool": attr.label(mandatory = True, executable = True, cfg = "exec"),
        "_jdk": attr.label(
            default = "@bazel_tools//tools/jdk:current_java_runtime",
            cfg = "exec",
        ),
    },
)

def _java_maven_repository_impl(ctx):
    library = ctx.attr.library[JavaInfo]
    output = ctx.actions.declare_file(ctx.label.name + ".zip")
    args = ctx.actions.args()
    args.add("maven")
    args.add("--output", output)
    args.add("--version-file", ctx.file.version)
    args.add("--api-jar", library.runtime_output_jars[0])
    if ctx.file.native_jar:
        args.add("--native-jar", ctx.file.native_jar)
    if ctx.file.aar:
        args.add("--aar", ctx.file.aar)
    args.add("--javadoc-jar", ctx.file.javadoc)
    args.add("--classifier", ctx.attr.classifier)
    args.add("--variant", ctx.attr.variant)
    args.add("--license", ctx.file.license)
    args.add("--notices", ctx.file.notices)
    args.add_all(library.source_jars, before_each = "--source")
    ctx.actions.run(
        executable = ctx.executable.tool,
        arguments = [args],
        inputs = library.runtime_output_jars + library.source_jars + [
            ctx.file.javadoc,
            ctx.file.version,
            ctx.file.license,
            ctx.file.notices,
        ] + ([ctx.file.native_jar] if ctx.file.native_jar else []) + ([ctx.file.aar] if ctx.file.aar else []),
        tools = [ctx.attr.tool[DefaultInfo].files_to_run],
        outputs = [output],
        mnemonic = "OnnxRuntimeMaven",
    )
    return [DefaultInfo(files = depset([output]))]

java_maven_repository = rule(
    implementation = _java_maven_repository_impl,
    attrs = {
        "library": attr.label(mandatory = True, providers = [JavaInfo]),
        "native_jar": attr.label(allow_single_file = [".jar"]),
        "aar": attr.label(allow_single_file = [".aar"]),
        "javadoc": attr.label(mandatory = True, allow_single_file = [".jar"]),
        "version": attr.label(mandatory = True, allow_single_file = True),
        "license": attr.label(mandatory = True, allow_single_file = True),
        "notices": attr.label(mandatory = True, allow_single_file = True),
        "classifier": attr.string(mandatory = True),
        "variant": attr.string(default = "cpu", values = ["cpu", "cuda"]),
        "tool": attr.label(mandatory = True, executable = True, cfg = "exec"),
    },
)
