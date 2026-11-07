# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Expose the JNI headers emitted by Bazel's Java compiler to native compilation."""

load("@rules_java//java/common:java_common.bzl", "java_common")
load("@rules_java//java/common:java_info.bzl", "JavaInfo")

def _jni_headers_impl(ctx):
    headers = ctx.actions.declare_directory(ctx.label.name)
    header_jar = ctx.attr.java[JavaInfo].java_outputs[0].native_headers_jar
    jdk = ctx.attr._jdk[java_common.JavaRuntimeInfo]
    ctx.actions.run_shell(
        inputs = [header_jar],
        tools = jdk.files,
        outputs = [headers],
        command = """
set -eu
jar=$(cd "{jdk}" && pwd)/bin/jar
archive="$PWD/{input}"
mkdir -p "{out}"
cd "{out}"
"$jar" xf "$archive"
""".format(
            out = headers.path,
            jdk = jdk.java_home,
            input = header_jar.path,
        ),
        mnemonic = "ExtractJniHeaders",
    )
    return [
        DefaultInfo(files = depset([headers])),
        CcInfo(compilation_context = cc_common.create_compilation_context(
            headers = depset([headers]),
            includes = depset([headers.path]),
        )),
    ]

jni_headers = rule(
    implementation = _jni_headers_impl,
    attrs = {
        "java": attr.label(mandatory = True, providers = [JavaInfo]),
        "_jdk": attr.label(
            default = "@bazel_tools//tools/jdk:current_java_runtime",
            cfg = "exec",
        ),
    },
)
