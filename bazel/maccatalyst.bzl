# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Compile Catalyst dependencies with the MacOSX SDK, never reuse macOS objects."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _catalyst_impl(settings, _attr):
    cpu = settings["//command_line_option:cpu"]
    architectures = {
        "darwin_arm64": "arm64",
        "darwin_x86_64": "x86_64",
    }
    if cpu not in architectures:
        fail("Catalyst requires a macOS arm64/x86_64 architecture split; received {}. Use the dedicated XCFramework target.".format(cpu))
    catalyst_cpu = "catalyst_" + architectures[cpu]
    return {
        "//command_line_option:apple_platform_type": "catalyst",
        "//command_line_option:cpu": catalyst_cpu,
        "//command_line_option:ios_minimum_os": "15.1",
        "//command_line_option:platforms": [str(Label("@apple_support//platforms:" + catalyst_cpu))],
    }

_catalyst = transition(
    implementation = _catalyst_impl,
    inputs = ["//command_line_option:cpu"],
    outputs = [
        "//command_line_option:apple_platform_type",
        "//command_line_option:cpu",
        "//command_line_option:ios_minimum_os",
        "//command_line_option:platforms",
    ],
)

def _library_impl(ctx):
    library = ctx.attr.library[0]
    providers = [library[CcInfo]]
    if apple_common.Objc in library:
        providers.append(library[apple_common.Objc])
    return providers

catalyst_library = rule(
    implementation = _library_impl,
    attrs = {
        "library": attr.label(cfg = _catalyst, providers = [CcInfo]),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
)

def _merge_impl(ctx):
    args = ctx.actions.args()
    args.add("--base", ctx.file.base)
    args.add("--catalyst", ctx.file.catalyst)
    args.add("--output", ctx.outputs.archive)
    ctx.actions.run(
        executable = ctx.executable._tool,
        arguments = [args],
        inputs = [ctx.file.base, ctx.file.catalyst],
        outputs = [ctx.outputs.archive],
        mnemonic = "OnnxRuntimeCatalystXCFramework",
    )
    return [DefaultInfo(files = depset([ctx.outputs.archive]))]

catalyst_xcframework = rule(
    implementation = _merge_impl,
    attrs = {
        "base": attr.label(allow_single_file = [".zip"]),
        "catalyst": attr.label(allow_single_file = [".zip"]),
        "_tool": attr.label(
            default = "//objectivec:catalyst_xcframework_merger",
            cfg = "exec",
            executable = True,
        ),
    },
    outputs = {"archive": "%{name}.xcframework.zip"},
)
