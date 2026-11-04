# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Keep Windows cgo's GNU ABI compiler separate from the MSVC runtime graph."""

load("@rules_go//go:def.bzl", "GoArchive", "GoInfo")

def _cgo_platform_impl(settings, attr):
    if not attr.windows:
        return {
            "//command_line_option:platforms": settings["//command_line_option:platforms"],
            "//command_line_option:cxxopt": settings["//command_line_option:cxxopt"],
        }
    return {
        "//command_line_option:platforms": [str(Label("//bazel:windows_cgo"))],
        "//command_line_option:cxxopt": [],
    }

_cgo_platform = transition(
    implementation = _cgo_platform_impl,
    inputs = ["//command_line_option:platforms", "//command_line_option:cxxopt"],
    outputs = ["//command_line_option:platforms", "//command_line_option:cxxopt"],
)

def _library_impl(ctx):
    library = ctx.attr.library[0]
    return [library[DefaultInfo], library[GoInfo], library[GoArchive]]

desktop_go_library = rule(
    implementation = _library_impl,
    attrs = {
        "library": attr.label(cfg = _cgo_platform),
        "windows": attr.bool(),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
)

def _test_impl(ctx):
    test = ctx.attr.test[0]
    executable = ctx.actions.declare_file(ctx.label.name + (".exe" if ctx.attr.windows else ""))
    ctx.actions.symlink(
        output = executable,
        target_file = test[DefaultInfo].files_to_run.executable,
        is_executable = True,
    )
    runfiles = ctx.runfiles(files = [ctx.file.runtime])
    return [
        DefaultInfo(
            executable = executable,
            runfiles = runfiles.merge(test[DefaultInfo].default_runfiles),
        ),
        RunEnvironmentInfo(environment = {"ORT_LIB_PATH": "../../" + ctx.file.runtime.short_path}),
    ]

desktop_go_test = rule(
    implementation = _test_impl,
    test = True,
    attrs = {
        "test": attr.label(cfg = _cgo_platform),
        "runtime": attr.label(allow_single_file = True),
        "windows": attr.bool(),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
)
