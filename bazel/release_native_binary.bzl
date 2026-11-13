# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create package-ready native binaries without changing link-time artifacts."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")

_CC_TOOLCHAIN_TYPE = "@bazel_tools//tools/cpp:toolchain_type"

def _matches_constraint(ctx, constraint):
    return ctx.target_platform_has_constraint(constraint[platform_common.ConstraintValueInfo])

def _matches_any_constraint(ctx, constraints):
    for constraint in constraints:
        if _matches_constraint(ctx, constraint):
            return True
    return False

def _release_native_binary_impl(ctx):
    source = ctx.file.src
    output = ctx.actions.declare_file(ctx.label.name + "/" + source.basename)
    if _matches_constraint(ctx, ctx.attr._windows):
        ctx.actions.symlink(
            output = output,
            target_file = source,
            progress_message = "Forwarding unstripped Windows binary for {}".format(ctx.label),
        )
        return [DefaultInfo(files = depset([output]))]

    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    if not cc_common.action_is_enabled(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.strip,
    ):
        fail(
            "The selected C++ toolchain does not provide the strip action required by {}. ".format(ctx.label) +
            "Select or register a C++ toolchain that defines the strip action and supplies its tool.",
        )

    if _matches_any_constraint(ctx, ctx.attr._darwin):
        arguments = ["-S", "-x", "-o", output.path, source.path]
    elif _matches_any_constraint(ctx, ctx.attr._gnu_strip):
        arguments = ["--strip-unneeded", "-o", output.path, source.path]
    else:
        fail(
            "No release strip policy is defined for the target platform of {}. ".format(ctx.label) +
            "Select a Linux, Android, macOS, iOS, or Windows target platform, or add its strip policy.",
        )
    action_environment = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.strip,
        variables = cc_common.empty_variables(),
    )
    ctx.actions.run(
        executable = cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = ACTION_NAMES.strip,
        ),
        arguments = arguments,
        env = action_environment,
        inputs = depset(
            direct = [source],
            transitive = [cc_toolchain.all_files],
        ),
        mnemonic = "CcStrip",
        outputs = [output],
        progress_message = "Stripping release binary for {}".format(ctx.label),
        toolchain = _CC_TOOLCHAIN_TYPE,
    )
    return [DefaultInfo(files = depset([output]))]

release_native_binary = rule(
    implementation = _release_native_binary_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "_darwin": attr.label_list(default = [
            "@platforms//os:ios",
            "@platforms//os:macos",
        ]),
        "_gnu_strip": attr.label_list(default = [
            "@platforms//os:android",
            "@platforms//os:linux",
        ]),
        "_windows": attr.label(default = "@platforms//os:windows"),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
)
