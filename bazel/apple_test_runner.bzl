# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Hostless native XCTest execution; no Xcode project or legacy build wrapper."""

load("@rules_apple//apple:providers.bzl", "apple_provider")

def _ort_macos_test_runner_impl(ctx):
    return [
        apple_provider.make_apple_test_runner_info(
            test_runner_template = ctx.file.template,
            execution_requirements = {"requires-darwin": ""},
        ),
        DefaultInfo(),
    ]

ort_macos_test_runner = rule(
    implementation = _ort_macos_test_runner_impl,
    attrs = {
        "template": attr.label(allow_single_file = True, mandatory = True),
    },
)
