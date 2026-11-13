# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression for Bazel 8's missing native Catalyst ARM64 platform lookup."""

load("@apple_support//lib:apple_support.bzl", "apple_support")
load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")

def _platform_frameworks_test_impl(ctx):
    env = unittest.begin(ctx)
    for architecture in ["arm64", "x86_64"]:
        # Deliberately omit single_arch_platform: Bazel 8 throws for ARM64.
        fragment = struct(apple_platform_type = "catalyst", single_arch_cpu = architecture)
        asserts.equals(
            env,
            "__BAZEL_XCODE_DEVELOPER_DIR__/Platforms/MacOSX.platform/Developer/Library/Frameworks",
            apple_support.path_placeholders.platform_frameworks(apple_fragment = fragment),
        )
    for platform_type, platform_name in [("macos", "MacOSX"), ("ios", "iPhoneOS")]:
        fragment = struct(
            apple_platform_type = platform_type,
            single_arch_platform = struct(name_in_plist = platform_name),
        )
        asserts.equals(
            env,
            "__BAZEL_XCODE_DEVELOPER_DIR__/Platforms/{}.platform/Developer/Library/Frameworks".format(platform_name),
            apple_support.path_placeholders.platform_frameworks(apple_fragment = fragment),
        )
    return unittest.end(env)

apple_platform_test = unittest.make(_platform_frameworks_test_impl)
