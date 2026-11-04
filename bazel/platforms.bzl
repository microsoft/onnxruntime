# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Desktop shared-library names and linker conventions."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

def desktop_shared_library(name, linux, windows, macos, **kwargs):
    """Build the platform-native artifact and expose a stable target alias."""
    for os, artifact in [("linux", linux), ("windows", windows), ("macos", macos)]:
        cc_binary(
            name = artifact,
            linkshared = True,
            target_compatible_with = ["@platforms//os:" + os],
            **kwargs
        )
    native.alias(
        name = name,
        actual = select({
            "//bazel:windows": ":" + windows,
            "//bazel:macos": ":" + macos,
            "//conditions:default": ":" + linux,
        }),
    )

ORIGIN_RPATH = select({
    "//bazel:windows": [],
    "//bazel:macos": ["-Wl,-rpath,@loader_path"],
    "//conditions:default": ["-Wl,-rpath,'$$ORIGIN'"],
})
