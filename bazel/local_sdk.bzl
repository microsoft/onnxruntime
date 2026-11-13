# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Explicit local cuDNN header SDK for CUDA provider builds."""

def _cudnn_impl(ctx):
    root = ctx.os.environ.get("CUDNN_PATH")
    if not root or not ctx.path(root + "/include/cudnn.h").exists:
        fail("Set CUDNN_PATH to a cuDNN SDK directory containing include/cudnn.h before building CUDA targets.")
    ctx.symlink(root + "/include", "include")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""")

cudnn_repository = repository_rule(
    implementation = _cudnn_impl,
    environ = ["CUDNN_PATH"],
    local = True,
    configure = True,
)
