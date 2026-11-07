# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bounded libtest selection against the source-built runtime."""

load("@rules_python//python:defs.bzl", "py_test")

def rust_selected_test(name, binary, manifest, cases, data = []):
    """Execute exact existing libtest cases, never silently accepting zero tests."""
    py_test(
        name = name,
        srcs = ["//:tools/bazel/rust_test_runner.py"],
        main = "//:tools/bazel/rust_test_runner.py",
        args = [
            "$(rlocationpath %s)" % binary,
            "$(rlocationpath //:runtime_library)",
            "$(rlocationpath %s)" % manifest,
        ] + cases,
        data = [binary, manifest, "//:runtime_library"] + data,
        deps = ["@rules_python//python/runfiles"],
    )
