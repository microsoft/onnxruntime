# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Restore Windows Node path conversion after top-level Bazel label parsing."""

WINDOWS_JS_ENV = select({
    "//bazel:windows": {"MSYS2_ARG_CONV_EXCL": ""},
    "//conditions:default": {},
})
