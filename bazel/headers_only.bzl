# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Expose compilation requirements without linking a provider's host runtime."""

load("@rules_cc//cc:defs.bzl", "CcInfo", "cc_common")

def _headers_only_impl(ctx):
    merged = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    return [CcInfo(compilation_context = merged.compilation_context)]

headers_only = rule(
    implementation = _headers_only_impl,
    attrs = {"deps": attr.label_list(providers = [CcInfo])},
)
