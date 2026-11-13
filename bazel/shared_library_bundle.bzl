# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Link a shared runtime while co-locating its lazily loaded plugins."""

load("@rules_cc//cc:defs.bzl", "CcInfo", "cc_common")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")

def _shared_library_bundle_impl(ctx):
    toolchain = find_cc_toolchain(ctx)
    features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    files = [ctx.file.runtime] + ctx.files.plugins
    libraries = [
        cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = features,
            cc_toolchain = toolchain,
            dynamic_library = file,
        )
        for file in files
    ]

    # Plugins need matching _solib runfiles, but must not become DT_NEEDED entries:
    # their initializers require the host interface installed by the runtime.
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(libraries[:1]),
    )
    return [
        CcInfo(linking_context = cc_common.create_linking_context(
            linker_inputs = depset([linker_input]),
        )),
        DefaultInfo(
            files = depset([ctx.file.runtime]),
            runfiles = ctx.runfiles(files = files + [library.dynamic_library for library in libraries]),
        ),
    ]

shared_library_bundle = rule(
    implementation = _shared_library_bundle_impl,
    attrs = {
        "runtime": attr.label(allow_single_file = [".so"], mandatory = True),
        "plugins": attr.label_list(allow_files = [".so"]),
    },
    fragments = ["cpp"],
    provides = [CcInfo],
    toolchains = use_cc_toolchain(),
)
