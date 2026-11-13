# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Check actual static XCFramework actions without compiling Apple binaries."""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")

def _intermediates_impl(ctx):
    env = analysistest.begin(ctx)
    target = analysistest.target_under_test(env)
    lipo_outputs = [
        output
        for action in analysistest.target_actions(env)
        if action.mnemonic == "AppleLipo"
        for output in action.outputs.to_list()
    ]
    asserts.true(env, len(lipo_outputs) > 0, "Expected real universal-library lipo actions.")
    for output in lipo_outputs:
        asserts.true(
            env,
            output.basename.startswith(target.label.name + "_"),
            "Intermediate must use the target name, not the shared framework bundle name: " + output.path,
        )
    return analysistest.end(env)

xcframework_intermediates_test = analysistest.make(_intermediates_impl)
