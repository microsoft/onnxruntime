# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Existing Node binding tests with source-assembled JS and native Bazel libraries."""

load("@aspect_rules_js//js:defs.bzl", "js_binary", "js_run_binary", "js_test")

def node_runtime_tests():
    """Declare source assembly and independently selectable existing test files."""
    js_binary(
        name = "node_package_assembler",
        entry_point = "tools/bazel/node_package.cjs",
        data = ["//bazel/node:compiler"],
    )
    js_test(
        name = "node_test_runner_test",
        entry_point = "tools/bazel/node_test_runner_test.cjs",
        data = [
            "tools/bazel/node_test_runner.cjs",
            "//bazel/node:test_dependencies",
        ],
    )
    js_run_binary(
        name = "node_js_package",
        tool = ":node_package_assembler",
        srcs = native.glob([
            "js/common/lib/*.ts",
            "js/node/lib/*.ts",
            "js/node/test/testdata/*",
        ]) + [
            "js/node/test/test-utils.ts",
            "js/node/test/ort-schema/protobuf/onnx.js",
            "js/node/test/unittests/lib/inference-session.ts",
            "js/node/test/unittests/lib/model-metadata.ts",
            "js/node/test/unittests/lib/tensor.ts",
            "js/node/test/api/simple-api-tests.ts",
            "onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc",
            "onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc",
        ],
        out_dirs = ["node_js_package"],
        args = ["node_js_package"],
    )
    targets = {
        "node_test": "all",
        "node_inference_session_test": "unittests/lib/inference-session",
        "node_model_metadata_test": "unittests/lib/model-metadata",
        "node_tensor_test": "unittests/lib/tensor",
        "node_simple_api_test": "api/simple-api-tests",
    }
    for name, file in targets.items():
        js_test(
            name = name,
            entry_point = "tools/bazel/node_test_runner.cjs",
            args = [
                "$(rootpath :node_js_package)",
                "$(rootpath :onnxruntime_binding.node)",
                "$(rootpath :runtime_library)",
                file,
            ],
            data = [
                ":node_js_package",
                ":onnxruntime_binding.node",
                ":runtime_library",
                "//bazel/node:test_dependencies",
            ],
            # The aggregate is opt-in; //... executes each file once.
            tags = ["manual"] if name == "node_test" else [],
        )
