# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native compilation and xUnit execution of existing desktop tests."""

load("@rules_dotnet//dotnet:defs.bzl", "csharp_test")

def csharp_runtime_tests():
    """Keep the original assertions, data attributes, fixtures and xUnit lifecycle."""
    fixtures = native.glob(["csharp/testdata/*"]) + [
        "onnxruntime/test/testdata/overridable_initializer.onnx",
        "onnxruntime/test/testdata/capi_symbolic_dims.onnx",
        "onnxruntime/test/testdata/if_mul.onnx",
    ]
    disk_fixtures = fixtures + [
        "onnxruntime/test/testdata/custom_op_library/custom_op_test.onnx",
        "onnxruntime/test/testdata/lora/two_params_lora_model.onnx",
        "onnxruntime/test/testdata/lora/two_params_lora_model.onnx_adapter",
    ]
    csharp_test(
        name = "csharp_tests",
        size = "medium",
        srcs = native.glob(
            ["csharp/test/Microsoft.ML.OnnxRuntime.Tests.Common/**/*.cs"],
            exclude = [
                "csharp/test/Microsoft.ML.OnnxRuntime.Tests.Common/AssemblyInfo.cs",
                "csharp/test/Microsoft.ML.OnnxRuntime.Tests.Common/TrainingTest.cs",
            ],
        ) + [
            "csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/InferenceTest.netcore.cs",
            "tools/bazel/CSharpTestRunner.cs",
        ],
        args = ["$(rootpath :runtime_library)"] + [
            "$(rootpath %s)" % path
            for path in disk_fixtures
        ],
        data = [":runtime_library"] + disk_fixtures,
        # Match the linked-resource names used by TestDataLoader, independently
        # of rules_dotnet's package-relative default naming.
        compile_data = fixtures,
        compiler_options = [
            "/resource:%s,Microsoft.ML.OnnxRuntime.Tests.NetCoreApp.TestData.%s" % (path, path.split("/")[-1])
            for path in fixtures
        ],
        allow_unsafe_blocks = True,
        defines = ["__NET_CORE_APP__"],
        keyfile = "csharp/OnnxRuntime.snk",
        out = "Microsoft.ML.OnnxRuntime.Tests.NetCoreApp",
        tags = ["manual"],
        target_frameworks = ["net10.0"],
        deps = [
            ":csharp_binding",
            "@ort_dotnet_tensors//:tensors",
            "@google_protobuf//:assembly",
            "@newtonsoft_json//:assembly",
            "@xunit_abstractions//:assembly",
            "@xunit_assert//:assembly",
            "@xunit_extensibility_core//:assembly",
            "@xunit_extensibility_execution//:assembly",
            "@xunit_skippablefact//:assembly",
        ],
    )
