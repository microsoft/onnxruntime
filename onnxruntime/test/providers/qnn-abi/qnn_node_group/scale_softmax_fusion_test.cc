// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

GetTestModelFn BuildTestCaseScalar(
    const TestInputDef<float>& input_def,
    float scale_value,
    bool use_constant,
    bool reverse_input_order,
    std::optional<int64_t> softmax_axis = std::nullopt) {
  return [&](ModelTestBuilder& builder) -> void {
    NodeArg* input = MakeTestInput<float>(builder, input_def);
    NodeArg* scale{nullptr};
    if (use_constant) {
      onnx::TensorProto scale_value_proto;
      scale_value_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      utils::SetRawDataInTensorProto(scale_value_proto, reinterpret_cast<const char*>(&scale_value), sizeof(float));
      scale = builder.MakeIntermediate();
      builder.AddNode("Constant", {}, {scale}).AddAttribute("value", scale_value_proto);
    } else {
      scale = builder.MakeScalarInitializer<float>(scale_value);
    }
    NodeArg* intermediate = builder.MakeIntermediate();
    auto mul_inputs = reverse_input_order ? std::vector<NodeArg*>{scale, input} : std::vector<NodeArg*>{input, scale};
    builder.AddNode("Mul", mul_inputs, {intermediate});
    Node& softmax = builder.AddNode("Softmax", {intermediate}, {builder.MakeOutput()});
    if (softmax_axis.has_value()) {
      softmax.AddAttribute("axis", softmax_axis.value());
    }
  };
}

GetTestModelFn BuildTestCaseNoScalar(const TestInputDef<float>& input_def1, const TestInputDef<float>& input_def2) {
  return [&input_def1, input_def2](ModelTestBuilder& builder) -> void {
    NodeArg* input = MakeTestInput<float>(builder, input_def1);
    NodeArg* scale = MakeTestInput<float>(builder, input_def2);
    NodeArg* intermediate = builder.MakeIntermediate();
    builder.AddNode("Mul", {input, scale}, {intermediate});
    builder.AddNode("Softmax", {intermediate}, {builder.MakeOutput()});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, DISABLED_ScaleSoftmaxFusionScalarInitializer) {
  ProviderOptions provider_options = GetProviderOptions();

  auto input_def = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseScalar(input_def, 0.125f, /*use_constant=*/false, /*reverse_input_order=*/false),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, DISABLED_ScaleSoftmaxFusionScalarConstant) {
  ProviderOptions provider_options = GetProviderOptions();

  auto input_def = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseScalar(input_def, 0.375f, /*use_constant=*/true, /*reverse_input_order=*/false),
                  provider_options,
                  /*opset_version=*/14,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, DISABLED_ScaleSoftmaxFusionScalarInitializerReversed) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseScalar(input_def, 0.375f, /*use_constant=*/false, /*reverse_input_order=*/true),
                  provider_options,
                  /*opset_version=*/15,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, DISABLED_ScaleSoftmaxFusionScalarConstantReversed) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseScalar(input_def, 0.125f, /*use_constant=*/true, /*reverse_input _order=*/true),
                  provider_options,
                  /*opset_version=*/16,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, DISABLED_ScaleSoftmaxFusionSoftmaxNegativeAxis) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseScalar(input_def, 0.125f,
                                      /*use_constant=*/true, /*reverse_input_order=*/true, /*softmax_axis=*/-1),
                  provider_options,
                  /*opset_version=*/22,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, ScaleSoftmaxFusionSkipNoScalar4d) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def1 = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  auto input_def2 = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseNoScalar(input_def1, input_def2),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, ScaleSoftmaxFusionSkipNoScalar1d) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def1 = TestInputDef<float>({1, 3, 5, 5}, false, -0.5f, 0.5f);
  auto input_def2 = TestInputDef<float>({1}, false, -0.5f, 0.5f);
  RunQnnModelTest(BuildTestCaseNoScalar(input_def1, input_def2),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
