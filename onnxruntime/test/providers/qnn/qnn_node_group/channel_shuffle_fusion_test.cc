// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

GetTestModelFn BuildTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    const int64_t num_channels = 12;
    const std::vector<int64_t> input_shape{1, num_channels, 8, 8};
    auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    // Conv
    NodeArg* output_conv1 = builder.MakeIntermediate();
    NodeArg* input = MakeTestInput<float>(builder, input_def);
    const std::vector<int64_t> conv_weight_shape = {num_channels, num_channels / 2, 1, 1};
    NodeArg* conv_weight = builder.MakeInitializer<float>({conv_weight_shape}, -2.f, 2.f);
    Node& conv1 = builder.AddNode("Conv", {input, conv_weight}, {output_conv1});
    conv1.AddAttribute("group", static_cast<int64_t>(2));
    // Reshape
    NodeArg* shape1 = builder.Make1DInitializer<int64_t>({input_shape[0],
                                                          2,
                                                          num_channels / 2,
                                                          input_shape[2],
                                                          input_shape[3]});
    NodeArg* output_reshape1 = builder.MakeIntermediate();
    builder.AddNode("Reshape", {output_conv1, shape1}, {output_reshape1});
    // Transpose
    NodeArg* output_transpose = builder.MakeIntermediate();
    Node& transpose = builder.AddNode("Transpose", {output_reshape1}, {output_transpose});
    transpose.AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3, 4});
    // Reshape
    NodeArg* output_reshape2 = builder.MakeIntermediate();
    NodeArg* shape2 = builder.Make1DInitializer<int64_t>(input_shape);
    builder.AddNode("Reshape", {output_transpose, shape2}, {output_reshape2});
    // Conv
    NodeArg* output_conv2 = builder.MakeOutput();
    const std::vector<int64_t> conv_weight_shape2 = {num_channels, 1, 3, 1};
    NodeArg* conv_weight2 = builder.MakeInitializer<float>({conv_weight_shape2}, -2.f, 2.f);
    Node& conv2 = builder.AddNode("Conv", {output_reshape2, conv_weight2}, {output_conv2});
    conv2.AddAttribute("group", static_cast<int64_t>(num_channels));
    conv2.AddAttribute("kernel_shape", std::vector<int64_t>{3, 1});
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

TEST_F(QnnHTPBackendTests, ChannelShuffleFusion) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildTestCase(),
                  provider_options,
                  /*opset_version=*/10,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
