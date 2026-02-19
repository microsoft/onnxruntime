// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Test GroupNormalization operator on HTP backend with default parameters
TEST_F(QnnHTPBackendTests, GroupNorm_Float_Default) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 2, 3, 4}, input_data);
    auto* scale = builder.MakeInitializer<float>({2}, scale_data);
    auto* bias = builder.MakeInitializer<float>({2}, bias_data);

    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, 2, 3, 4});
    Node& group_norm_node = builder.AddNode("GroupNormalization", {input, scale, bias}, {output});
    group_norm_node.AddAttribute("num_groups", static_cast<int64_t>(1));
    group_norm_node.AddAttribute("epsilon", 1e-05f);
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All,
                  0.01f);
}

// Test GroupNormalization operator on CPU backend
TEST_F(QnnCPUBackendTests, GroupNorm_Float_CPU) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 2, 3, 4}, input_data);
    auto* scale = builder.MakeInitializer<float>({2}, scale_data);
    auto* bias = builder.MakeInitializer<float>({2}, bias_data);

    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, 2, 3, 4});
    Node& group_norm_node = builder.AddNode("GroupNormalization", {input, scale, bias}, {output});
    group_norm_node.AddAttribute("num_groups", static_cast<int64_t>(1));
    group_norm_node.AddAttribute("epsilon", 1e-05f);
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All,
                  0.01f);
}

// Test GroupNormalization operator with multiple groups
TEST_F(QnnHTPBackendTests, GroupNorm_Float_MultipleGroups) {
  // Input with 4 channels, to be divided into 2 groups
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f,
      0.11f, 0.31f, 0.51f, 0.71f, 0.21f, 0.41f, 0.61f, 0.81f,
      0.13f, 0.33f, 0.53f, 0.73f, 0.23f, 0.43f, 0.63f, 0.83f,
      0.14f, 0.34f, 0.54f, 0.74f, 0.24f, 0.44f, 0.64f, 0.84f};
  std::vector<float> scale_data = {1.0f, 2.0f, 0.5f, 1.5f};
  std::vector<float> bias_data = {0.1f, 0.2f, 0.3f, 0.4f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 4, 3, 4}, input_data);
    auto* scale = builder.MakeInitializer<float>({4}, scale_data);
    auto* bias = builder.MakeInitializer<float>({4}, bias_data);

    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, 4, 3, 4});
    Node& group_norm_node = builder.AddNode("GroupNormalization", {input, scale, bias}, {output});
    group_norm_node.AddAttribute("num_groups", static_cast<int64_t>(2));  // 4 channels / 2 groups = 2 channels per group
    group_norm_node.AddAttribute("epsilon", 1e-05f);
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All,
                  0.01f);
}

// Test GroupNormalization operator with different epsilon value
TEST_F(QnnHTPBackendTests, GroupNorm_Float_LargeEpsilon) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 2, 3, 4}, input_data);
    auto* scale = builder.MakeInitializer<float>({2}, scale_data);
    auto* bias = builder.MakeInitializer<float>({2}, bias_data);

    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, 2, 3, 4});
    Node& group_norm_node = builder.AddNode("GroupNormalization", {input, scale, bias}, {output});
    group_norm_node.AddAttribute("num_groups", static_cast<int64_t>(1));
    group_norm_node.AddAttribute("epsilon", 0.1f);  // Larger epsilon value
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All,
                  0.01f);
}

// Test GroupNormalization operator with 3D input
TEST_F(QnnHTPBackendTests, GroupNorm_Float_3D) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f,
      0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f,
      0.25f, 0.45f, 0.65f, 0.85f};
  std::vector<float> scale_data = {1.0f, 2.0f, 0.5f, 1.5f};
  std::vector<float> bias_data = {0.1f, 0.2f, 0.3f, 0.4f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 4, 4}, input_data);
    auto* scale = builder.MakeInitializer<float>({4}, scale_data);
    auto* bias = builder.MakeInitializer<float>({4}, bias_data);

    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, 4, 4});
    Node& group_norm_node = builder.AddNode("GroupNormalization", {input, scale, bias}, {output});
    group_norm_node.AddAttribute("num_groups", static_cast<int64_t>(2));  // 4 channels / 2 groups = 2 channels per group
    group_norm_node.AddAttribute("epsilon", 1e-05f);
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All,
                  0.01f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
