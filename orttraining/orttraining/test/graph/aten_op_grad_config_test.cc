// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "orttraining/core/graph/aten_op_grad_config.h"

namespace onnxruntime {
namespace test {

using namespace training;

namespace {
void CompareBackwardInputSourceConfigs(const std::vector<BackwardInputSourceConfig>& configs,
                                       const std::vector<BackwardInputSourceConfig>& others) {
  EXPECT_TRUE(configs.size() == others.size());
  for (size_t i = 0; i < configs.size(); i++) {
    const auto& config = configs[i];
    const auto& other = others[i];
    EXPECT_TRUE(config.kind == other.kind && config.index == other.index &&
                config.transform_func == other.transform_func);
  }
}

void Compare(const ATenOpGradConfig& config, const ATenOpGradConfig& other) {
  EXPECT_TRUE(config.backward_op_name == other.backward_op_name);
  CompareBackwardInputSourceConfigs(config.backward_input_source_configs, other.backward_input_source_configs);
  EXPECT_TRUE(config.gradient_input_indices == other.gradient_input_indices);
}
}  // namespace

TEST(ATenOpGradConfigTest, ValidATenOpGradConfig) {
  {
    std::string grad_str = "at::a_bw(GO(0), I(1), I(0).size(0), I(2).sizes(), I(3), O(1)) -> GI(0)";
    ATenOpGradConfig config = ParseATenOpGradConfig(grad_str);
    ATenOpGradConfig expected;
    expected.backward_op_name = "at::a_bw";
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 0UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 1UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 0UL, "size(0)"));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 2UL, "sizes()"));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 3UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_OUTPUT, 1UL, ""));
    expected.gradient_input_indices.emplace_back(0UL);
    Compare(config, expected);
  }

  {
    std::string grad_str = "at::b_bw(GO(0), GO(1), I(1)) -> GI(1), GI(2)";
    ATenOpGradConfig config = ParseATenOpGradConfig(grad_str);
    ATenOpGradConfig expected;
    expected.backward_op_name = "at::b_bw";
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 0UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 1UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 1UL, ""));
    expected.gradient_input_indices.emplace_back(1UL);
    expected.gradient_input_indices.emplace_back(2UL);
    Compare(config, expected);
  }
}

TEST(ATenOpGradConfigTest, InvalidATenOpGradConfig) {
  // Invalid space.
  bool is_valid = true;
  try {
    std::string grad_str = "at::a_bw(GO(0), I(0))->GI(0)";
    ParseATenOpGradConfig(grad_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not valid.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string grad_str = "at::a_bw(GO(0),I(0)) -> GI(0)";
    ParseATenOpGradConfig(grad_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not valid.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  // Invalid type.
  is_valid = true;
  try {
    std::string grad_str = "at::a_bw(GG(0), I(0)) -> GI(0)";
    ParseATenOpGradConfig(grad_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not valid.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string grad_str = "at::a_bw(GI(0), I(0)) -> GI(1)";
    ParseATenOpGradConfig(grad_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("Input of gradient Op cannot be input's gradient.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string grad_str = "at::a_bw(GO(0), I(0)) -> I(1)";
    ParseATenOpGradConfig(grad_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("Output of gradient Op should be one of input's gradient.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  // Invalid index.
  is_valid = true;
  try {
    std::string grad_str = "at::a_bw(GO(0), I()) -> GI(0)";
    ParseATenOpGradConfig(grad_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not valid.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);
}

}  // namespace test
}  // namespace onnxruntime
