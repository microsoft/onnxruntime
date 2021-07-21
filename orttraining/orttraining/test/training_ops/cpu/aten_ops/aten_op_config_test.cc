// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_config.h"

namespace onnxruntime {
namespace test {

using namespace contrib::aten_ops;

namespace {
void CompareArgumentConfigs(const std::vector<ArgumentConfig>& configs, const std::vector<ArgumentConfig>& others) {
  EXPECT_TRUE(configs.size() == others.size());
  for (size_t i = 0; i < configs.size(); i++) {
    const auto& config = configs[i];
    const auto& other = others[i];
    EXPECT_TRUE(config.kind == other.kind && config.name == other.name && config.is_cpu_tensor == other.is_cpu_tensor &&
                config.is_optional == other.is_optional);
  }
}

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

void Compare(const ATenOperatorConfig& config, const ATenOperatorConfig& other) {
  EXPECT_TRUE(config.op_name == other.op_name && config.backward_op_name == other.backward_op_name);
  CompareArgumentConfigs(config.forward_argument_configs, other.forward_argument_configs);
  CompareArgumentConfigs(config.backward_argument_configs, other.backward_argument_configs);
  CompareBackwardInputSourceConfigs(config.backward_input_source_configs, other.backward_input_source_configs);
  EXPECT_TRUE(config.forward_output_type_infer_configs == other.forward_output_type_infer_configs);
  EXPECT_TRUE(config.gradient_input_indices == other.gradient_input_indices);
  EXPECT_TRUE(config.default_int_values == other.default_int_values);
  EXPECT_TRUE(config.default_float_values == other.default_float_values);
  EXPECT_TRUE(config.default_bool_values == other.default_bool_values);
  EXPECT_TRUE(config.default_int_array_values == other.default_int_array_values);
  EXPECT_TRUE(config.default_float_array_values == other.default_float_array_values);
  EXPECT_TRUE(config.default_bool_array_values == other.default_bool_array_values);
}
}  // namespace

TEST(ATenOpConfigTest, ValidATenOpConfig) {
  {
    std::string forward_str =
        "at::a(Tensor<T> input, Tensor<int64> indices, int p=-1, bool scale=False) -> Tensor<T> output";
    std::string backward_str =
        "at::a_bw(Tensor<T> grad_output, Tensor<int64> indices, int p=-1, float d=0.5, float e=1e-4) -> Tensor<T> "
        "grad_input";
    ATenOperatorConfig config = Parse(forward_str, backward_str);
    ATenOperatorConfig expected;
    expected.op_name = "at::a";
    expected.backward_op_name = "at::a_bw";
    expected.forward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "input", false, false));
    expected.forward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "indices", false, false));
    expected.forward_argument_configs.emplace_back(ArgumentConfig(INT, "p", false, false));
    expected.forward_argument_configs.emplace_back(ArgumentConfig(BOOL, "scale", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "grad_output", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "indices", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(INT, "p", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(FLOAT, "d", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(FLOAT, "e", false, false));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 0UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 1UL, ""));
    expected.forward_output_type_infer_configs.emplace_back(std::make_pair(PROPAGATE_FROM_INPUT, 0));
    expected.gradient_input_indices.emplace_back(0UL);
    expected.default_int_values["p"] = -1;
    expected.default_bool_values["scale"] = false;
    expected.default_float_values["d"] = .5f;
    expected.default_float_values["e"] = 1e-4f;
    Compare(config, expected);
  }

  {
    std::string forward_str = "at::b(Tensor<T> weight, Tensor<U>? bias) -> (Tensor<T> r1, Tensor<float> r2)";
    std::string backward_str =
        "at::b_bw(Tensor<T> grad_r1, Tensor<float> grad_r2, Tensor<float> r2) -> (Tensor<T> grad_weight, Tensor<U> "
        "grad_bias)";
    ATenOperatorConfig config = Parse(forward_str, backward_str);
    ATenOperatorConfig expected;
    expected.op_name = "at::b";
    expected.backward_op_name = "at::b_bw";
    expected.forward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "weight", false, false));
    expected.forward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "bias", false, true));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "grad_r1", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "grad_r2", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "r2", false, false));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 0UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 1UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_OUTPUT, 1UL, ""));
    expected.forward_output_type_infer_configs.emplace_back(std::make_pair(PROPAGATE_FROM_INPUT, 0));
    expected.forward_output_type_infer_configs.emplace_back(std::make_pair(CONCRETE_TYPE, 1));
    expected.gradient_input_indices.emplace_back(0UL);
    expected.gradient_input_indices.emplace_back(1UL);
    Compare(config, expected);
  }

  {
    std::string forward_str = "at::c(Tensor<T> input, int[]? axes=[-1,0,1]) -> Tensor<T> output";
    std::string backward_str =
        "at::c_bw(Tensor<T> grad_output, bool[] flags=[True,False], float[]? es=[]) -> Tensor<T> grad_input";
    ATenOperatorConfig config = Parse(forward_str, backward_str);
    ATenOperatorConfig expected;
    expected.op_name = "at::c";
    expected.backward_op_name = "at::c_bw";
    expected.forward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "input", false, false));
    expected.forward_argument_configs.emplace_back(ArgumentConfig(INT_ARRAY, "axes", false, true));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "grad_output", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(BOOL_ARRAY, "flags", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(FLOAT_ARRAY, "es", false, true));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 0UL, ""));
    expected.forward_output_type_infer_configs.emplace_back(std::make_pair(PROPAGATE_FROM_INPUT, 0));
    expected.gradient_input_indices.emplace_back(0UL);
    expected.default_int_array_values["axes"] = {-1, 0, 1};
    expected.default_bool_array_values["flags"] = {true, false};
    expected.default_float_array_values["es"] = {};
    Compare(config, expected);
  }

  {
    std::string forward_str = "at::d(Tensor<T> weight, float<CPU> delta) -> Tensor<T> output";
    std::string backward_str =
        "at::d_bw(Tensor<T> grad_output, int<CPU>[] weight.sizes(), int<CPU> output.size(0), float<CPU> delta) -> "
        "Tensor<T> grad_weight";
    ATenOperatorConfig config = Parse(forward_str, backward_str);
    ATenOperatorConfig expected;
    expected.op_name = "at::d";
    expected.backward_op_name = "at::d_bw";
    expected.forward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "weight", false, false));
    expected.forward_argument_configs.emplace_back(ArgumentConfig(FLOAT, "delta", true, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(TENSOR, "grad_output", false, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(INT_ARRAY, "weight.sizes()", true, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(INT, "output.size(0)", true, false));
    expected.backward_argument_configs.emplace_back(ArgumentConfig(FLOAT, "delta", true, false));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(GRAD_OUTPUT, 0UL, ""));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 0UL, "sizes()"));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_OUTPUT, 0UL, "size(0)"));
    expected.backward_input_source_configs.emplace_back(BackwardInputSourceConfig(FORWARD_INPUT, 1UL, ""));
    expected.forward_output_type_infer_configs.emplace_back(std::make_pair(PROPAGATE_FROM_INPUT, 0));
    expected.gradient_input_indices.emplace_back(0UL);
    Compare(config, expected);
  }
}

TEST(ATenOpConfigTest, InvalidATenOpConfig) {
  // Invalid space.
  bool is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input)->Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not a valid function.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output,Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not a vaild argument.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  // Invalid tensor type.
  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor input) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("must have element type.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input) -> Tensor<T1> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("Unknown template type in returns.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  // Invalid gradients.
  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_result, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not forward input, output or output gradient.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_weight";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("Returnd input gradient is not for any of the forward inputs.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  // Invalid default values.
  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input, bool? flag=1) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not a valid bool string.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input, int[] p=[0.5]) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("is not a valid integer string.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);

  is_valid = true;
  try {
    std::string forward_str = "at::a(Tensor<T> input, float[]? es=0.5) -> Tensor<T> output";
    std::string backward_str = "at::a_bw(Tensor<T> grad_output, Tensor<T> input) -> Tensor<T> grad_input";
    Parse(forward_str, backward_str);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("Array values must be inside square brackets.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);
}

}  // namespace test
}  // namespace onnxruntime
