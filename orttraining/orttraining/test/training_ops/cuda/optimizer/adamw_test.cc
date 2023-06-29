// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "gtest/gtest.h"

#include "nlohmann/json.hpp"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "orttraining/test/training_ops/cuda/optimizer/common.h"

namespace onnxruntime {
namespace test {
namespace optimizer {

using json = nlohmann::json;
namespace {

constexpr const char* kParamName = "Parameters";
constexpr const char* kGradientName = "Gradients";
constexpr const char* kMomentum1Name = "Momentum1s";
constexpr const char* kMomentum2Name = "Momentum2s";

const PathString ADAM_TEST_DATA_FOLDER = ORT_TSTR("testdata/test_data_generation/adamw_test/");

void TorchAdamWSingleWeightTestLoop10Steps(bool use_baseline_inputs_for_each_iteration, bool* update_signal = nullptr) {
  size_t total_step = 10;
  float lr = 1e-03f;

  std::pair<float, float> weight_tolerance{1e-4f, 1e-5f};  // rtol, atol
  std::pair<float, float> momentum1_tolerance{1e-3f, 1e-6f};
  std::pair<float, float> momentum2_tolerance{1e-2f, 1e-7f};

  if (!use_baseline_inputs_for_each_iteration) {
    // Loose the tolerance as all states are maintained (without reloading from baseline) across different steps.
    momentum2_tolerance.first = 1e-3f;
    momentum2_tolerance.second = 1e-6f;
  }

  std::vector<std::pair<const ORTCHAR_T*, ExecutionProviderCreationFunc>> testdata_ep_pair_vector;
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cpu/adamw_test_single_weight_mode_0.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); }));
#if USE_CUDA
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cuda/adamw_test_single_weight_mode_0.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCudaExecutionProvider(); }));
#endif

  for (auto it = testdata_ep_pair_vector.begin(); it != testdata_ep_pair_vector.end(); ++it) {
    const PathString data_uri = ADAM_TEST_DATA_FOLDER + it->first;
    std::ifstream in{data_uri};

    TestDataDictType test_data;
    const json j = json::parse(in);
    j.get_to<TestDataDictType>(test_data);

    // 11 steps of weight values before applying optimization.
    WeightDictType& named_weights = test_data[kParamName];

    // 10 steps of gradient values used to apply optimization.
    WeightDictType& named_gradients = test_data[kGradientName];

    // 11 steps of momentum1 values before applying optimization.
    WeightDictType& named_momentum1s = test_data[kMomentum1Name];

    // 11 steps of momentum2 values before applying optimization.
    WeightDictType& named_momentum2s = test_data[kMomentum2Name];

    ASSERT_EQ(named_weights.size(), 1);
    ASSERT_EQ(named_gradients.size(), 1);
    ASSERT_EQ(named_momentum1s.size(), 1);
    ASSERT_EQ(named_momentum2s.size(), 1);

    ASSERT_EQ(named_weights["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_gradients["fc1.weight"].size(), total_step);
    ASSERT_EQ(named_momentum1s["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_momentum2s["fc1.weight"].size(), total_step + 1);

    std::unordered_map<std::string, VectorInt64> weight_name_shape_mapping =
        {{"fc1.weight", {2, 3}}};

    AdamWTestLoop(it->second,
                  use_baseline_inputs_for_each_iteration, total_step, lr,
                  static_cast<float>(0.9f),    // alpha
                  static_cast<float>(0.999f),  // beta
                  static_cast<float>(1e-8f),   // epsilon
                  static_cast<float>(1e-2f),   // weight_decay
                  static_cast<int64_t>(0),     // adam_mode
                  static_cast<int64_t>(1),     // correct_bias
                  named_weights, named_gradients,
                  named_momentum1s, named_momentum2s,
                  weight_name_shape_mapping,
                  weight_tolerance,
                  momentum1_tolerance,
                  momentum2_tolerance,
                  update_signal);
  }
}

TEST(AdamWTest, TorchAdamWSingleWeightTest_Loop10Steps) {
  TorchAdamWSingleWeightTestLoop10Steps(true);
}

TEST(AdamWTest, TorchAdamWSingleWeightStrictTest_Loop10Steps) {
  TorchAdamWSingleWeightTestLoop10Steps(false);
}

TEST(AdamWTest, TorchAdamWSingleWeightNoUpdateTest_Loop10Steps) {
  bool update_signal = false;
  TorchAdamWSingleWeightTestLoop10Steps(true, &update_signal);
}

void TorchAdamWMultipleWeightsTestLoop10Steps(bool use_baseline_inputs_for_each_iteration,
                                              bool* update_signal = nullptr) {
  size_t total_step = 10;
  float lr = 1e-03f;

  std::pair<float, float> weight_tolerance{1e-4f, 1e-5f};  // rtol, atol
  std::pair<float, float> momentum1_tolerance{1e-3f, 1e-6f};
  std::pair<float, float> momentum2_tolerance{1e-2f, 1e-7f};

  if (!use_baseline_inputs_for_each_iteration) {
    // Loose the tolerance as all states are maintained (without reloading from baseline) across different steps.
    momentum2_tolerance.first = 1e-3f;
    momentum2_tolerance.second = 1e-6f;
  }

  std::vector<std::pair<const ORTCHAR_T*, ExecutionProviderCreationFunc>> testdata_ep_pair_vector;
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cpu/adamw_test_multiple_weights_mode_0.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); }));
#if USE_CUDA
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cuda/adamw_test_multiple_weights_mode_0.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCudaExecutionProvider(); }));
#endif

  for (auto it = testdata_ep_pair_vector.begin(); it != testdata_ep_pair_vector.end(); ++it) {
    const PathString data_uri = ADAM_TEST_DATA_FOLDER + it->first;

    std::ifstream in{data_uri};

    TestDataDictType test_data;
    const json j = json::parse(in);
    j.get_to<TestDataDictType>(test_data);

    // 11 steps of weight values before applying optimization.
    WeightDictType& named_weights = test_data[kParamName];

    // 10 steps of gradient values used to apply optimization.
    WeightDictType& named_gradients = test_data[kGradientName];

    // 11 steps of momentum1 values before applying optimization.
    WeightDictType& named_momentum1s = test_data[kMomentum1Name];

    // 11 steps of momentum2 values before applying optimization.
    WeightDictType& named_momentum2s = test_data[kMomentum2Name];

    ASSERT_EQ(named_weights.size(), 4);
    ASSERT_EQ(named_gradients.size(), 4);
    ASSERT_EQ(named_momentum1s.size(), 4);
    ASSERT_EQ(named_momentum2s.size(), 4);

    ASSERT_EQ(named_weights["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_gradients["fc1.weight"].size(), total_step);
    ASSERT_EQ(named_momentum1s["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_momentum2s["fc1.weight"].size(), total_step + 1);

    std::unordered_map<std::string, VectorInt64> weight_name_shape_mapping =
        {{"fc1.weight", {2, 3}}, {"fc1.bias", {3}}, {"fc2.weight", {3, 2}}, {"fc2.bias", {2}}};

    AdamWTestLoop(std::move(it->second),
                  use_baseline_inputs_for_each_iteration, total_step, lr,
                  static_cast<float>(0.9f),    // alpha
                  static_cast<float>(0.999f),  // beta
                  static_cast<float>(1e-8f),   // epsilon
                  static_cast<float>(1e-2f),   // weight_decay
                  static_cast<int64_t>(0),     // adam_mode
                  static_cast<int64_t>(1),     // correct_bias
                  named_weights, named_gradients,
                  named_momentum1s, named_momentum2s,
                  weight_name_shape_mapping,
                  weight_tolerance,
                  momentum1_tolerance,
                  momentum2_tolerance,
                  update_signal);
  }
}

TEST(AdamWTest, TorchAdamWMultipleWeightsTest_Loop10Steps) {
  TorchAdamWMultipleWeightsTestLoop10Steps(true);
}

TEST(AdamWTest, TorchAdamWMultipleWeightsStrictTest_Loop10Steps) {
  TorchAdamWMultipleWeightsTestLoop10Steps(false);
}

TEST(AdamWTest, TorchAdamWMultipleWeightsNoUpdateTest_Loop10Steps) {
  bool update_signal = false;
  TorchAdamWMultipleWeightsTestLoop10Steps(true, &update_signal);
}

void HFAdamWSingleWeightTestLoop10Steps(bool use_baseline_inputs_for_each_iteration) {
  size_t total_step = 10;
  float lr = 1e-03f;

  std::pair<float, float> weight_tolerance{1e-4f, 1e-5f};  // rtol, atol
  std::pair<float, float> momentum1_tolerance{1e-3f, 1e-6f};
  std::pair<float, float> momentum2_tolerance{1e-2f, 1e-7f};

  std::vector<std::pair<const ORTCHAR_T*, ExecutionProviderCreationFunc>> testdata_ep_pair_vector;
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cpu/adamw_test_single_weight_mode_1.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); }));
#if USE_CUDA
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cuda/adamw_test_single_weight_mode_1.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCudaExecutionProvider(); }));
#endif

  for (auto it = testdata_ep_pair_vector.begin(); it != testdata_ep_pair_vector.end(); ++it) {
    const PathString data_uri = ADAM_TEST_DATA_FOLDER + it->first;
    std::ifstream in{data_uri};

    TestDataDictType test_data;
    const json j = json::parse(in);
    j.get_to<TestDataDictType>(test_data);

    // 11 steps of weight values before applying optimization.
    WeightDictType& named_weights = test_data[kParamName];

    // 10 steps of gradient values used to apply optimization.
    WeightDictType& named_gradients = test_data[kGradientName];

    // 11 steps of momentum1 values before applying optimization.
    WeightDictType& named_momentum1s = test_data[kMomentum1Name];

    // 11 steps of momentum2 values before applying optimization.
    WeightDictType& named_momentum2s = test_data[kMomentum2Name];

    ASSERT_EQ(named_weights.size(), 1);
    ASSERT_EQ(named_gradients.size(), 1);
    ASSERT_EQ(named_momentum1s.size(), 1);
    ASSERT_EQ(named_momentum2s.size(), 1);

    ASSERT_EQ(named_weights["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_gradients["fc1.weight"].size(), total_step);
    ASSERT_EQ(named_momentum1s["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_momentum2s["fc1.weight"].size(), total_step + 1);

    std::unordered_map<std::string, VectorInt64> weight_name_shape_mapping =
        {{"fc1.weight", {2, 3}}};

    AdamWTestLoop(std::move(it->second),
                  use_baseline_inputs_for_each_iteration, total_step, lr,
                  static_cast<float>(0.9f),    // alpha
                  static_cast<float>(0.999f),  // beta
                  static_cast<float>(1e-6f),   // epsilon
                  static_cast<float>(0.0f),    // weight_decay
                  static_cast<int64_t>(1),     // adam_mode
                  static_cast<int64_t>(1),     // correct_bias
                  named_weights, named_gradients,
                  named_momentum1s, named_momentum2s,
                  weight_name_shape_mapping,
                  weight_tolerance,
                  momentum1_tolerance,
                  momentum2_tolerance);
  }
}

TEST(AdamWTest, HFAdamWSingleWeightTest_Loop10Steps) {
  HFAdamWSingleWeightTestLoop10Steps(false);
}

TEST(AdamWTest, HFAdamWSingleWeightStrictTest_Loop10Steps) {
  HFAdamWSingleWeightTestLoop10Steps(true);
}

void HFAdamWMultipleWeightsTestLoop10Steps(
    bool use_baseline_inputs_for_each_iteration) {
  size_t total_step = 10;
  float lr = 1e-03f;

  std::pair<float, float> weight_tolerance{1e-4f, 1e-5f};  // rtol, atol
  std::pair<float, float> momentum1_tolerance{1e-3f, 1e-6f};
  std::pair<float, float> momentum2_tolerance{1e-2f, 1e-7f};

  if (!use_baseline_inputs_for_each_iteration) {
    // Loose the tolerance as all states are maintained (without reloading from baseline) across different steps.
    momentum2_tolerance.first = 1e-3f;
    momentum2_tolerance.second = 1e-6f;
  }

  std::vector<std::pair<const ORTCHAR_T*, ExecutionProviderCreationFunc>> testdata_ep_pair_vector;
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cpu/adamw_test_multiple_weights_mode_1.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); }));
#if USE_CUDA
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cuda/adamw_test_multiple_weights_mode_1.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCudaExecutionProvider(); }));
#endif

  for (auto it = testdata_ep_pair_vector.begin(); it != testdata_ep_pair_vector.end(); ++it) {
    const PathString data_uri = ADAM_TEST_DATA_FOLDER + it->first;
    std::ifstream in{data_uri};

    TestDataDictType test_data;
    const json j = json::parse(in);
    j.get_to<TestDataDictType>(test_data);

    // 11 steps of weight values before applying optimization.
    WeightDictType& named_weights = test_data[kParamName];

    // 10 steps of gradient values used to apply optimization.
    WeightDictType& named_gradients = test_data[kGradientName];

    // 11 steps of momentum1 values before applying optimization.
    WeightDictType& named_momentum1s = test_data[kMomentum1Name];

    // 11 steps of momentum2 values before applying optimization.
    WeightDictType& named_momentum2s = test_data[kMomentum2Name];

    ASSERT_EQ(named_weights.size(), 4);
    ASSERT_EQ(named_gradients.size(), 4);
    ASSERT_EQ(named_momentum1s.size(), 4);
    ASSERT_EQ(named_momentum2s.size(), 4);

    ASSERT_EQ(named_weights["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_gradients["fc1.weight"].size(), total_step);
    ASSERT_EQ(named_momentum1s["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_momentum2s["fc1.weight"].size(), total_step + 1);

    std::unordered_map<std::string, VectorInt64> weight_name_shape_mapping =
        {{"fc1.weight", {2, 3}}, {"fc1.bias", {3}}, {"fc2.weight", {3, 2}}, {"fc2.bias", {2}}};

    AdamWTestLoop(std::move(it->second),
                  use_baseline_inputs_for_each_iteration, total_step, lr,
                  static_cast<float>(0.9f),    // alpha
                  static_cast<float>(0.999f),  // beta
                  static_cast<float>(1e-6f),   // epsilon
                  static_cast<float>(0.0f),    // weight_decay
                  static_cast<int64_t>(1),     // adam_mode
                  static_cast<int64_t>(1),     // correct_bias
                  named_weights, named_gradients,
                  named_momentum1s, named_momentum2s,
                  weight_name_shape_mapping,
                  weight_tolerance,
                  momentum1_tolerance,
                  momentum2_tolerance);
  }
}

TEST(AdamWTest, HFAdamWMultipleWeightsTest_Loop10Steps) {
  HFAdamWMultipleWeightsTestLoop10Steps(false);
}

TEST(AdamWTest, HFAdamWMultipleWeightsStrictTest_Loop10Steps) {
  HFAdamWMultipleWeightsTestLoop10Steps(true);
}

}  // namespace

}  // namespace optimizer
}  // namespace test
}  // namespace onnxruntime
