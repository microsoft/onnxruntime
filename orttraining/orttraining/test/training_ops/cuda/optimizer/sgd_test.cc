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

template <typename T>
struct SGDTestInputOutput {
  SGDTestInputOutput(
      float lr,
      int64_t step,
      const std::vector<TensorInfo>& weight_tensor_infos,
      const std::vector<TensorInfo>& gradient_tensor_infos,
      const std::vector<TensorInfo>& updated_weight_tensor_infos) {
    lr_vector.push_back(lr);
    step_vector.push_back(step);

    // Input Sequence tensors.

    for (const TensorInfo& ti : weight_tensor_infos) {
      weight_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    for (const TensorInfo& ti : gradient_tensor_infos) {
      gradient_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    // Update sequence tensors.

    for (const TensorInfo& ti : updated_weight_tensor_infos) {
      updated_weight_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }
  }

  SeqTensors<T>& WeightSeq() {
    return weight_seq_tensors_;
  }

  SeqTensors<T>& GradientSeq() {
    return gradient_seq_tensors_;
  }

  SeqTensors<T>& UpdatedWeightSeq() {
    return updated_weight_seq_tensors_;
  }

  std::vector<float> lr_vector;
  std::vector<int64_t> step_vector;

 private:
  SeqTensors<T> weight_seq_tensors_;
  SeqTensors<T> gradient_seq_tensors_;

  SeqTensors<T> updated_weight_seq_tensors_;
};

void GetPerStepInput(
    const std::unordered_map<std::string, VectorInt64>& weight_name_shape_mapping,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_weights,
    size_t step,
    std::unordered_map<std::string, std::vector<float>>& weights_to_train) {
  weights_to_train.clear();

  for (auto it = weight_name_shape_mapping.begin(); it != weight_name_shape_mapping.end(); ++it) {
    const std::string& weight_name = it->first;

    ASSERT_TRUE(weights_to_train.find(weight_name) == weights_to_train.end());
    weights_to_train.insert({weight_name, named_weights[weight_name][step]});
  }
}

void SGDTestLoop(
    ExecutionProviderCreationFunc execution_provider_creator,
    bool use_baseline_inputs_for_each_iteration, size_t total_step, float lr,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_weights,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_gradients,
    std::unordered_map<std::string, VectorInt64>& weight_name_shape_mapping,
    std::pair<float, float> weight_tolerance,
    bool* update_signal) {
  std::vector<std::string> ordered_weight_names;
  for (auto it = weight_name_shape_mapping.begin(); it != weight_name_shape_mapping.end(); ++it) {
    const std::string& weight_name = it->first;
    ASSERT_TRUE(
        std::find(ordered_weight_names.begin(), ordered_weight_names.end(), weight_name) == ordered_weight_names.end());
    ordered_weight_names.push_back(weight_name);
  }

  std::unordered_map<std::string, std::vector<float>> weights_to_train;
  GetPerStepInput(weight_name_shape_mapping, named_weights, 0, weights_to_train);

  for (size_t step = 0; step < total_step; ++step) {
    OpTester test("SGDOptimizerV2", 1, onnxruntime::kMSDomain);

    // Weights/momentums before applying optimization.
    std::vector<TensorInfo> weight_tensor_infos;
    std::vector<TensorInfo> gradient_tensor_infos;

    // Updated weights/momentums values for validation.
    std::vector<TensorInfo> updated_weight_tensor_infos;

    for (auto& weight_name : ordered_weight_names) {
      VectorInt64 weight_shape = weight_name_shape_mapping[weight_name];
      weight_tensor_infos.emplace_back(TensorInfo(weight_shape, weights_to_train[weight_name]));
      gradient_tensor_infos.emplace_back(TensorInfo(weight_shape, named_gradients[weight_name][step]));

      updated_weight_tensor_infos.emplace_back(TensorInfo(weight_shape, named_weights[weight_name][step + 1]));
    }

    SGDTestInputOutput<float> data(lr, step, weight_tensor_infos, gradient_tensor_infos, updated_weight_tensor_infos);

    // Add test inputs.
    test.AddInput<float>("lr", {}, data.lr_vector);
    test.AddSeqInput("weights", data.WeightSeq());
    test.AddSeqInput("gradients", data.GradientSeq());
    if (update_signal != nullptr) {
      test.AddInput<bool>("update_signal", {}, {*update_signal});
    }

    // Add test outputs as baseline.
    if (update_signal == nullptr || *update_signal) {
      test.AddOutput<bool>("update_completed", {}, {true});
      test.AddSeqOutput("updated_weights", data.UpdatedWeightSeq(), weight_tolerance.first, weight_tolerance.second);

    } else {
      // No update happens.
      test.AddOutput<bool>("update_completed", {}, {false});
      test.AddSeqOutput("updated_weights", data.WeightSeq(), weight_tolerance.first, weight_tolerance.second);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.emplace_back(std::move(execution_provider_creator()));

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

    if (use_baseline_inputs_for_each_iteration) {
      GetPerStepInput(weight_name_shape_mapping, named_weights, step + 1, weights_to_train);
    } else {
      std::vector<OrtValue> outputs = test.GetFetches();
      ASSERT_EQ(outputs.size(), 2);

      const TensorSeq& updated_seq_weight = outputs[1].Get<TensorSeq>();

      size_t weight_index = 0;
      for (auto& weight_name : ordered_weight_names) {
        ASSERT_TRUE(weights_to_train.find(weight_name) != weights_to_train.end());

        const float* updated_weight_buffer = updated_seq_weight.Get(weight_index).Data<float>();
        std::copy(updated_weight_buffer, updated_weight_buffer + weights_to_train[weight_name].size(),
                  weights_to_train[weight_name].begin());

        weight_index += 1;
      }
    }
  }
}

constexpr const char* kParamName = "Parameters";
constexpr const char* kGradientName = "Gradients";

const PathString SGD_TEST_DATA_FOLDER = ORT_TSTR("testdata/test_data_generation/sgd_test/");

void SGDSingleWeightTestLoop10Steps(bool use_baseline_inputs_for_each_iteration, bool* update_signal = nullptr) {
  size_t total_step = 10;
  float lr = 1e-03f;

  std::pair<float, float> weight_tolerance{1e-4f, 1e-5f};  // rtol, atol

  std::vector<std::pair<const ORTCHAR_T*, ExecutionProviderCreationFunc>> testdata_ep_pair_vector;
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cpu/sgd_test_single_weight.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); }));
#if USE_CUDA
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cuda/sgd_test_single_weight.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCudaExecutionProvider(); }));
#endif

  for (auto it = testdata_ep_pair_vector.begin(); it != testdata_ep_pair_vector.end(); ++it) {
    const PathString data_uri = SGD_TEST_DATA_FOLDER + it->first;
    std::ifstream in{data_uri};

    TestDataDictType test_data;
    const json j = json::parse(in);
    j.get_to<TestDataDictType>(test_data);

    // 11 steps of weight values before applying optimization.
    WeightDictType& named_weights = test_data[kParamName];

    // 10 steps of gradient values used to apply optimization.
    WeightDictType& named_gradients = test_data[kGradientName];

    ASSERT_EQ(named_weights.size(), 1);
    ASSERT_EQ(named_gradients.size(), 1);

    ASSERT_EQ(named_weights["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_gradients["fc1.weight"].size(), total_step);

    std::unordered_map<std::string, VectorInt64> weight_name_shape_mapping =
        {{"fc1.weight", {2, 3}}};

    SGDTestLoop(it->second,
                use_baseline_inputs_for_each_iteration, total_step, lr,
                named_weights, named_gradients,
                weight_name_shape_mapping,
                weight_tolerance,
                update_signal);
  }
}

TEST(SGDOptimizerV2Test, SGDSingleWeightTest_Loop10Steps) {
  SGDSingleWeightTestLoop10Steps(true);
}

TEST(SGDOptimizerV2Test, SGDSingleWeightStrictTest_Loop10Steps) {
  SGDSingleWeightTestLoop10Steps(false);
}

TEST(SGDOptimizerV2Test, SGDSingleWeightNoUpdateTest_Loop10Steps) {
  bool update_signal = false;
  SGDSingleWeightTestLoop10Steps(true, &update_signal);
}

void SGDMultipleWeightsTestLoop10Steps(bool use_baseline_inputs_for_each_iteration,
                                       bool* update_signal = nullptr) {
  size_t total_step = 10;
  float lr = 1e-03f;

  std::pair<float, float> weight_tolerance{1e-4f, 1e-5f};  // rtol, atol

  std::vector<std::pair<const ORTCHAR_T*, ExecutionProviderCreationFunc>> testdata_ep_pair_vector;
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cpu/sgd_test_multiple_weights.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); }));
#if USE_CUDA
  testdata_ep_pair_vector.push_back(std::make_pair(
      ORT_TSTR("cuda/sgd_test_multiple_weights.json"),
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCudaExecutionProvider(); }));
#endif

  for (auto it = testdata_ep_pair_vector.begin(); it != testdata_ep_pair_vector.end(); ++it) {
    const PathString data_uri = SGD_TEST_DATA_FOLDER + it->first;

    std::ifstream in{data_uri};

    TestDataDictType test_data;
    const json j = json::parse(in);
    j.get_to<TestDataDictType>(test_data);

    // 11 steps of weight values before applying optimization.
    WeightDictType& named_weights = test_data[kParamName];

    // 10 steps of gradient values used to apply optimization.
    WeightDictType& named_gradients = test_data[kGradientName];

    ASSERT_EQ(named_weights.size(), 4);
    ASSERT_EQ(named_gradients.size(), 4);

    ASSERT_EQ(named_weights["fc1.weight"].size(), total_step + 1);
    ASSERT_EQ(named_gradients["fc1.weight"].size(), total_step);

    std::unordered_map<std::string, VectorInt64> weight_name_shape_mapping =
        {{"fc1.weight", {2, 3}}, {"fc1.bias", {3}}, {"fc2.weight", {3, 2}}, {"fc2.bias", {2}}};

    SGDTestLoop(std::move(it->second),
                use_baseline_inputs_for_each_iteration, total_step, lr,
                named_weights, named_gradients,
                weight_name_shape_mapping,
                weight_tolerance,
                update_signal);
  }
}

TEST(SGDOptimizerV2Test, SGDMultipleWeightsTest_Loop10Steps) {
  SGDMultipleWeightsTestLoop10Steps(true);
}

TEST(SGDOptimizerV2Test, SGDMultipleWeightsStrictTest_Loop10Steps) {
  SGDMultipleWeightsTestLoop10Steps(false);
}

TEST(SGDOptimizerV2Test, SGDMultipleWeightsNoUpdateTest_Loop10Steps) {
  bool update_signal = false;
  SGDMultipleWeightsTestLoop10Steps(true, &update_signal);
}

}  // namespace

}  // namespace optimizer
}  // namespace test
}  // namespace onnxruntime
