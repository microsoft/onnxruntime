// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "orttraining/test/training_ops/cuda/optimizer/common.h"

namespace onnxruntime {
namespace test {
namespace optimizer {

TensorInfo::TensorInfo(const VectorInt64& shapes, const std::vector<float>& values) {
  shapes_ = shapes;
  fp32_values_ = values;

  size_t total_size = 1;
  for (size_t i = 0; i < shapes_.size(); ++i) {
    total_size *= shapes_[i];
  }

  EXPECT_TRUE(fp32_values_.size() == total_size) << "Number of elements mismatch between shapes and values."
                                                 << "fp32_values_.size():" << fp32_values_.size()
                                                 << ", total_size: " << total_size;
}

template <typename T>
AdamTestInputOutput<T>::AdamTestInputOutput(
    float lr,
    int64_t step,
    const std::vector<TensorInfo>& weight_tensor_infos,
    const std::vector<TensorInfo>& gradient_tensor_infos,
    const std::vector<TensorInfo>& momentum_1_tensor_infos,
    const std::vector<TensorInfo>& momentum_2_tensor_infos,
    const std::vector<TensorInfo>& updated_weight_tensor_infos,
    const std::vector<TensorInfo>& updated_momentum_1_tensor_infos,
    const std::vector<TensorInfo>& updated_momentum_2_tensor_infos) {
  lr_vector.push_back(lr);
  step_vector.push_back(step);

  // Input Sequence tensors.

  for (const TensorInfo& ti : weight_tensor_infos) {
    weight_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }

  for (const TensorInfo& ti : gradient_tensor_infos) {
    gradient_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }

  for (const TensorInfo& ti : momentum_1_tensor_infos) {
    momentum_1_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }

  for (const TensorInfo& ti : momentum_2_tensor_infos) {
    momentum_2_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }

  // Update sequence tensors.

  for (const TensorInfo& ti : updated_weight_tensor_infos) {
    updated_weight_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }

  for (const TensorInfo& ti : updated_momentum_1_tensor_infos) {
    updated_momentum_1_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }

  for (const TensorInfo& ti : updated_momentum_2_tensor_infos) {
    updated_momentum_2_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
  }
}

void GetPerStepInput(
    const std::unordered_map<std::string, VectorInt64>& weight_name_shape_mapping,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_weights,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_momentum1s,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_momentum2s,
    size_t step,
    std::unordered_map<std::string, std::vector<float>>& weights_to_train,
    std::unordered_map<std::string, std::vector<float>>& momentum1_to_train,
    std::unordered_map<std::string, std::vector<float>>& momentum2_to_train) {
  weights_to_train.clear();
  momentum1_to_train.clear();
  momentum2_to_train.clear();

  for (auto it = weight_name_shape_mapping.begin(); it != weight_name_shape_mapping.end(); ++it) {
    const std::string& weight_name = it->first;

    ASSERT_TRUE(weights_to_train.find(weight_name) == weights_to_train.end());
    weights_to_train.insert({weight_name, named_weights[weight_name][step]});

    ASSERT_TRUE(momentum1_to_train.find(weight_name) == momentum1_to_train.end());
    momentum1_to_train.insert({weight_name, named_momentum1s[weight_name][step]});

    ASSERT_TRUE(momentum2_to_train.find(weight_name) == momentum2_to_train.end());
    momentum2_to_train.insert({weight_name, named_momentum2s[weight_name][step]});
  }
}

void AdamWTestLoop(
    ExecutionProviderCreationFunc execution_provider_creator,
    bool use_baseline_inputs_for_each_iteration, size_t total_step, float lr,
    float alpha, float beta, float epsilon, float weight_decay, int64_t adam_mode, int64_t correct_bias,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_weights,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_gradients,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_momentums_1,
    std::unordered_map<std::string, std::vector<std::vector<float>>>& named_momentums_2,
    std::unordered_map<std::string, VectorInt64>& weight_name_shape_mapping,
    std::pair<float, float> weight_tolerance,
    std::pair<float, float> momentum_1_tolerance,
    std::pair<float, float> momentum_2_tolerance,
    bool* update_signal) {
  std::vector<std::string> ordered_weight_names;
  for (auto it = weight_name_shape_mapping.begin(); it != weight_name_shape_mapping.end(); ++it) {
    const std::string& weight_name = it->first;
    ASSERT_TRUE(
        std::find(ordered_weight_names.begin(), ordered_weight_names.end(), weight_name) == ordered_weight_names.end());
    ordered_weight_names.push_back(weight_name);
  }

  std::unordered_map<std::string, std::vector<float>> weights_to_train;
  std::unordered_map<std::string, std::vector<float>> momentum1_to_train;
  std::unordered_map<std::string, std::vector<float>> momentum2_to_train;
  GetPerStepInput(weight_name_shape_mapping, named_weights, named_momentums_1, named_momentums_2,
                  0, weights_to_train, momentum1_to_train, momentum2_to_train);

  for (size_t step = 0; step < total_step; ++step) {
    OpTester test("AdamWOptimizer", 1, onnxruntime::kMSDomain);

    // Update the steps for each param group update.
    // Both torch and HF increase training step before applying gradients.
    // The test aligns with them.
    int64_t increased_update_count = step + 1;

    // Weights/momentums before applying optimization.
    std::vector<TensorInfo> weight_tensor_infos;
    std::vector<TensorInfo> momentum1_tensor_infos;
    std::vector<TensorInfo> momentum2_tensor_infos;
    std::vector<TensorInfo> gradient_tensor_infos;

    // Updated weights/momentums values for validation.
    std::vector<TensorInfo> updated_weight_tensor_infos;
    std::vector<TensorInfo> updated_momentum1_tensor_infos;
    std::vector<TensorInfo> updated_momentum2_tensor_infos;
    for (auto& weight_name : ordered_weight_names) {
      VectorInt64 weight_shape = weight_name_shape_mapping[weight_name];
      weight_tensor_infos.emplace_back(TensorInfo(weight_shape, weights_to_train[weight_name]));
      momentum1_tensor_infos.emplace_back(TensorInfo(weight_shape, momentum1_to_train[weight_name]));
      momentum2_tensor_infos.emplace_back(TensorInfo(weight_shape, momentum2_to_train[weight_name]));
      gradient_tensor_infos.emplace_back(TensorInfo(weight_shape, named_gradients[weight_name][step]));

      updated_weight_tensor_infos.emplace_back(TensorInfo(weight_shape, named_weights[weight_name][step + 1]));
      updated_momentum1_tensor_infos.emplace_back(TensorInfo(weight_shape, named_momentums_1[weight_name][step + 1]));
      updated_momentum2_tensor_infos.emplace_back(TensorInfo(weight_shape, named_momentums_2[weight_name][step + 1]));
    }

    AdamTestInputOutput<float> data(
        lr, increased_update_count, weight_tensor_infos, gradient_tensor_infos, momentum1_tensor_infos,
        momentum2_tensor_infos, updated_weight_tensor_infos, updated_momentum1_tensor_infos,
        updated_momentum2_tensor_infos);

    test.AddAttribute("alpha", alpha);
    test.AddAttribute("beta", beta);
    test.AddAttribute("epsilon", epsilon);
    test.AddAttribute("weight_decay", weight_decay);
    test.AddAttribute("adam_mode", adam_mode);
    test.AddAttribute("correct_bias", correct_bias);

    // Add test inputs.
    test.AddInput<float>("lr", {}, data.lr_vector);
    test.AddInput<int64_t>("step", {}, data.step_vector);
    test.AddSeqInput("weights", data.WeightSeq());
    test.AddSeqInput("gradients", data.GradientSeq());
    test.AddSeqInput("momentums_1", data.Momentum_1_Seq());
    test.AddSeqInput("momentums_2", data.Momentum_2_Seq());
    if (update_signal != nullptr) {
      test.AddInput<bool>("update_signal", {}, {*update_signal});
    }

    // Add test outputs as baseline.
    if (update_signal == nullptr || *update_signal) {
      test.AddOutput<bool>("updated_flag", {}, {1});
      test.AddSeqOutput("updated_weights", data.UpdatedWeightSeq(), weight_tolerance.first, weight_tolerance.second);
      test.AddSeqOutput("updated_momentums_1", data.UpdatedMomentum_1_Seq(), momentum_1_tolerance.first,
                        momentum_1_tolerance.second);
      test.AddSeqOutput("updated_momentums_2", data.UpdatedMomentum_2_Seq(), momentum_2_tolerance.first,
                        momentum_2_tolerance.second);

    } else {
      // No update happens.
      test.AddOutput<bool>("updated_flag", {}, {0});
      test.AddSeqOutput("updated_weights", data.WeightSeq(), weight_tolerance.first, weight_tolerance.second);
      test.AddSeqOutput("updated_momentums_1", data.Momentum_1_Seq(), momentum_1_tolerance.first,
                        momentum_1_tolerance.second);
      test.AddSeqOutput("updated_momentums_2", data.Momentum_2_Seq(), momentum_2_tolerance.first,
                        momentum_2_tolerance.second);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.emplace_back(std::move(execution_provider_creator()));

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

    if (use_baseline_inputs_for_each_iteration) {
      GetPerStepInput(weight_name_shape_mapping, named_weights, named_momentums_1, named_momentums_2,
                      step + 1, weights_to_train, momentum1_to_train, momentum2_to_train);
    } else {
      std::vector<OrtValue> outputs = test.GetFetches();
      ASSERT_EQ(outputs.size(), 4);

      const TensorSeq& updated_seq_weight = outputs[1].Get<TensorSeq>();
      const TensorSeq& updated_seq_momentum1 = outputs[2].Get<TensorSeq>();
      const TensorSeq& updated_seq_momentum2 = outputs[3].Get<TensorSeq>();

      size_t weight_index = 0;
      for (auto& weight_name : ordered_weight_names) {
        ASSERT_TRUE(weights_to_train.find(weight_name) != weights_to_train.end());
        ASSERT_TRUE(momentum1_to_train.find(weight_name) != momentum1_to_train.end());
        ASSERT_TRUE(momentum2_to_train.find(weight_name) != momentum2_to_train.end());

        const float* updated_weight_buffer = updated_seq_weight.Get(weight_index).Data<float>();
        std::copy(updated_weight_buffer, updated_weight_buffer + weights_to_train[weight_name].size(),
                  weights_to_train[weight_name].begin());

        const float* updated_momentum1_buffer = updated_seq_momentum1.Get(weight_index).Data<float>();
        std::copy(updated_momentum1_buffer, updated_momentum1_buffer + momentum1_to_train[weight_name].size(),
                  momentum1_to_train[weight_name].begin());

        const float* updated_momentum2_buffer = updated_seq_momentum2.Get(weight_index).Data<float>();
        std::copy(updated_momentum2_buffer, updated_momentum2_buffer + momentum2_to_train[weight_name].size(),
                  momentum2_to_train[weight_name].begin());

        weight_index += 1;
      }
    }
  }
}
}  // namespace optimizer
}  // namespace test
}  // namespace onnxruntime
