// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace optimizer {

struct TensorInfo {
  TensorInfo(const VectorInt64& shapes, const std::vector<float>& values);

  template <typename OutT>
  std::vector<OutT> Values() const {
    if (std::is_same<OutT, MLFloat16>::value) {
      std::vector<OutT> fp16_values;
      fp16_values.reserve(fp32_values_.size());
      ConvertFloatToMLFloat16(fp32_values_.data(),
                              reinterpret_cast<onnxruntime::MLFloat16*>(fp16_values.data()),
                              static_cast<int>(fp32_values_.size()));
      return fp16_values;
    } else if (std::is_same<OutT, float>::value) {
      return std::vector<OutT>(fp32_values_);
    } else {
      ORT_THROW("Not supported data type.");
    }
  }

  VectorInt64 Shapes() const {
    return shapes_;
  }

  VectorInt64 shapes_;
  std::vector<float> fp32_values_;
};

template <typename T>
struct AdamTestInputOutput {
  AdamTestInputOutput(
      float lr,
      int64_t step,
      const std::vector<TensorInfo>& weight_tensor_infos,
      const std::vector<TensorInfo>& gradient_tensor_infos,
      const std::vector<TensorInfo>& momentum_1_tensor_infos,
      const std::vector<TensorInfo>& momentum_2_tensor_infos,
      const std::vector<TensorInfo>& updated_weight_tensor_infos,
      const std::vector<TensorInfo>& updated_momentum_1_tensor_infos,
      const std::vector<TensorInfo>& updated_momentum_2_tensor_infos);

  SeqTensors<T>& WeightSeq() {
    return weight_seq_tensors_;
  }

  SeqTensors<T>& GradientSeq() {
    return gradient_seq_tensors_;
  }

  SeqTensors<T>& Momentum_1_Seq() {
    return momentum_1_seq_tensors_;
  }

  SeqTensors<T>& Momentum_2_Seq() {
    return momentum_2_seq_tensors_;
  }

  SeqTensors<T>& UpdatedWeightSeq() {
    return updated_weight_seq_tensors_;
  }

  SeqTensors<T>& UpdatedMomentum_1_Seq() {
    return updated_momentum_1_seq_tensors_;
  }

  SeqTensors<T>& UpdatedMomentum_2_Seq() {
    return updated_momentum_2_seq_tensors_;
  }

  std::vector<float> lr_vector;
  std::vector<int64_t> step_vector;

 private:
  SeqTensors<T> weight_seq_tensors_;
  SeqTensors<T> gradient_seq_tensors_;
  SeqTensors<T> momentum_1_seq_tensors_;
  SeqTensors<T> momentum_2_seq_tensors_;

  SeqTensors<T> updated_weight_seq_tensors_;
  SeqTensors<T> updated_momentum_1_seq_tensors_;
  SeqTensors<T> updated_momentum_2_seq_tensors_;
};

void AdamWTestLoop(
    std::unique_ptr<IExecutionProvider> execution_provider,
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
    bool* update_signal = nullptr);

}  // namespace optimizer
}  // namespace test
}  // namespace onnxruntime
