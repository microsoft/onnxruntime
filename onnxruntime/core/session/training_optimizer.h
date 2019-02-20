// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/framework/ml_value.h"
#include "core/common/status.h"
#include "core/framework/framework_common.h"
#include "core/session/training_session.h"

namespace onnxruntime {
namespace training {

class GradientDescent {
 public:
  struct Parameter {
    float learning_rate_;
    AllocatorPtr allocator_ptr_;
  };
  using ParameterType = Parameter;

  GradientDescent(ParameterType param) : param_(param) {
  }

  NameMLValMap CalculateNewWeights(const NameMLValMap& original_weights,
                                   const std::vector<NameMLValMap>& gradients_multi_batches) const;

 private:
  ParameterType param_;
};

template <typename OptimizationMethod>
class Optimizer {
 public:
  Optimizer(TrainingSession& training_session, const typename OptimizationMethod::ParameterType& param)
      : training_session_(training_session), optimization_(param) {
  }

  common::Status Optimizer::Optimize(const std::vector<NameMLValMap>& gradients_multi_batches) {
    if (gradients_multi_batches.size() < 0) {
      return common::Status(common::ONNXRUNTIME, common::FAIL);
    }

    // Get the weights from the session_state for the very first time
    if (current_weights_.empty()) {
      current_weights_ = training_session_.GetWeights();
    }

    current_weights_ = optimization_.CalculateNewWeights(current_weights_, gradients_multi_batches);
    return training_session_.UpdateWeights(current_weights_);
  }

  const NameMLValMap& GetCurrentWeight() const {
    return current_weights_;
  }

 private:
  TrainingSession& training_session_;
  OptimizationMethod optimization_;
  NameMLValMap current_weights_;
};
}  // namespace training
}  // namespace onnxruntime
