// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
#include "core/training/training_session.h"

namespace onnxruntime {
namespace training {

template <typename TrainingOptimizer>
class WeightUpdater {
 public:
  WeightUpdater(TrainingSession& training_session, const typename TrainingOptimizer::ParameterType& param)
      : training_session_(training_session), optimizer_(param) {
  }

  common::Status Update(const NameMLValMap& gradients, size_t batch_size) {
    if (gradients.size() < 0) {
      return common::Status(common::ONNXRUNTIME, common::FAIL);
    }

    // Get the weights from the session_state for the very first time
    if (current_weights_.empty()) {
      current_weights_ = training_session_.GetWeights();
    }

    current_weights_ = optimizer_.CalculateNewWeights(current_weights_, gradients, batch_size);
    return training_session_.UpdateWeights(current_weights_);
  }

  const NameMLValMap& GetCurrentWeight() const {
    return current_weights_;
  }

 private:
  TrainingSession& training_session_;
  TrainingOptimizer optimizer_;
  NameMLValMap current_weights_;
};
}  // namespace training
}  // namespace onnxruntime
