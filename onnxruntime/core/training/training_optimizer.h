// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"

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
                                   const NameMLValMap& gradients,
                                   size_t batch_size) const;

 private:
  ParameterType param_;
};
}  // namespace training
}  // namespace onnxruntime
