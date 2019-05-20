// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
#include "core/graph/training/generic_registry.h"
#include "core/graph/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {
namespace out_graph_optimizer {

class GradientDescent {
 public:
  struct Parameter {
    float learning_rate_;
    AllocatorPtr allocator_ptr_;
  };
  typedef Parameter ParameterType;

  GradientDescent(const ParameterType& param) : param_(param) {
  }

  NameMLValMap CalculateNewWeights(const NameMLValMap& original_weights,
                                   const NameMLValMap& gradients,
                                   size_t batch_size) const;

 private:
  ParameterType param_;
};
}  // namespace out_graph_optimizer
}  // namespace training
}  // namespace onnxruntime
