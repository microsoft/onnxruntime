// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class SGDOptimizerV2Base {
 public:
  class Prepare {
   public:
    // Inputs
    const Tensor* learning_rate;
    const TensorSeq* weights;
    const TensorSeq* gradients;

    size_t num_of_weights;
    std::vector<int> grouped_tensor_sizes;
    std::vector<std::vector<void*>> grouped_tensor_pointers;

    // Outputs
    Tensor* update_completed;
    TensorSeq* updated_weights;
  };

  Status PrepareForCompute(OpKernelContext* ctx, SGDOptimizerV2Base::Prepare& prepare) const;
};

}  // namespace contrib
}  // namespace onnxruntime
