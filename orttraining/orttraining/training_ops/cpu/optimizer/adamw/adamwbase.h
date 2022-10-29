// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AdamWOptimizerBase {
 public:
  class Prepare {
   public:
    // Inputs
    const Tensor* learning_rate;
    const Tensor* step;
    const TensorSeq* weights;
    const TensorSeq* gradients;
    const TensorSeq* momentums_1;
    const TensorSeq* momentums_2;

    size_t num_of_weights;
    std::vector<int> grouped_tensor_sizes;
    std::vector<std::vector<void*>> grouped_tensor_pointers;

    // Outputs
    Tensor* updated_flag;
    TensorSeq* updated_weights;
    TensorSeq* updated_momentums_1;
    TensorSeq* updated_momentums_2;
  };

  AdamWOptimizerBase(const OpKernelInfo& info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-8f);

    info.GetAttrOrDefault("weight_decay", &weight_decay_, 0.f);
    info.GetAttrOrDefault("adam_mode", &adam_mode_, static_cast<int64_t>(0));
    info.GetAttrOrDefault("correct_bias", &correct_bias_, static_cast<int64_t>(1));

    ORT_ENFORCE(adam_mode_ == 0 || adam_mode_ == 1, "The value of adam_mode is invalid.");
    ORT_ENFORCE(correct_bias_ == 0 || correct_bias_ == 1, "The value of correct_bias is invalid.");

    // To have torch adamw equivalence, correct_bias must be 1 for adam_mode=0.
    ORT_ENFORCE(adam_mode_ != 0 || correct_bias_ == 1, "The correct_bias should be 1 for adam_mode = 0.");
  }

  Status PrepareForCompute(OpKernelContext* ctx, AdamWOptimizerBase::Prepare& prepare) const;

  Status GenerateOutputs(OpKernelContext* ctx, size_t number_of_values,
                         const TensorSeq* values, TensorSeq* updated_values) const;

 protected:
  virtual Status CopyInputTensorToOutputTensor(const Tensor& source_tensor, Tensor& dest_tensor) const = 0;

  float alpha_;
  float beta_;
  float epsilon_;

  float weight_decay_;
  int64_t adam_mode_{0};
  int64_t correct_bias_{0};
};

}  // namespace contrib
}  // namespace onnxruntime
