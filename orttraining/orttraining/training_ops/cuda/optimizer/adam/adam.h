// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <vector>

namespace onnxruntime {
namespace cuda {

class Adam final : public CudaKernel {
 public:
  Adam(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-8f);

    info.GetAttrOrDefault("weight_decay", &weight_decay_, 0.f);
    info.GetAttrOrDefault("adam_mode", &adam_mode_, static_cast<int64_t>(0));
    info.GetAttrOrDefault("correct_bias", &correct_bias_, static_cast<int64_t>(1));

    ORT_ENFORCE(adam_mode_ == 0 || adam_mode_ == 1, "The value of adam_mode is invalid.");
    ORT_ENFORCE(correct_bias_ == 0 || correct_bias_ == 1, "The value of adam_mode is invalid.");

    // For make sure to have torch adamw equivlance, correct_bias must be 1 for adam_mode=0.
    ORT_ENFORCE(adam_mode_ != 0 || correct_bias_ == 1, "The correct_bias should be 1 for adam_mode = 0.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float epsilon_;

  float weight_decay_;
  int64_t adam_mode_;
  int64_t correct_bias_;
};

}  // namespace cuda
}  // namespace onnxruntime
