// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Shrink final : public CudaKernel {
 public:
  Shrink(const OpKernelInfo& info) : CudaKernel(info) {
    float bias_temp;
    // if the attribute exists, use the value
    if (info.GetAttr<float>("bias", &bias_temp).IsOK())
      bias_ = bias_temp;

    float lambd_temp;
    // if the attribute exists, use the value
    if (info.GetAttr<float>("lambd", &lambd_temp).IsOK())
      lambd_ = lambd_temp;
  }

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const;

 private:
  float bias_ = 0.0f;   // default as per spec
  float lambd_ = 0.5f;  // default as per spec
};

}  // namespace cuda
}  // namespace onnxruntime
