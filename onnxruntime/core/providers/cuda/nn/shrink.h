// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl_util"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Shrink final : public CudaKernel {
 public:
  Shrink(const OpKernelInfo& op_kernel_info)
      : CudaKernel{op_kernel_info} {
    float bias_temp;
    // if the attribute exists, use the value
    if(op_kernel_info.GetAttr<float>("bias", &bias_temp).IsOK())
       bias_ = gsl::narrow_cast<float>(bias_temp);

    float lambd_temp;
    // if the attribute exists, use the value
    if(op_kernel_info.GetAttr<float>("lambd", &lambd_temp).IsOK())
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float bias_ = 0.0f; // default as per spec
  int64_t lambd_ = 0.5f;  // default as per spec
};

}  // namespace cuda
}  // namespace onnxruntime
