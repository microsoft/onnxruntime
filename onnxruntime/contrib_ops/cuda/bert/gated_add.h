// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// output = x + round_to_T(y * gate), with gate broadcast across the last dimension.
template <typename T>
class GatedAdd final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit GatedAdd(const OpKernelInfo& info)
      : CudaKernel(info),
        sigmoid_in_fp32_(info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1) == 1) {
    const int64_t fp32 = info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1);
    ORT_ENFORCE(fp32 == 0 || fp32 == 1, "GatedAdd sigmoid_in_fp32 must be zero or one");
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool sigmoid_in_fp32_{true};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime