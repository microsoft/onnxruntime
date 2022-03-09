// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
class IsAllFiniteOp final : public CudaKernel {
 public:
  IsAllFiniteOp(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t isinf_only;
    info.GetAttrOrDefault("isinf_only", &isinf_only, static_cast<int64_t>(0));
    isinf_only_ = (isinf_only != 0);

    int64_t isnan_only;
    info.GetAttrOrDefault("isnan_only", &isnan_only, static_cast<int64_t>(0));
    isnan_only_ = (isnan_only != 0);

    ORT_ENFORCE(!(isinf_only_ && isnan_only_),
                "Both attributes isinf_only and isnan_only cannot be set. Unset both to check for both conditions.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool isinf_only_, isnan_only_;
};

}  // namespace cuda
}  // namespace onnxruntime