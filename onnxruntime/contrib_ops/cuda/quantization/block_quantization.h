// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class BlockQuantize final : public CudaKernel {
 public:
  BlockQuantize(const OpKernelInfo& info) : CudaKernel(info) {
    force_fp32_scale_ = (info.GetAttrOrDefault<int64_t>("use_fp32_scale", 0LL) != 0LL);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool force_fp32_scale_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
