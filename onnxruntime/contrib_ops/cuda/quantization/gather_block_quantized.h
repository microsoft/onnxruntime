// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class GatherBlockQuantized final : public CudaKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t gather_axis_;
  int64_t quantize_axis_;
  int64_t block_size_;
  int64_t bits_;
};

}  // namespace cuda
}  // namespace onnxruntime

