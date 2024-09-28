// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class PagedAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  PagedAttention(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float scale_;
  int64_t page_size_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t kv_quant_group_size_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
