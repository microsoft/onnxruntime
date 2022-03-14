// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class NGramRepeatBlock final : public CudaKernel {
 public:
  NGramRepeatBlock(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
 private:
  int64_t ngram_size_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
