// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class EmbedLayerNorm final : public CudaKernel {
 public:
  EmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
  ~EmbedLayerNorm();

 private:
  float epsilon_;
  mutable void* random_data_ = nullptr;
  bool should_randomize_ = false;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
