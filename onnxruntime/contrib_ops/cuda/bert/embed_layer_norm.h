// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class EmbedLayerNorm final : public CudaKernel {
 public:
  EmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
 private:
  float epsilon_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
