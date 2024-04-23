// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, typename U>
class GemmaRotaryEmbeddingGrad final : public CudaKernel {
 public:
  GemmaRotaryEmbeddingGrad(const OpKernelInfo& info);
  Status ComputeInternal    (OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
