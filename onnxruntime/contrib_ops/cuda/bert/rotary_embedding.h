// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class RotaryEmbedding final : public CudaKernel {
 public:
  RotaryEmbedding(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  float scale;
  bool interleaved;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
