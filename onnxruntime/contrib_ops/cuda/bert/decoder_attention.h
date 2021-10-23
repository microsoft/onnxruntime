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
class DecoderAttention final : public CudaKernel {
 public:
  DecoderAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  int num_heads_;
  bool static_kv_;
  bool use_past_;
  bool has_layer_state_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
