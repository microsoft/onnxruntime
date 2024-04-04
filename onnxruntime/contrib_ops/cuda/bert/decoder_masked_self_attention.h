// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T1, typename T2>
class DecoderMaskedSelfAttention final : public CudaKernel, public AttentionBase {
 public:
  DecoderMaskedSelfAttention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, true) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
