// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class Attention final : public RocmKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

  // Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed, PrePackedWeights* /*prepacked_weights*/) override {
  //   is_packed = false;

  //   if (input_idx == 1) {


  //   }

  //   return Status::OK();
  // }

  mutable IAllocatorUniquePtr<char> gemm_rcr_bias_permute_m2n3_weight;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
