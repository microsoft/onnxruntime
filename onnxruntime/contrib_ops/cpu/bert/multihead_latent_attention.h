// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

// Absorbed multi-head latent attention over one latent KV cache shared by every query head. See the
// MultiHeadLatentAttention schema in bert_defs.cc for the contract.
template <typename T>
class MultiHeadLatentAttention final : public OpKernel {
 public:
  explicit MultiHeadLatentAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int num_heads_;
  int v_head_size_;
  int v_offset_;
  float scale_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace contrib
}  // namespace onnxruntime
