// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/layer_norm_impl.h"

namespace onnxruntime::contrib {

template <typename T>
class BranchwiseRMSNorm final : public LayerNormImpl {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 private:
  float epsilon_;
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib
