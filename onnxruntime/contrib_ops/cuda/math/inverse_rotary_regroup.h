// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// Fuses the inverse rotary embedding applied to the attention output with the reshape,
// transpose and reshape that regroup the heads for the grouped output projection.
template <typename T>
class InverseRotaryRegroup final : public CudaKernel {
 public:
  InverseRotaryRegroup(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t num_groups_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
