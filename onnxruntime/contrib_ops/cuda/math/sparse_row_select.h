// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::CudaKernel;
using onnxruntime::cuda::OpKernelContext;
using onnxruntime::cuda::OpKernelInfo;
using onnxruntime::cuda::Status;

// Selects which compressed KV rows each query token is allowed to attend to (DeepSeek-V4's
// Lightning Indexer is one such scorer). It rotates the query, folds this step's compressed
// rows into the dense scorer cache,
// scores every row against every head, and keeps the best `topk`. The two projections that feed
// it stay outside as ordinary MatMuls; everything after them is about sixty small nodes per
// layer, which is what this replaces.
template <typename T>
class SparseRowSelect final : public CudaKernel {
 public:
  explicit SparseRowSelect(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t ratio_;
  int64_t topk_;
  int64_t row_id_offset_;
  float scale_;
  bool simulate_rotated_fp4_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
