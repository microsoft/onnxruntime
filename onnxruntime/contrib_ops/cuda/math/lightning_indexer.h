// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// DeepSeek-V4's Lightning Indexer: which compressed rows each query token is allowed to attend
// to. It rotates the query, folds this step's compressed rows into the dense indexer cache,
// scores every row against every head, and keeps the best `topk`. The two projections that feed
// it stay outside as ordinary MatMuls; everything after them is about sixty small nodes per
// layer, which is what this replaces.
template <typename T>
class LightningIndexer final : public CudaKernel {
 public:
  explicit LightningIndexer(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t ratio_;
  int64_t topk_;
  int64_t max_seq_len_;
  float scale_;
  bool rotate_fp4_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
