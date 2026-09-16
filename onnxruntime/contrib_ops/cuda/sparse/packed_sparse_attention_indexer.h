// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class PackedSparseAttentionIndexer final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit PackedSparseAttentionIndexer(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status ComputeQsa(OpKernelContext* context) const;
  Status ComputeCsa(OpKernelContext* context) const;

  packed_sparse_attention_indexer::Policy policy_;
  int64_t compress_ratio_;
  int64_t token_budget_;
  int64_t index_topk_;
  float epsilon_;
  float scale_;
  float head_weight_scale_;
  bool has_scale_;
  bool has_head_weight_scale_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
