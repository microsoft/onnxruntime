// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

template <typename T>
class SparseAttention final : public CudaKernel {
 public:
  SparseAttention(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;            // number of attention heads for q
  int kv_num_heads_;         // number of attention heads for k and v
  float scale_;              // Scaling factor applied prior to softmax.
  bool is_causal_;           // unidirectional attention or not
  int sparse_block_size_;    // block size for sparsity
  bool do_rotary_;           // Has rotary positional embedding
  bool rotary_interleaved_;  // Interleaved rotary positional embedding
  bool disable_v1_kernel_;   // Whether disable v1 kernel and use v2 kernel for prompt.
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
