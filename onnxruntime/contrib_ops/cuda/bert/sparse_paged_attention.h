// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/sparse_paged_attention_impl.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, typename TCACHE>
class SparsePagedAttention final : public CudaKernel {
 public:
  explicit SparsePagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  bool is_causal_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
  float softcap_;
  float qk_norm_epsilon_;
  KVQuantizationType k_quant_type_;
  KVQuantizationType v_quant_type_;
  int rotary_offset_;
  SparseAttentionMode attention_mode_;
  SelectedKvSource selected_kv_source_;
  bool auxiliary_kv_shared_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
