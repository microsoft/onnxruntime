// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class DecoderMaskedMultiHeadAttention final : public OpKernel, public AttentionCPUBase {
 public:
  DecoderMaskedMultiHeadAttention(const OpKernelInfo& info);
  Status ApplyAttentionWithBeams(const T* Q,
                                 const T* K,
                                 const T* V,
                                 const Tensor* mask_index,
                                 const Tensor* past_key,
                                 const Tensor* past_value,
                                 Tensor* output,
                                 Tensor* present_key,
                                 Tensor* present_value,
                                 int batch_size,
                                 int past_sequence_length,
                                 int max_sequence_length,
                                 int head_size,
                                 int v_head_size,
                                 const Tensor* attn_bias,
                                 bool broadcast_attn_bias_dim_0,
                                 bool broadcast_attn_bias_dim_1,
                                 const Tensor* cache_indir,
                                 OpKernelContext* context,
                                 int beam_width,
                                 Tensor* output_qk = nullptr) const;
  void ComputeAttentionProbsWithBeams(T* attention_probs,
                                      const T* Q,
                                      const T* K,
                                      const int32_t* mask_index_data,
                                      int batch_size,
                                      int past_sequence_length,
                                      int max_sequence_length,
                                      int head_size,
                                      const T* past_key,
                                      T* present_key,
                                      ThreadPool* tp,
                                      const T* attn_bias_data,
                                      bool broadcast_attn_bias_dim_0,
                                      bool broadcast_attn_bias_dim_1,
                                      const int32_t* cache_indir_data,
                                      int beam_width,
                                      T* output_qk_data = nullptr) const;
  void ComputeVxAttentionScoreWithBeams(T* output,
                                        T* tmp_buffer,
                                        const T* attention_probs,
                                        const T* V,
                                        int batch_size,
                                        int past_sequence_length,
                                        int max_sequence_length,
                                        int v_head_size,
                                        const T* past_value,
                                        T* present_value,
                                        const int32_t* cache_indir_data,
                                        int beam_width,
                                        ThreadPool* tp) const;
  Status Compute(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
  bool past_present_share_buffer_;
  bool output_qk_;
};

}  // namespace contrib
}  // namespace onnxruntime
