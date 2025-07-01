// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/llm/attention_helper.h"

namespace onnxruntime {

template <typename T>
class AttentionBase : public OpKernel {
 public:
  AttentionBase(const OpKernelInfo& info) : OpKernel(info) {}

  Status ApplyAttention(OpKernelContext* context,
                        const T* Q,                                               // Q data with shape BxNxSxH
                        const T* K,                                               // K data with shape BxNxLxH
                        const T* V,                                               // V value with size BxNxLxH_v
                        const Tensor* mask_index,                                 // mask index. nullptr if no mask or its size is B
                        const Tensor* past,                                       // past state
                        const Tensor* past_key,                                   // past K input tensor (if not using past state)
                        const Tensor* past_value,                                 // past V input tensor (if not using past state)
                        Tensor* output,                                           // output tensor
                        Tensor* present_key,                                      // present K output tensor (if separating present KV)
                        Tensor* present_value,                                    // present V output tensor (if separating present KV)
                        Tensor* output_qk,                                        // Q*K output tensor (if returning Q*K value)
                        const attention_helper::AttentionParameters& parameters,  // attention parameters
                        const Tensor* attn_bias,                                  // additive bias applied on scaled QK.
                        bool past_present_share_buffer = false                    // memory optimization
  ) const;

 protected:
  void ComputeVxAttentionScore(T* output,                 // buffer for the result with size BxSxNxH_v
                               T* tmp_buffer,             // buffer for temp use with size is BxNxSxH_v
                               const T* attention_probs,  // Attention probs with size BxNxSxT
                               const T* V,                // V value with size BxNxLxH_v
                               int batch_size,            // batch size
                               int sequence_length,       // sequence length
                               int kv_sequence_length,    // sequence length of K or V
                               int past_sequence_length,  // sequence length in past state
                               int v_head_size,           // head size of V (H_v)
                               int v_hidden_size,         // hidden size of V (D_v)
                               int num_heads,             // number of attention heads
                               const T* past,             // past state
                               const T* past_value,       // past value only (if not using past state)
                               T* present,                // present state
                               T* present_value,          // present value only (if not using present state)
                               bool transpose_output,     // whether to transpose the output from BxNxSxH to BxSxNxH
                               concurrency::ThreadPool* tp,
                               bool past_present_share_buffer = false,
                               int max_sequence_length = 0) const;

  void ComputeAttentionProbs(T* attention_probs,                       // output buffer with size BxNxSxT
                             const T* Q,                               // Q data. Its size is BxNxSxH
                             const T* K,                               // k data. Its size is BxNxLxH
                             T* mask_data,                             // buffer for mask data.
                             int batch_size,                           // batch size of self-attention
                             int sequence_length,                      // sequence length of self-attention (S)
                             int kv_sequence_length,                   // sequence length of cross-attention (L)
                             int past_sequence_length,                 // sequence length of past state
                             int head_size,                            // head size of self-attention
                             int num_heads,                            // number of attention heads
                             const T* past,                            // past state
                             const T* past_key,                        // past key only (if not using past state)
                             T* present,                               // present state
                             T* present_key,                           // present key only (if not using present state)
                             T* output_qk,                             // Q*K output
                             concurrency::ThreadPool* tp,              // thread pool
                             float scale,                              // scale factor
                             const T* attn_bias_data,                  // attention bias
                             gsl::span<const int64_t> attn_bias_dims,  // attention bias shape
                             bool past_present_share_buffer = false,
                             int max_sequence_length = 0) const;

  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int num_heads,
                     int kv_sequence_length,
                     int past_sequence_length) const;

  T* ConcatStateChunk(const T* past,
                      const T* chunk,
                      T* present,
                      size_t past_chunk_length,
                      size_t present_chunk_length,
                      std::ptrdiff_t i) const;

  template <typename U>
  void PrepareMask(const U* mask_index,
                   gsl::span<const int64_t> mask_index_dims,
                   T* mask_data,
                   bool causal,
                   int batch_size,
                   int sequence_length,
                   int kv_sequence_length,
                   int past_sequence_length) const;
};

template <typename T>
class Attention final : public AttentionBase<T> {
 public:
  Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  bool is_causal_;
  int kv_num_heads_;
  int q_num_heads_;
  attention_helper::QKMatMulOutputMode qk_matmul_output_mode_;
  float scale_;
  float softcap_;
  int softmax_precision_;
};

}  // namespace onnxruntime