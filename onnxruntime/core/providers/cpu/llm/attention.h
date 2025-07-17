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
                        const T* Q,                                              // Q data with shape BxNxSxH
                        const T* K,                                              // K data with shape BxNxLxH
                        const T* V,                                              // V value with size BxNxLxH_v
                        const Tensor* mask_index,                                // mask index. nullptr if no mask or its size is B
                        const Tensor* past_key,                                  // past K input tensor (if not using past state)
                        const Tensor* past_value,                                // past V input tensor (if not using past state)
                        Tensor* output,                                          // output tensor
                        Tensor* present_key,                                     // present K output tensor (if separating present KV)
                        Tensor* present_value,                                   // present V output tensor (if separating present KV)
                        Tensor* output_qk,                                       // Q*K output tensor (if returning Q*K value)
                        const attention_helper::AttentionParameters& parameters  // attention parameters
  ) const;

 protected:
  void ComputeVxAttentionScore(T* output,                  // buffer for the result with size BxSxNxH_v
                               const T* attention_probs,   // Attention probs with size BxNxSxT
                               const T* V,                 // V value with size BxNxLxH_v
                               int batch_size,             // batch size
                               int sequence_length,        // sequence length
                               int kv_sequence_length,     // sequence length of K or V
                               int past_sequence_length,   // sequence length in past state
                               int total_sequence_length,  // total sequence length = past_sequence_length + kv_sequence_length
                               int v_head_size,            // head size of V (H_v)
                               int num_heads,              // number of attention heads
                               int kv_num_heads,           // number of KV heads
                               const T* past_value,        // past value only (if not using past state)
                               T* present_value,           // present value only (if not using present state)
                               bool transpose_output,      // whether to transpose the output from BxNxSxH to BxSxNxH
                               concurrency::ThreadPool* tp) const;

  void ComputeAttentionProbs(T* attention_probs,                                       // output buffer with size BxNxSxT
                             const T* Q,                                               // Q data. Its size is BxNxSxH
                             const T* K,                                               // k data. Its size is BxNxLxH
                             const Tensor* mask_index,                                 // mask_index
                             const attention_helper::AttentionParameters& parameters,  // attention parameters
                             const T* past_key,                                        // past key only (if not using past state)
                             T* present_key,                                           // present key only (if not using present state)
                             T* output_qk,                                             // Q*K output
                             concurrency::ThreadPool* tp,
                             AllocatorPtr allocator) const;

  T* ConcatStateChunk(const T* past,
                      const T* chunk,
                      T* present,
                      size_t past_chunk_length,
                      size_t input_chunk_length,
                      size_t present_chunk_length,
                      size_t num_heads,
                      size_t head_size,
                      std::ptrdiff_t batch_i,
                      std::ptrdiff_t head_i,
                      bool transposed) const;
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