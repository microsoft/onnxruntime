// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "attention_base.h"
#include "attention_helper.h"

#include "core/common/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class GQAAttentionBase : public AttentionBase {
 protected:
  GQAAttentionBase(const OpKernelInfo& info, bool require_same_hidden_size)
      : AttentionBase(info, require_same_hidden_size) {}

  int local_window_size_;
  bool do_rotary_;
  bool rotary_interleaved_;

  template <typename T>
  Status ApplyAttention(const T* Q,                                 // Q data with shape BxNxSxH
                        const T* K,                                 // K data with shape BxN_kvxSxH
                        const T* V,                                 // V data with shape BxN_kvxSxH
                        const Tensor* past_key,                     // past K input tensor (if not using past state)
                        const Tensor* past_value,                   // past V input tensor (if not using past state)
                        Tensor* output,                             // output tensor
                        Tensor* present_key,                        // present K output tensor (if separating present KV)
                        Tensor* present_value,                      // present V output tensor (if separating present KV)
                        const Tensor* seqlens_k,                    // past sequence lengths tensor
                        GroupQueryAttentionParameters& parameters,  // attention parameters
                        AllocatorPtr allocator,                     // allocator for temporary tensors
                        OpKernelContext* context) const {
    const int batch_size = parameters.batch_size;
    const int sequence_length = parameters.sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = static_cast<int>(present_key->Shape().GetDims()[2]);

    // Compute the attention score.
    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * seqlen_present_kv_cache * sizeof(T);
    auto attention_probs = allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

    bool past_present_share_buffer = past_key_data == present_key_data && past_value_data == present_value_data;

    const T* k = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;
    ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), Q, k, seqlens_k->Data<int32_t>(), batch_size,
                             sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size, past_key_data,
                             present_key_data, past_present_share_buffer, packed_qkv, tp);

    // Compute the attentionScore * Value: out(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
    const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;
    ComputeVxAttentionScore(output->MutableData<T>(), static_cast<T*>(attention_probs), v, seqlens_k->Data<int32_t>(),
                            batch_size, sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size,
                            hidden_size, past_value_data, present_value_data, past_present_share_buffer, packed_qkv,
                            tp);

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  template <typename T>
  void ComputeAttentionProbs(T* attention_probs,                  // output buffer with size BxNxSxT
                             const T* Q,                          // Q data. Its size is BxNxSxH
                             const T* K,                          // k data. Its size is BxNxLxH
                             const int32_t* seqlens_k,            // past sequence lengths tensor
                             int batch_size,                      // batch size of self-attention
                             int sequence_length,                 // sequence length of self-attention (S)
                             int past_buffer_sequence_length,     // sequence length of past state
                             int present_buffer_sequence_length,  // sequence length of present state
                             int head_size,                       // head size of self-attention
                             const T* past_key,                   // past key only
                             T* present_key,                      // present key only
                             bool past_present_share_buffer,      // whether present key and value share the same buffer
                             bool packed_qkv,                     // whether Q, K, V are packed
                             ThreadPool* tp) const {              // thread pool
    const bool is_prompt = sequence_length != 1;
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const int kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t q_input_chunk_length = static_cast<size_t>(sequence_length) * head_size;                      // S x H
    const size_t kv_input_chunk_length = static_cast<size_t>(sequence_length) * head_size;                     // L x H
    const size_t past_buff_chunk_length = static_cast<size_t>(past_buffer_sequence_length) * head_size;        // L x H
    const size_t present_buff_chunk_length = static_cast<size_t>(present_buffer_sequence_length) * head_size;  // T x H

    if (!past_present_share_buffer) {
      memset(present_key, 0, batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
    }

    const int loop_len = batch_size * num_heads_;
    const float alpha = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    TensorOpCost unit_cost;
    const ptrdiff_t probs_matrix_bytes =
        SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * sizeof(T);
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * present_buffer_sequence_length);
    unit_cost.bytes_loaded =
        static_cast<double>((sequence_length + present_buffer_sequence_length) * head_size * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

    unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes);
    unit_cost.bytes_stored += static_cast<double>(probs_matrix_bytes);

    if (present_key) {
      double bytes_to_copy_key = static_cast<double>(sizeof(T) * present_buff_chunk_length);
      unit_cost.bytes_loaded += bytes_to_copy_key;
      unit_cost.bytes_stored += bytes_to_copy_key;
    }

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>(i) / num_heads_;
        const int head_index = static_cast<int>(i) % num_heads_;
        const int past_seqlen =
            sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index]) : past_buffer_sequence_length;
        const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
        const int total_seqlen = seqlens_k[batch_index] + 1;

        const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * sequence_length * present_buffer_sequence_length;
        T* output = attention_probs + output_offset;

        const T* k;
        if (packed_qkv) {
          k = K + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          k = K + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_key) {
          k = ConcatStateChunkGQA(past_key, k, present_key, present_buff_chunk_length, past_buff_chunk_length,
                                  past_chunk_length, kv_input_chunk_length, is_prompt, past_present_share_buffer,
                                  i / kv_num_heads_factor);
        }

        // Compute Q*K' + AttentionMask
        //                     original                 transposed             each iteration
        // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
        // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
        // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
        const T* q;
        if (packed_qkv) {
          q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        } else {
          q = Q + q_input_chunk_length * i;
        }
        math::GemmEx<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q,
                                    head_size, k, head_size, 0.0f /*bata*/, output, present_buffer_sequence_length,
                                    nullptr);

        // compute Softmax
        T* output_softmax = output;
        for (int seq = 0; seq < sequence_length; seq++) {
          int seq_causal_length = sequence_length == 1 ? total_seqlen : seq + 1;
          if (local_window_size_ > 0 && seq_causal_length > local_window_size_ + 1) {
            for (int total_seq_id = 0; total_seq_id < seq_causal_length - local_window_size_ - 1; total_seq_id++) {
              output_softmax[total_seq_id] = 0.f;
            }
            ComputeAttentionSoftmaxInplace(output_softmax + seq_causal_length - local_window_size_ - 1, 1,
                                           local_window_size_ + 1, nullptr);
          } else {
            ComputeAttentionSoftmaxInplace(output_softmax, 1, seq_causal_length, nullptr);
          }

          // set causal [seq_causal_length, total_seqlen) to 0.f
          for (int total_seq_id = seq_causal_length; total_seq_id < total_seqlen; total_seq_id++) {
            output_softmax[total_seq_id] = 0.f;
          }

          output_softmax += present_buffer_sequence_length;
        }
      }
    });
  }

  template <typename T>
  void ComputeVxAttentionScore(T* output,                           // buffer for the result with size BxSxNxH
                               const T* attention_probs,            // Attention probs with size BxNxSxT
                               const T* V,                          // V value with size BxN_kvxSxH
                               const int32_t* seqlens_k,            // past sequence lengths tensor
                               int batch_size,                      // batch size
                               int sequence_length,                 // sequence length
                               int past_buffer_sequence_length,     // sequence length in past state
                               int present_buffer_sequence_length,  // sequence length in past state
                               int head_size,                       // head size of Q, K, V
                               int hidden_size,                     // hidden size of Output
                               const T* past_value,                 // past value only
                               T* present_value,                    // present value only
                               bool past_present_share_buffer,      // whether present key and value share the same buffer
                               bool packed_qkv,                     // whether Q, K, V are packed
                               ThreadPool* tp) const {
    const bool is_prompt = sequence_length != 1;
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const int kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const int kv_input_chunk_length = sequence_length * head_size;                                             // L x H
    const size_t past_buff_chunk_length = static_cast<size_t>(past_buffer_sequence_length) * head_size;        // L x H
    const size_t present_buff_chunk_length = static_cast<size_t>(present_buffer_sequence_length) * head_size;  // T x H

    if (!past_present_share_buffer) {
      memset(present_value, 0, batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
    }

    // The cost of Gemm
    TensorOpCost unit_cost;
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * present_buffer_sequence_length);
    unit_cost.bytes_loaded = static_cast<double>(SafeInt<ptrdiff_t>(sequence_length + head_size) *
                                                 present_buffer_sequence_length * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(sequence_length * head_size * sizeof(T));

    if (present_value) {
      double bytes_to_copy_value = static_cast<double>(present_buff_chunk_length * sizeof(T));
      unit_cost.bytes_loaded += bytes_to_copy_value;
      unit_cost.bytes_stored += bytes_to_copy_value;
    }

    const size_t bytes_to_copy_trans = SafeInt<size_t>(head_size) * sizeof(T);
    double bytes_to_copy_trans_all = static_cast<double>(sequence_length * bytes_to_copy_trans);
    unit_cost.bytes_loaded += bytes_to_copy_trans_all;
    unit_cost.bytes_stored += bytes_to_copy_trans_all;

    ThreadPool::TryParallelFor(
        tp, SafeInt<ptrdiff_t>(batch_size) * num_heads_, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          for (std::ptrdiff_t i = begin; i != end; ++i) {
            const int batch_index = static_cast<int>(i / num_heads_);
            const int head_index = static_cast<int>(i % num_heads_);
            const int past_seqlen =
                sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index]) : past_buffer_sequence_length;
            const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
            const int total_seqlen = seqlens_k[batch_index] + 1;

            const T* v;
            if (packed_qkv) {
              v = V + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
            } else {
              v = V + kv_input_chunk_length * (i / kv_num_heads_factor);
            }
            if (nullptr != present_value) {
              v = ConcatStateChunkGQA(past_value, v, present_value, present_buff_chunk_length, past_buff_chunk_length,
                                      past_chunk_length, kv_input_chunk_length, is_prompt, past_present_share_buffer,
                                      i / kv_num_heads_factor);
            }

            T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * i;

            math::GemmEx<T, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                        1.f, /*alpha*/
                                        attention_probs + attention_probs_offset, present_buffer_sequence_length, v,
                                        head_size, 0.0f /*beta*/, output_current, hidden_size, nullptr);
          }
        });
  }
};

}  // namespace contrib
}  // namespace onnxruntime
