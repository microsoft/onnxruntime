// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/attention_helper.h"

#include "core/common/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class GQAAttentionBase {
 protected:
  GQAAttentionBase(const OpKernelInfo& info, bool has_local) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
    kv_num_heads_ = static_cast<int>(kv_num_heads);

    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);

    do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

    use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;

    local_window_size_ = has_local ? static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1)) : -1;
  }

  int num_heads_;     // number of attention heads of Q
  int kv_num_heads_;  // number of attention heads of K or V
  float scale_;       // the scaling factor applied before softmax
  float softcap_;
  bool do_rotary_;  // whether or not to use rotary embeddings
  bool rotary_interleaved_;
  int local_window_size_;

  bool use_smooth_softmax_;

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
    const bool is_prompt = parameters.is_first_prompt;
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
    bool gqa_mlas_supported = MlasGQASupported<T>(CblasNoTrans, CblasTrans) &&
                              MlasGQASupported<T>(CblasNoTrans, CblasNoTrans);
    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * seqlen_present_kv_cache *
                   (gqa_mlas_supported ? sizeof(T) : sizeof(float));
    auto attention_probs = allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

    bool past_present_share_buffer = past_key_data == present_key_data && past_value_data == present_value_data;

    const T* k = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;

    if (gqa_mlas_supported) {
      ComputeAttentionProbs(static_cast<T*>(attention_probs), Q, k, seqlens_k->Data<int32_t>(), batch_size,
                            sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size, past_key_data,
                            present_key_data, past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);

      // Compute the attentionScore * Value: out(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
      const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;
      ComputeVxAttentionScore(output->MutableData<T>(), static_cast<T*>(attention_probs), v,
                              seqlens_k->Data<int32_t>(),
                              batch_size, sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size,
                              hidden_size, past_value_data, present_value_data, past_present_share_buffer, packed_qkv,
                              is_prompt, tp, allocator);
    } else {
      ComputeAttentionProbs(static_cast<float*>(attention_probs), Q, k, seqlens_k->Data<int32_t>(), batch_size,
                            sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size, past_key_data,
                            present_key_data, past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);

      // Compute the attentionScore * Value: out(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
      const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;
      ComputeVxAttentionScore(output->MutableData<T>(), static_cast<float*>(attention_probs), v,
                              seqlens_k->Data<int32_t>(),
                              batch_size, sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size,
                              hidden_size, past_value_data, present_value_data, past_present_share_buffer, packed_qkv,
                              is_prompt, tp, allocator);
    }

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  // If T is float32, U is float32. If T is float16, U could be float16 or float32.
  template <typename T, typename U>
  void ComputeAttentionProbs(U* attention_probs,                           // output buffer with size BxNxSxT
                             const T* Q,                                   // Q data. Its size is BxNxSxH
                             const T* K,                                   // k data. Its size is BxNxLxH
                             const int32_t* seqlens_k,                     // total - 1 sequence lengths tensor
                             const size_t batch_size,                      // batch size of self-attention
                             const size_t sequence_length,                 // sequence length of self-attention (S)
                             const size_t past_buffer_sequence_length,     // sequence length of past state
                             const size_t present_buffer_sequence_length,  // sequence length of present state
                             const size_t head_size,                       // head size of self-attention
                             const T* past_key,                            // past key only
                             T* present_key,                               // present key only
                             const bool past_present_share_buffer,         // whether present key and value share the same buffer
                             const bool packed_qkv,                        // whether Q, K, V are packed
                             const bool is_prompt,                         // whether it is prompt
                             ThreadPool* tp,                               // thread pool
                             AllocatorPtr allocator) const {               // allocator for temporary buffer
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t q_input_chunk_length = sequence_length * head_size;                      // S x H
    const size_t kv_input_chunk_length = sequence_length * head_size;                     // L x H
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;        // L x H
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;  // T x H

    if (!past_present_share_buffer) {
      memset((void*)present_key,
             0,
             batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
    }

    const size_t loop_len = batch_size * num_heads_;
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
        const size_t batch_index = i / num_heads_;
        const size_t head_index = i % num_heads_;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;  // Assume no padding sequence length
        const size_t past_chunk_length = past_seqlen * head_size;

        const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * sequence_length * present_buffer_sequence_length;
        U* output = attention_probs + output_offset;

        const T* k;
        if (packed_qkv) {
          k = K + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          k = K + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_key) {
          k = ConcatStateChunkGQA(past_key, k, present_key, present_buff_chunk_length, past_buff_chunk_length,
                                  past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
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

        if constexpr (std::is_same<T, float>::value) {
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q,
                                          static_cast<int>(head_size), k, static_cast<int>(head_size), 0.0f /*bata*/,
                                          output, static_cast<int>(present_buffer_sequence_length), nullptr);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          MlasGemm(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size,
                   q, static_cast<int>(head_size), k, static_cast<int>(head_size), output,
                   static_cast<int>(present_buffer_sequence_length),
                   MLFloat16(alpha).val, static_cast<uint16_t>(0) /*beta*/, nullptr);
        } else {
          size_t bytes = head_size * (sequence_length + total_seqlen) * sizeof(float);
          auto q_k_fp32 = allocator->Alloc(bytes);
          BufferUniquePtr scratch_buffer(q_k_fp32, BufferDeleter(allocator));

          float* q_fp32 = static_cast<float*>(q_k_fp32);
          MlasConvertHalfToFloatBuffer(q, q_fp32, head_size * sequence_length);

          float* k_fp32 = q_fp32 + head_size * sequence_length;
          MlasConvertHalfToFloatBuffer(k, k_fp32, head_size * total_seqlen);

          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q_fp32,
                                          static_cast<int>(head_size), k_fp32, static_cast<int>(head_size), 0.0f /*bata*/,
                                          output, static_cast<int>(present_buffer_sequence_length), nullptr);
        }

        // compute Softmax
        U* output_softmax = output;
        for (size_t seq = 0; seq < sequence_length; seq++) {
          size_t seq_causal_length = past_seqlen + seq + 1;
          if (local_window_size_ > 0 && seq_causal_length > static_cast<size_t>(local_window_size_) + 1) {
            for (size_t total_seq_id = 0; total_seq_id < seq_causal_length - local_window_size_ - 1; total_seq_id++) {
              if constexpr (std::is_same<U, float>::value) {
                output_softmax[total_seq_id] = 0.f;
              } else {
                output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
              }
            }
            if (softcap_ > 0.f) {
              ComputeAttentionSoftcapInplace(output_softmax + seq_causal_length - local_window_size_ - 1,
                                             local_window_size_ + 1, static_cast<U>(softcap_));
            }
            if (use_smooth_softmax_) {
              ComputeSmoothSoftmaxInplace(output_softmax + seq_causal_length - local_window_size_ - 1, 1,
                                          local_window_size_ + 1, nullptr);
            } else {
              ComputeAttentionSoftmaxInplace(output_softmax + seq_causal_length - local_window_size_ - 1, 1,
                                             local_window_size_ + 1, nullptr);
            }
          } else {
            if (softcap_ > 0.f) {
              ComputeAttentionSoftcapInplace(output_softmax, static_cast<int>(seq_causal_length),
                                             static_cast<U>(softcap_));
            }
            if (use_smooth_softmax_) {
              ComputeSmoothSoftmaxInplace(output_softmax, 1, static_cast<int>(seq_causal_length), nullptr);
            } else {
              ComputeAttentionSoftmaxInplace(output_softmax, 1, static_cast<int>(seq_causal_length), nullptr);
            }
          }

          // set causal [seq_causal_length, total_seqlen) to 0.f
          for (size_t total_seq_id = seq_causal_length; total_seq_id < total_seqlen; total_seq_id++) {
            if constexpr (std::is_same<U, float>::value) {
              output_softmax[total_seq_id] = 0.f;
            } else {
              output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
            }
          }

          output_softmax += present_buffer_sequence_length;
        }
      }
    });
  }

  template <typename T, typename U>
  void ComputeVxAttentionScore(T* output,                                    // buffer for the result with size BxSxNxH
                               const U* attention_probs,                     // Attention probs with size BxNxSxT
                               const T* V,                                   // V value with size BxN_kvxSxH
                               const int32_t* seqlens_k,                     // total - 1 sequence lengths tensor
                               const size_t batch_size,                      // batch size
                               const size_t sequence_length,                 // sequence length
                               const size_t past_buffer_sequence_length,     // sequence length in past state
                               const size_t present_buffer_sequence_length,  // sequence length in past state
                               const size_t head_size,                       // head size of Q, K, V
                               const size_t hidden_size,                     // hidden size of Output
                               const T* past_value,                          // past value only
                               T* present_value,                             // present value only
                               const bool past_present_share_buffer,         // whether present key and value share the same buffer
                               const bool packed_qkv,                        // whether Q, K, V are packed
                               const bool is_prompt,                         // whether it is prompt
                               ThreadPool* tp,
                               AllocatorPtr allocator) const {
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t kv_input_chunk_length = sequence_length * head_size;                     // L x H
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;        // L x H
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;  // T x H

    if (!past_present_share_buffer) {
      memset((void*)present_value,
             0,
             batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
    }

    const size_t loop_len = batch_size * num_heads_;

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

    size_t output_fp32_bytes = 0;
    if constexpr (std::is_same<T, MLFloat16>::value && std::is_same<U, float>::value) {
      output_fp32_bytes = SafeInt<size_t>(sequence_length) * batch_size * num_heads_ * head_size * sizeof(float);
    }
    auto output_fp32 = allocator->Alloc(output_fp32_bytes);
    BufferUniquePtr scratch_buffer(output_fp32, BufferDeleter(allocator));

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const size_t batch_index = i / num_heads_;
        const size_t head_index = i % num_heads_;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;  // Assume no padding sequence length
        const size_t past_chunk_length = past_seqlen * head_size;

        const T* v;
        if (packed_qkv) {
          v = V + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          v = V + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_value) {
          v = ConcatStateChunkGQA(past_value, v, present_value, present_buff_chunk_length, past_buff_chunk_length,
                                  past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
                                  i / kv_num_heads_factor);
        }

        ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * i;

        if constexpr (std::is_same<T, float>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                          1.f, /*alpha*/ attention_probs + attention_probs_offset,
                                          static_cast<int>(present_buffer_sequence_length), v,
                                          static_cast<int>(head_size), 0.0f /*beta*/, output_current,
                                          static_cast<int>(hidden_size), nullptr);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          MlasGemm(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                   attention_probs + attention_probs_offset, static_cast<int>(present_buffer_sequence_length),
                   v, static_cast<int>(head_size), output_current, static_cast<int>(hidden_size),
                   MLFloat16(1.0f).val, static_cast<uint16_t>(0) /*beta*/, nullptr);
        } else {
          size_t bytes = head_size * total_seqlen * sizeof(float);
          auto v_fp32 = allocator->Alloc(bytes);
          BufferUniquePtr scratch_buffer(v_fp32, BufferDeleter(allocator));

          float* v_fp32_ptr = static_cast<float*>(v_fp32);
          MlasConvertHalfToFloatBuffer(v, v_fp32_ptr, head_size * total_seqlen);

          float* output_fp32_current = static_cast<float*>(output_fp32) +
                                       (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                          1.f, /*alpha*/ attention_probs + attention_probs_offset,
                                          static_cast<int>(present_buffer_sequence_length), v_fp32_ptr,
                                          static_cast<int>(head_size), 0.0f /*beta*/, output_fp32_current,
                                          static_cast<int>(hidden_size), nullptr);
        }
      }
    });

    if constexpr (std::is_same<T, MLFloat16>::value && std::is_same<U, float>::value) {
      MlasConvertFloatToHalfBuffer(static_cast<float*>(output_fp32),
                                   output,
                                   SafeInt<size_t>(sequence_length) * batch_size * num_heads_ * head_size);
    }
  }
};

}  // namespace contrib
}  // namespace onnxruntime
