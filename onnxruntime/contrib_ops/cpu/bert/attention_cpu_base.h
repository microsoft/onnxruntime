// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/attention_helper.h"
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"

namespace onnxruntime {
namespace contrib {

class AttentionCPUBase : public AttentionBase {
 protected:
  AttentionCPUBase(const OpKernelInfo& info, bool require_same_hidden_size)
      : AttentionBase(info, require_same_hidden_size) {}

  template <typename T>
  Status ApplyAttention(const T* Q,                // Q data with shape BxNxSxH
                        const T* K,                // K data with shape BxNxLxH
                        const T* V,                // V value with size BxNxLxH_v
                        const Tensor* mask_index,  // mask index. nullptr if no mask or its size is B
                        const Tensor* past,        // past state
                        const Tensor* past_key,    // past K input tensor (if not using past state)
                        const Tensor* past_value,  // past V input tensor (if not using past state)
                        Tensor* output,            // output tensor
                        Tensor* present_key,       // present K output tensor (if separating present KV)
                        Tensor* present_value,     // present V output tensor (if separating present KV)
                        int batch_size,            // batch size (B)
                        int sequence_length,       // sequence length of Q (S)
                        int kv_sequence_length,    // sequence length of K or V (L)
                        int qk_head_size,          // head size of Q or K (H)
                        int v_head_size,           // head size of V (H_v)
                        int v_hidden_size,         // hidden size of V (D_v)
                        const Tensor* attn_bias,   // additive bias applied on scaled QK.
                        OpKernelContext* context) const {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

    auto* tp = context->GetOperatorThreadPool();

    int past_sequence_length = 0;
    Tensor* present = nullptr;
    if (present_key == nullptr && present_value == nullptr) {
      present = GetPresent(context, past, batch_size, v_head_size, kv_sequence_length, past_sequence_length);
    } else if (past_key != nullptr && past_value != nullptr) {
      past_sequence_length = static_cast<int>(past_key->Shape().GetDims()[2]);
    }

    // Total sequence length including that of past state: T = P + L
    const int total_sequence_length = past_sequence_length + kv_sequence_length;

    // Merge causal mask with padding mask, and convert values from 0/1 to -inf/0, then broadcast to 3D (BxSxT).
    bool causal = (is_unidirectional_ && sequence_length > 1);
    void* mask_data = nullptr;
    if (mask_index != nullptr || causal) {
      size_t mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * total_sequence_length * sizeof(T);
      mask_data = allocator->Alloc(mask_data_bytes);
      memset(mask_data, 0, mask_data_bytes);
    }
    BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));
    const int32_t* mask_index_data = mask_index != nullptr ? mask_index->Data<int32_t>() : nullptr;
    gsl::span<const int64_t> mask_index_dims = mask_index != nullptr
                                                   ? mask_index->Shape().GetDims()
                                                   : gsl::span<const int64_t>{};
    DUMP_CPU_TENSOR_INIT();
    DUMP_CPU_TENSOR("Mask", mask_index_data, mask_index_dims);

    if (mask_data != nullptr) {
      // Convert mask from boolean (0/1) to float (mask_filter_value/0.0f).
      // Merge padding mask with causual mask, and broadcast to 3D (BxSxT).
      PrepareMask(mask_index_data, mask_index_dims, static_cast<T*>(mask_data),
                  causal, batch_size, sequence_length, past_sequence_length, mask_filter_value_);
      DUMP_CPU_TENSOR("Mask3D", static_cast<T*>(mask_data), batch_size, sequence_length, total_sequence_length);
    }

    float scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(qk_head_size)) : scale_;

    const T* past_data = past != nullptr ? past->Data<T>() : nullptr;
    T* present_data = present != nullptr ? present->MutableData<T>() : nullptr;
    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

    const T* attn_bias_data = (attn_bias != nullptr) ? attn_bias->Data<T>() : nullptr;
    auto attn_bias_dims = (attn_bias != nullptr) ? attn_bias->Shape().GetDims() : gsl::span<const int64_t>{};

    // Compute the attention score.
    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * total_sequence_length * sizeof(T);
    auto attention_probs = allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));
    ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), Q, K,
                             static_cast<T*>(mask_data),
                             batch_size, sequence_length, kv_sequence_length, past_sequence_length,
                             qk_head_size == 0 ? v_head_size : qk_head_size, past_data, past_key_data,
                             present_data, present_key_data, tp, scale, attn_bias_data, attn_bias_dims);

    // Compute the attentionScore * Value: out_tmp(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
    auto out_tmp_data =
        allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * v_head_size * sizeof(T));
    BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(std::move(allocator)));

    ComputeVxAttentionScore(output->MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(attention_probs),
                            V, batch_size, sequence_length, kv_sequence_length, past_sequence_length, v_head_size,
                            v_hidden_size, past_data, past_value_data, present_data, present_value_data, tp);

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T) +
  //                                1 x mask_data(B, N, S, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  template <typename T>
  void ComputeAttentionProbs(T* attention_probs,                      // output buffer with size BxNxSxT
                             const T* Q,                              // Q data. Its size is BxNxSxH
                             const T* K,                              // k data. Its size is BxNxLxH
                             T* mask_data,                            // buffer for mask data.
                             int batch_size,                          // batch size of self-attention
                             int sequence_length,                     // sequence length of self-attention (S)
                             int kv_sequence_length,                  // sequence length of cross-attention (L)
                             int past_sequence_length,                // sequence length of past state
                             int head_size,                           // head size of self-attention
                             const T* past,                           // past state
                             const T* past_key,                       // past key only (if not using past state)
                             T* present,                              // present state
                             T* present_key,                          // present key only (if not using present state)
                             ThreadPool* tp,                          // thread pool
                             float scale,                             // scale factor
                             const T* attn_bias_data,                 // attention bias
                             gsl::span<const int64_t> attn_bias_dims  // attention bias shape
  ) const {
    const int total_sequence_length = past_sequence_length + kv_sequence_length;               // T = P + L
    const size_t past_chunk_length = static_cast<size_t>(past_sequence_length) * head_size;    // P x H
    const size_t q_input_chunk_length = static_cast<size_t>(sequence_length) * head_size;      // S x H
    const size_t kv_input_chunk_length = static_cast<size_t>(kv_sequence_length) * head_size;  // L x H
    const size_t present_chunk_length = past_chunk_length + kv_input_chunk_length;             // T x H

    DUMP_CPU_TENSOR_INIT();
    DUMP_CPU_TENSOR("Q", Q, batch_size, num_heads_, sequence_length, head_size);
    DUMP_CPU_TENSOR("K", K, batch_size, num_heads_, total_sequence_length, head_size);
    DUMP_CPU_TENSOR("Attn_Bias", attn_bias_data, attn_bias_dims);

    {
      const int loop_len = batch_size * num_heads_;
      const float alpha = scale;

      TensorOpCost unit_cost;
      const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length;
      const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);
      unit_cost.compute_cycles =
          static_cast<double>(SafeInt<ptrdiff_t>(2) * head_size * probs_matrix_size);
      unit_cost.bytes_loaded = static_cast<double>((sequence_length + total_sequence_length) * head_size * sizeof(T));
      unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

      if (mask_data != nullptr) {
        unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes);
        unit_cost.bytes_stored += static_cast<double>(probs_matrix_bytes);
      }

      if (present || present_key) {
        double bytes_to_copy_key = static_cast<double>(sizeof(T) * present_chunk_length);
        unit_cost.bytes_loaded += bytes_to_copy_key;
        unit_cost.bytes_stored += bytes_to_copy_key;
      }

      if (attn_bias_data != nullptr) {
        unit_cost.compute_cycles += static_cast<double>(probs_matrix_size);
        unit_cost.bytes_loaded += probs_matrix_bytes * 2;
        unit_cost.bytes_stored += probs_matrix_bytes;
      }

      ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          const int batch_index = static_cast<int>(i) / num_heads_;
          const std::ptrdiff_t head_index = i % static_cast<std::ptrdiff_t>(num_heads_);

          const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
          const ptrdiff_t mask_offset = SafeInt<ptrdiff_t>(batch_index) * probs_matrix_size;

          T* output = attention_probs + output_offset;

          if (attn_bias_data != nullptr) {
            // Attention bias has shape (B or 1, N or 1, S, T)
            // Here we handle the broadcast of batch_size and num_heads dimensions.
            ptrdiff_t attn_bias_offset = 0;
            if (attn_bias_dims[0] != 1) {
              attn_bias_offset += SafeInt<ptrdiff_t>(batch_index) * num_heads_ * probs_matrix_size;
            }
            if (attn_bias_dims[1] != 1) {
              attn_bias_offset += head_index * probs_matrix_size;
            }

            memcpy(output, attn_bias_data + attn_bias_offset, probs_matrix_bytes);

            if (mask_data != nullptr) {
              // This can be optimized with vectorized add using MlasAddFloat32x4.
              for (ptrdiff_t j = 0; j < probs_matrix_size; j++) {
                output[j] += mask_data[mask_offset + j];
              }
            }
          } else if (mask_data != nullptr) {
            // Broadcast mask data: (Bx)SxT -> (BxNx)SxT
            memcpy(output, mask_data + mask_offset, probs_matrix_bytes);
          }

          const T* k = K + kv_input_chunk_length * i;
          if (nullptr != present) {
            // Concatenate past_K and K : (BxNx)PxH, (BxNx)LxH -> (BxNx)TxH
            k = ConcatStateChunk(past, k, present, past_chunk_length, present_chunk_length, i);
          } else if (nullptr != present_key) {
            k = ConcatStateChunk(past_key, k, present_key, past_chunk_length, present_chunk_length, i);
          }

          // Compute Q*K' + AttentionMask
          //                     original                 transposed             each iteration
          // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
          // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
          // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
          math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_sequence_length, head_size, alpha,
                                    Q + q_input_chunk_length * i, k,
                                    (mask_data != nullptr || attn_bias_data != nullptr) ? 1.0f : 0.0f,
                                    output, nullptr);
        }
      });
    }

    DUMP_CPU_TENSOR("QK (scaled)", attention_probs, batch_size, num_heads_, sequence_length, total_sequence_length);

    // attention_probs(B, N, S, T) = Softmax(attention_probs)
    {
      const int N = batch_size * num_heads_ * sequence_length;
      const int D = total_sequence_length;
      ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);
    }

    DUMP_CPU_TENSOR("Softmax(QK)", attention_probs, batch_size, num_heads_, sequence_length, total_sequence_length);
  }

  template <typename T>
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
                               const T* past,             // past state
                               const T* past_value,       // past value only (if not using past state)
                               T* present,                // present state
                               T* present_value,          // present value only (if not using present state)
                               ThreadPool* tp) const {
    const int total_sequence_length = past_sequence_length + kv_sequence_length;                   // T = P + L
    const ptrdiff_t past_chunk_length = SafeInt<ptrdiff_t>(past_sequence_length) * v_head_size;    // P x H_v
    const ptrdiff_t q_input_chunk_length = SafeInt<ptrdiff_t>(sequence_length) * v_head_size;      // S x H_v
    const ptrdiff_t kv_input_chunk_length = SafeInt<ptrdiff_t>(kv_sequence_length) * v_head_size;  // L x H_v
    const ptrdiff_t present_chunk_length = past_chunk_length + kv_input_chunk_length;              // T x H_v

    // Move the pointer of past and present to start of v values.
    if (nullptr != past) {
      past += SafeInt<ptrdiff_t>(batch_size) * num_heads_ * past_sequence_length * v_head_size;
    }
    if (nullptr != present) {
      present += SafeInt<ptrdiff_t>(batch_size) * num_heads_ * total_sequence_length * v_head_size;
    }

    // The cost of Gemm
    TensorOpCost unit_cost;
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * v_head_size * total_sequence_length);
    unit_cost.bytes_loaded =
        static_cast<double>(SafeInt<ptrdiff_t>(sequence_length + v_head_size) * total_sequence_length * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(sequence_length * v_head_size * sizeof(T));

    if (present || present_value) {
      double bytes_to_copy_value = static_cast<double>(present_chunk_length * sizeof(T));
      unit_cost.bytes_loaded += bytes_to_copy_value;
      unit_cost.bytes_stored += bytes_to_copy_value;
    }

    const size_t bytes_to_copy_trans = SafeInt<size_t>(v_head_size) * sizeof(T);
    double bytes_to_copy_trans_all = static_cast<double>(sequence_length * bytes_to_copy_trans);
    unit_cost.bytes_loaded += bytes_to_copy_trans_all;
    unit_cost.bytes_stored += bytes_to_copy_trans_all;

    ThreadPool::TryParallelFor(
        tp, SafeInt<ptrdiff_t>(batch_size) * num_heads_, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          for (std::ptrdiff_t i = begin; i != end; ++i) {
            const T* v = V + kv_input_chunk_length * i;
            if (nullptr != present) {
              // Concatenate past_V and V: (BxNx)PxH_v, (BxNx)LxH_v -> (BxNx)TxH_v
              v = ConcatStateChunk(past, v, present, past_chunk_length, present_chunk_length, i);
            } else if (nullptr != present_value) {
              v = ConcatStateChunk(past_value, v, present_value, past_chunk_length, present_chunk_length, i);
            }

            T* current_tmp_data = reinterpret_cast<T*>(tmp_buffer) + q_input_chunk_length * i;
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
            math::MatMul<T>(sequence_length, v_head_size, total_sequence_length,
                            attention_probs + attention_probs_offset, v, current_tmp_data, nullptr);

            // Transpose: out(B, S, N, H_v) -> out_tmp(B, N, S, H_v)
            const int batch_index = static_cast<int>(i / num_heads_);
            const int head_index = static_cast<int>(i % num_heads_);
            T* src = current_tmp_data;
            ptrdiff_t dest_offset =
                (SafeInt<ptrdiff_t>(batch_index) * sequence_length * num_heads_ + head_index) * v_head_size;
            T* dest = output + dest_offset;
            for (int j = 0; j < sequence_length; j++) {
              memcpy(dest, src, bytes_to_copy_trans);
              src += v_head_size;
              dest += v_hidden_size;
            }
          }
        });
  }
};

}  // namespace contrib
}  // namespace onnxruntime
