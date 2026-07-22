// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <new>
#include <type_traits>

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
      : AttentionBase(info, require_same_hidden_size) {
  }

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
                        Tensor* output_qk,         // Q*K output tensor (if returning Q*K value)
                        int batch_size,            // batch size (B)
                        int sequence_length,       // sequence length of Q (S)
                        int kv_sequence_length,    // sequence length of K or V (L)
                        int qk_head_size,          // head size of Q or K (H)
                        int v_head_size,           // head size of V (H_v)
                        int v_hidden_size,         // hidden size of V (D_v)
                        const Tensor* attn_bias,   // additive bias applied on scaled QK.
                        OpKernelContext* context,
                        int past_sequence_length = 0,  // sequence length of past state
                        bool past_present_share_buffer = false) const {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

    auto* tp = context->GetOperatorThreadPool();

    Tensor* present = nullptr;
    if (past_sequence_length == 0) {
      if (present_key == nullptr && present_value == nullptr) {
        present = GetPresent(context, past, batch_size, v_head_size, kv_sequence_length, past_sequence_length);
      } else if (past_key != nullptr && past_value != nullptr) {
        past_sequence_length = static_cast<int>(past_key->Shape().GetDims()[2]);
      }
    }

    // Total sequence length including that of past state: T = P + L
    const int total_sequence_length = past_sequence_length + kv_sequence_length;
    if (total_sequence_length == 0) {
      const size_t output_bytes =
          SafeInt<size_t>(batch_size) * sequence_length * v_hidden_size * sizeof(T);
      if (output_bytes > 0) {
        memset(output->MutableData<T>(), 0, output_bytes);
      }
      return Status::OK();
    }

    // Merge causal mask with padding mask, and convert values from 0/1 to -inf/0.
    bool causal = (is_unidirectional_ && sequence_length > 1);

    const int32_t* mask_index_data = mask_index != nullptr ? mask_index->Data<int32_t>() : nullptr;
    gsl::span<const int64_t> mask_index_dims = mask_index != nullptr
                                                   ? mask_index->Shape().GetDims()
                                                   : gsl::span<const int64_t>{};

    // A pure padding mask (1D mask_index of shape (B) or (2B), or a 2D raw key mask of shape (B, T)),
    // without a causal mask, does not depend on the query position S. Store it as [B, T] and broadcast
    // across S during softmax (mask_seq_stride = 0) instead of materializing the full [B, S, T] mask.
    const bool padding_mask_only = (mask_index_data != nullptr) && !causal &&
                                   (mask_index_dims.size() == 1 || mask_index_dims.size() == 2);

    void* mask_data = nullptr;
    ptrdiff_t mask_batch_stride = 0;  // stride (in elements) between consecutive batches in mask_data
    ptrdiff_t mask_seq_stride = 0;    // stride (in elements) between consecutive query positions (0 broadcasts)
    if (mask_index != nullptr || causal) {
      const size_t mask_rows = padding_mask_only ? SafeInt<size_t>(batch_size)
                                                 : SafeInt<size_t>(batch_size) * sequence_length;
      const size_t mask_data_bytes = SafeInt<size_t>(mask_rows) * total_sequence_length * sizeof(T);
      if (mask_data_bytes > 0) {
        mask_data = allocator->Alloc(mask_data_bytes);
        memset(mask_data, 0, mask_data_bytes);
      }

      mask_batch_stride = padding_mask_only
                              ? static_cast<ptrdiff_t>(total_sequence_length)
                              : static_cast<ptrdiff_t>(SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length);
      mask_seq_stride = padding_mask_only ? static_cast<ptrdiff_t>(0) : static_cast<ptrdiff_t>(total_sequence_length);
    }
    BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));

    DUMP_CPU_TENSOR_INIT();
    DUMP_CPU_TENSOR("Mask", mask_index_data, mask_index_dims);

    if (mask_data != nullptr) {
      // Convert mask from boolean (0/1) to float (mask_filter_value/0.0f).
      if (padding_mask_only) {
        PreparePaddingMask(mask_index_data, mask_index_dims, static_cast<T*>(mask_data),
                           batch_size, kv_sequence_length, past_sequence_length, mask_filter_value_);
        DUMP_CPU_TENSOR("Mask2D", static_cast<T*>(mask_data), batch_size, total_sequence_length);
      } else {
        // Merge padding mask with causal mask, and broadcast to 3D (BxSxT).
        PrepareMask(mask_index_data, mask_index_dims, static_cast<T*>(mask_data),
                    causal, batch_size, sequence_length, kv_sequence_length, past_sequence_length, mask_filter_value_);
        DUMP_CPU_TENSOR("Mask3D", static_cast<T*>(mask_data), batch_size, sequence_length, total_sequence_length);
      }
    }

    float scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(qk_head_size)) : scale_;

    const T* past_data = past != nullptr ? past->Data<T>() : nullptr;
    T* present_data = present != nullptr ? present->MutableData<T>() : nullptr;
    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;
    T* output_qk_data = output_qk != nullptr ? output_qk->MutableData<T>() : nullptr;

    const T* attn_bias_data = (attn_bias != nullptr) ? attn_bias->Data<T>() : nullptr;
    auto attn_bias_dims = (attn_bias != nullptr) ? attn_bias->Shape().GetDims() : gsl::span<const int64_t>{};

    // Used for DecoderMaskedMultiHeadAttention
    int max_sequence_length = 0;
    if (past_present_share_buffer) {
      ORT_ENFORCE(past_key != nullptr && past_value != nullptr);
      max_sequence_length = static_cast<int>(past_key->Shape().GetDims()[2]);
    }

    // Compute the attention score.
    const ptrdiff_t batch_head_count = SafeInt<ptrdiff_t>(batch_size) * num_heads_;
    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * total_sequence_length * sizeof(T);
    auto attention_probs = bytes == 0 ? nullptr : allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));
    const size_t gemm_data_bytes = SafeInt<size_t>(batch_head_count) * sizeof(MLAS_SGEMM_DATA_PARAMS);
    auto* gemm_data = gemm_data_bytes == 0
                          ? nullptr
                          : static_cast<MLAS_SGEMM_DATA_PARAMS*>(allocator->Alloc(gemm_data_bytes));
    BufferUniquePtr gemm_data_buffer(gemm_data, BufferDeleter(allocator));
    ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), gemm_data, Q, K,
                             static_cast<T*>(mask_data), mask_batch_stride, mask_seq_stride,
                             batch_size, batch_head_count, sequence_length, kv_sequence_length, past_sequence_length,
                             qk_head_size == 0 ? v_head_size : qk_head_size, past_data, past_key_data, present_data,
                             present_key_data, output_qk_data, tp, scale, attn_bias_data, attn_bias_dims,
                             past_present_share_buffer, max_sequence_length);

    // Compute the attentionScore * Value: out_tmp(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
    const size_t out_tmp_bytes =
        SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * v_head_size * sizeof(T);
    auto out_tmp_data = out_tmp_bytes == 0 ? nullptr : allocator->Alloc(out_tmp_bytes);
    BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(std::move(allocator)));

    ComputeVxAttentionScore(output->MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(attention_probs),
                            V, batch_size, sequence_length, kv_sequence_length, past_sequence_length, v_head_size,
                            v_hidden_size, past_data, past_value_data, present_data, present_value_data, tp,
                            past_present_share_buffer, max_sequence_length);

    return Status::OK();
  }

  // For DecoderMaskedMultiHeadAttention
  template <typename T>
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
                                 Tensor* output_qk) const {
    ORT_RETURN_IF_ERROR(ValidateCacheIndirectionValues(cache_indir->Data<int32_t>(), batch_size, beam_width,
                                                       past_sequence_length, max_sequence_length));

    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

    auto* tp = context->GetOperatorThreadPool();

    int total_sequence_length = past_sequence_length + 1;  // This is +1 because this is used during token generation via DecoderMaskedMultiHeadAttention
    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * total_sequence_length * sizeof(T);
    auto attention_probs = allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;
    T* output_qk_data = (output_qk != nullptr) ? output_qk->MutableData<T>() : nullptr;

    const int32_t* mask_index_data = mask_index != nullptr ? mask_index->Data<int32_t>() : nullptr;
    const T* attn_bias_data = attn_bias != nullptr ? attn_bias->Data<T>() : nullptr;

    ComputeAttentionProbsWithBeams(static_cast<T*>(attention_probs), Q, K, mask_index_data, batch_size,
                                   past_sequence_length, max_sequence_length, head_size, past_key_data,
                                   present_key_data, tp, attn_bias_data, broadcast_attn_bias_dim_0,
                                   broadcast_attn_bias_dim_1, cache_indir->Data<int32_t>(), beam_width, output_qk_data);

    // Compute the attentionScore * Value: out_tmp(B, N, 1, H_v) = attention_probs(B, N, 1, T) x V(B, N, T, H_v)
    auto out_tmp_data = allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * v_head_size * sizeof(T));
    BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(std::move(allocator)));

    ComputeVxAttentionScoreWithBeams(output->MutableData<T>(), static_cast<T*>(out_tmp_data),
                                     static_cast<const T*>(attention_probs), V, batch_size,
                                     past_sequence_length, max_sequence_length, v_head_size, past_value_data,
                                     present_value_data, cache_indir->Data<int32_t>(), beam_width, tp);

    return Status::OK();
  }

 private:
  static Status ValidateCacheIndirectionValues(const int32_t* cache_indirection_data,
                                               int batch_beam_size,
                                               int beam_width,
                                               int past_sequence_length,
                                               int max_sequence_length) {
    if (cache_indirection_data == nullptr || beam_width <= 0 || past_sequence_length <= 0) {
      return Status::OK();
    }

    for (int batch_beam_index = 0; batch_beam_index < batch_beam_size; ++batch_beam_index) {
      const int32_t* beam_indices = cache_indirection_data +
                                    static_cast<std::ptrdiff_t>(batch_beam_index) * max_sequence_length;
      for (int position = 0; position < past_sequence_length; ++position) {
        const int32_t beam_index = beam_indices[position];
        if (beam_index < 0 || beam_index >= beam_width) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "cache_indirection beam index out of range. Expected [0, ", beam_width,
                                 "), got ", beam_index,
                                 " at flattened batch_beam index ", batch_beam_index,
                                 ", sequence position ", position);
        }
      }
    }

    return Status::OK();
  }

  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T) +
  //                                1 x mask_data(B, N, S, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  template <typename T>
  void ComputeAttentionProbs(T* attention_probs,                       // output buffer with size BxNxSxT
                             MLAS_SGEMM_DATA_PARAMS* gemm_data,        // scratch array with batch_size*num_heads entries
                             const T* Q,                               // Q data. Its size is BxNxSxH
                             const T* K,                               // k data. Its size is BxNxLxH
                             T* mask_data,                             // buffer for mask data.
                             ptrdiff_t mask_batch_stride,              // element stride between batches in mask_data
                             ptrdiff_t mask_seq_stride,                // element stride between query positions (0 broadcasts)
                             int batch_size,                           // batch size of self-attention
                             ptrdiff_t batch_head_count,               // batch_size * num_heads_
                             int sequence_length,                      // sequence length of self-attention (S)
                             int kv_sequence_length,                   // sequence length of cross-attention (L)
                             int past_sequence_length,                 // sequence length of past state
                             int head_size,                            // head size of self-attention
                             const T* past,                            // past state
                             const T* past_key,                        // past key only (if not using past state)
                             T* present,                               // present state
                             T* present_key,                           // present key only (if not using present state)
                             T* output_qk,                             // Q*K output
                             ThreadPool* tp,                           // thread pool
                             float scale,                              // scale factor
                             const T* attn_bias_data,                  // attention bias
                             gsl::span<const int64_t> attn_bias_dims,  // attention bias shape
                             bool past_present_share_buffer = false,
                             int max_sequence_length = 0) const {
    const int total_sequence_length = past_sequence_length + kv_sequence_length;               // T = P + L
    const size_t past_chunk_length = static_cast<size_t>(past_sequence_length) * head_size;    // P x H
    const size_t q_input_chunk_length = static_cast<size_t>(sequence_length) * head_size;      // S x H
    const size_t kv_input_chunk_length = static_cast<size_t>(kv_sequence_length) * head_size;  // L x H
    const size_t present_chunk_length = past_chunk_length + kv_input_chunk_length;             // T x H
    const size_t cache_chunk_length = static_cast<size_t>(max_sequence_length) * head_size;    // M x H

    DUMP_CPU_TENSOR_INIT();
    DUMP_CPU_TENSOR("Q", Q, batch_size, num_heads_, sequence_length, head_size);
    DUMP_CPU_TENSOR("K", K, batch_size, num_heads_, total_sequence_length, head_size);
    DUMP_CPU_TENSOR("Attn_Bias", attn_bias_data, attn_bias_dims);

    const float alpha = scale;
    const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length;

    // Step 1: attention_probs(B, N, S, T) = alpha * Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
    // The scaled Q*K' is written with a clean (beta=0) store; the additive mask and attention bias
    // are fused into the softmax pass below.
    {
      if constexpr (std::is_same_v<T, float>) {
        // Issue all batch*num_heads matmuls as a single batched GEMM so MLAS can parallelize the
        // work across all threads (partitioning both the batch and the M dimension), instead of one
        // single-threaded GEMM per head which caps parallelism at batch*num_heads units.
        // Prepare the K pointer for each (batch, head). When there is past/present state this
        // concatenates past K and current K into the present buffer, which must complete before the
        // batched GEMM reads it.
        const bool needs_concat = (present != nullptr) || (present_key != nullptr);
        auto prepare_gemm_data = [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          for (std::ptrdiff_t i = begin; i < end; ++i) {
            const T* k = K + kv_input_chunk_length * i;
            if (nullptr != present) {
              // Concatenate past_K and K : (BxNx)PxH, (BxNx)LxH -> (BxNx)TxH
              k = ConcatStateChunk(past, k, present, past_chunk_length, present_chunk_length, i);
            } else if (nullptr != present_key) {
              if (past_present_share_buffer) {
                k = present_key + cache_chunk_length * i;
                memcpy(const_cast<T*>(k) + past_chunk_length, K + head_size * i, head_size * sizeof(T));
              } else {
                k = ConcatStateChunk(past_key, k, present_key, past_chunk_length, present_chunk_length, i);
              }
            }

            if (sequence_length == 0 || total_sequence_length == 0) {
              continue;
            }

            const ptrdiff_t probs_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
            auto& params = *new (&gemm_data[i]) MLAS_SGEMM_DATA_PARAMS{};
            params.A = reinterpret_cast<const float*>(Q + q_input_chunk_length * i);
            params.lda = static_cast<size_t>(head_size);
            params.B = reinterpret_cast<const float*>(k);
            params.ldb = static_cast<size_t>(head_size);
            params.C = reinterpret_cast<float*>(attention_probs + probs_offset);
            params.ldc = static_cast<size_t>(total_sequence_length);
            params.alpha = alpha;
            params.beta = 0.0f;
          }
        };

        if (needs_concat) {
          TensorOpCost prep_cost;
          const double concat_bytes = static_cast<double>(present_chunk_length * sizeof(T));
          prep_cost.compute_cycles = 0.0;
          prep_cost.bytes_loaded = concat_bytes;
          prep_cost.bytes_stored = concat_bytes;
          ThreadPool::TryParallelFor(tp, batch_head_count, prep_cost, prepare_gemm_data);
        } else {
          prepare_gemm_data(0, batch_head_count);
        }

        if (batch_head_count > 0 && sequence_length > 0 && total_sequence_length > 0) {
          MlasGemmBatch(CblasNoTrans, CblasTrans,
                        static_cast<size_t>(sequence_length),
                        static_cast<size_t>(total_sequence_length),
                        static_cast<size_t>(head_size),
                        gemm_data, static_cast<size_t>(batch_head_count), tp,
                        &mlas_backend_kernel_selector_config_);
        }
      } else {
        // Fallback for non-float T (not currently instantiated): per-head single-threaded GEMM.
        TensorOpCost unit_cost;
        unit_cost.compute_cycles = static_cast<double>(SafeInt<ptrdiff_t>(2) * head_size * probs_matrix_size);
        unit_cost.bytes_loaded = static_cast<double>((sequence_length + total_sequence_length) * head_size * sizeof(T));
        unit_cost.bytes_stored = static_cast<double>(probs_matrix_size * sizeof(T));
        if (present || present_key) {
          double bytes_to_copy_key = (past_present_share_buffer ? kv_input_chunk_length : present_chunk_length) *
                                     static_cast<double>(sizeof(T));
          unit_cost.bytes_loaded += bytes_to_copy_key;
          unit_cost.bytes_stored += bytes_to_copy_key;
        }

        ThreadPool::TryParallelFor(tp, batch_head_count, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          for (std::ptrdiff_t i = begin; i != end; ++i) {
            const ptrdiff_t probs_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
            T* output = attention_probs + probs_offset;

            const T* k = K + kv_input_chunk_length * i;
            if (nullptr != present) {
              k = ConcatStateChunk(past, k, present, past_chunk_length, present_chunk_length, i);
            } else if (nullptr != present_key) {
              if (past_present_share_buffer) {
                k = present_key + cache_chunk_length * i;
                memcpy(const_cast<T*>(k) + past_chunk_length, K + head_size * i, head_size * sizeof(T));
              } else {
                k = ConcatStateChunk(past_key, k, present_key, past_chunk_length, present_chunk_length, i);
              }
            }

            if (sequence_length > 0 && total_sequence_length > 0) {
              math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_sequence_length, head_size,
                                        alpha, Q + q_input_chunk_length * i, k, 0.0f, output, nullptr,
                                        &mlas_backend_kernel_selector_config_);
            }
          }
        });
      }
    }

    if (batch_head_count == 0 || sequence_length == 0 || total_sequence_length == 0) {
      return;
    }

    DUMP_CPU_TENSOR("QK (scaled)", attention_probs, batch_size, num_heads_, sequence_length, total_sequence_length);

    // Step 2: attention_probs(B, N, S, T) = Softmax(attention_probs + mask + attn_bias)
    if (mask_data == nullptr && attn_bias_data == nullptr && output_qk == nullptr) {
      // No mask, attention bias or QK output: softmax across all rows (MLAS parallelizes internally).
      const int N = batch_size * num_heads_ * sequence_length;
      ComputeAttentionSoftmaxInplace(attention_probs, N, total_sequence_length, tp);
    } else {
      // Fused pass over B*N*S rows: apply the additive mask and attention bias, optionally emit the
      // pre-softmax QK, then softmax the row - all while the score row is hot in cache. This avoids a
      // separate materialization/add pass over the [B, N, S, T] score buffer and runs across all
      // threads.
      const ptrdiff_t num_rows = SafeInt<ptrdiff_t>(batch_size) * num_heads_ * sequence_length;
      const int D = total_sequence_length;
      const int num_addends = (mask_data != nullptr ? 1 : 0) + (attn_bias_data != nullptr ? 1 : 0);

      TensorOpCost row_cost;
      const double row_bytes = static_cast<double>(D) * sizeof(T);
      row_cost.compute_cycles = static_cast<double>(D) * (num_addends + 6);  // adds + softmax exp/normalize
      row_cost.bytes_loaded = row_bytes * (1 + num_addends);
      row_cost.bytes_stored = row_bytes * (output_qk != nullptr ? 2 : 1);

      ThreadPool::TryParallelFor(tp, num_rows, row_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t r = begin; r != end; ++r) {
          const int s_i = static_cast<int>(r % sequence_length);
          const std::ptrdiff_t bn = r / sequence_length;  // batch_index * num_heads_ + head_index
          const int batch_index = static_cast<int>(bn / num_heads_);

          const ptrdiff_t row_offset = SafeInt<ptrdiff_t>(r) * D;
          T* row = attention_probs + row_offset;

          if (attn_bias_data != nullptr) {
            // Attention bias has shape (B or 1, N or 1, S, T). Handle the broadcast of the
            // batch_size and num_heads dimensions.
            const int head_index = static_cast<int>(bn % num_heads_);
            ptrdiff_t attn_bias_offset = 0;
            if (attn_bias_dims[0] != 1) {
              attn_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attn_bias_dims[1] * probs_matrix_size;
            }
            if (attn_bias_dims[1] != 1) {
              attn_bias_offset += SafeInt<ptrdiff_t>(head_index) * probs_matrix_size;
            }
            attn_bias_offset += SafeInt<ptrdiff_t>(s_i) * D;
            MlasEltwiseAdd<T>(row, attn_bias_data + attn_bias_offset, row, static_cast<size_t>(D));
          }

          if (mask_data != nullptr) {
            // Mask data is broadcast across the num_heads dimension. It is either the full [B, S, T]
            // mask (mask_seq_stride = T) or a [B, T] padding mask broadcast across S (mask_seq_stride = 0).
            const ptrdiff_t mask_offset = SafeInt<ptrdiff_t>(batch_index) * mask_batch_stride + s_i * mask_seq_stride;
            MlasEltwiseAdd<T>(row, mask_data + mask_offset, row, static_cast<size_t>(D));
          }

          if (output_qk != nullptr) {
            // Emit the scaled Q*K^T (including mask and attention bias) before softmax.
            memcpy(output_qk + row_offset, row, static_cast<size_t>(D) * sizeof(T));
          }

          ComputeAttentionSoftmaxInplace(row, 1, D, nullptr);
        }
      });
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
                               ThreadPool* tp,
                               bool past_present_share_buffer = false,
                               int max_sequence_length = 0) const {
    const int total_sequence_length = past_sequence_length + kv_sequence_length;                   // T = P + L
    const ptrdiff_t past_chunk_length = SafeInt<ptrdiff_t>(past_sequence_length) * v_head_size;    // P x H_v
    const ptrdiff_t q_input_chunk_length = SafeInt<ptrdiff_t>(sequence_length) * v_head_size;      // S x H_v
    const ptrdiff_t kv_input_chunk_length = SafeInt<ptrdiff_t>(kv_sequence_length) * v_head_size;  // L x H_v
    const ptrdiff_t present_chunk_length = past_chunk_length + kv_input_chunk_length;              // T x H_v
    const ptrdiff_t cache_chunk_length = SafeInt<ptrdiff_t>(max_sequence_length) * v_head_size;    // M x H_v

    // Move the pointer of past and present to start of v values.
    if (nullptr != past) {
      past += SafeInt<ptrdiff_t>(batch_size) * num_heads_ * past_sequence_length * v_head_size;
    }
    if (nullptr != present) {
      present += SafeInt<ptrdiff_t>(batch_size) * num_heads_ * total_sequence_length * v_head_size;
    }

    const ptrdiff_t batch_head_count = SafeInt<ptrdiff_t>(batch_size) * num_heads_;
    if (batch_head_count == 0 || (sequence_length == 0 && present == nullptr && present_value == nullptr)) {
      return;
    }

    // The cost of Gemm
    TensorOpCost unit_cost;
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * v_head_size * total_sequence_length);
    unit_cost.bytes_loaded =
        static_cast<double>(SafeInt<ptrdiff_t>(sequence_length + v_head_size) * total_sequence_length * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(sequence_length * v_head_size * sizeof(T));

    if (present || present_value) {
      double bytes_to_copy_value = (past_present_share_buffer ? kv_input_chunk_length : present_chunk_length) *
                                   static_cast<double>(sizeof(T));
      unit_cost.bytes_loaded += bytes_to_copy_value;
      unit_cost.bytes_stored += bytes_to_copy_value;
    }

    const size_t bytes_to_copy_trans = SafeInt<size_t>(v_head_size) * sizeof(T);
    double bytes_to_copy_trans_all = static_cast<double>(sequence_length * bytes_to_copy_trans);
    unit_cost.bytes_loaded += bytes_to_copy_trans_all;
    unit_cost.bytes_stored += bytes_to_copy_trans_all;

    ThreadPool::TryParallelFor(
        tp, batch_head_count, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          for (std::ptrdiff_t i = begin; i != end; ++i) {
            const T* v = V + kv_input_chunk_length * i;
            if (nullptr != present) {
              // Concatenate past_V and V: (BxNx)PxH_v, (BxNx)LxH_v -> (BxNx)TxH_v
              v = ConcatStateChunk(past, v, present, past_chunk_length, present_chunk_length, i);
            } else if (nullptr != present_value) {
              if (past_present_share_buffer) {
                v = present_value + cache_chunk_length * i;
                memcpy(const_cast<T*>(v) + past_chunk_length, V + v_head_size * i, v_head_size * sizeof(T));
              } else {
                v = ConcatStateChunk(past_value, v, present_value, past_chunk_length, present_chunk_length, i);
              }
            }

            if (sequence_length == 0) {
              continue;
            }

            T* current_tmp_data = reinterpret_cast<T*>(tmp_buffer) + q_input_chunk_length * i;
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
            math::MatMul<T>(sequence_length, v_head_size, total_sequence_length,
                            attention_probs + attention_probs_offset, v, current_tmp_data, nullptr, &mlas_backend_kernel_selector_config_);

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

  // Used for DecoderMaskedMultiHeadAttention where sequence_length = 1
  template <typename T>
  void ComputeAttentionProbsWithBeams(T* attention_probs,
                                      const T* Q,
                                      const T* K,
                                      const int32_t* mask_index_data,
                                      int batch_size,
                                      int past_sequence_length,
                                      int max_sequence_length,
                                      int head_size,
                                      const T* past_key_data,
                                      T* present_key_data,
                                      ThreadPool* tp,
                                      const T* attn_bias_data,
                                      bool broadcast_attn_bias_dim_0,
                                      bool broadcast_attn_bias_dim_1,
                                      const int32_t* cache_indir_data,
                                      int beam_width,
                                      T* output_qk_data) const {
    float scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    TensorOpCost unit_cost;
    auto total_sequence_length = past_sequence_length + 1;
    const ptrdiff_t probs_matrix_size = total_sequence_length;
    const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);

    unit_cost.compute_cycles = static_cast<double>((SafeInt<ptrdiff_t>(2) * head_size - 1) * total_sequence_length);
    unit_cost.bytes_loaded = static_cast<double>(SafeInt<ptrdiff_t>(2) * head_size * total_sequence_length * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(SafeInt<ptrdiff_t>(head_size) * total_sequence_length * sizeof(T));

    if (attn_bias_data != nullptr) {
      unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes) * 2;
      unit_cost.bytes_stored += probs_matrix_bytes;
    }

    if (mask_index_data != nullptr) {
      unit_cost.bytes_stored += probs_matrix_bytes;
    }

    // Cost of appending current key to present key
    unit_cost.compute_cycles += static_cast<double>(head_size);
    unit_cost.bytes_loaded += static_cast<double>(head_size);

    // Parallel for loop
    const int loop_len = batch_size * num_heads_;
    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const std::ptrdiff_t batch_index = i / num_heads_;
        const std::ptrdiff_t head_index = i % num_heads_;
        const std::ptrdiff_t beam_batch_index = batch_index / beam_width;
        const T* q_vec = Q + i * head_size;
        const std::ptrdiff_t attn_bias_base_offset = ((broadcast_attn_bias_dim_0 ? 0 : (beam_batch_index * num_heads_)) +
                                                      (broadcast_attn_bias_dim_1 ? 0 : head_index)) *
                                                     probs_matrix_size;

        {
          // Calculate the latest position of the attention_probs
          // (1, H) x (T, H)^T -> (1, T)
          // Decompose into T (1, H) x (1, H)^T -> (1, 1) operations
          auto last_offset = past_sequence_length + i * probs_matrix_size;
          T* attention_probs_ptr = reinterpret_cast<T*>(attention_probs) + last_offset;
          math::Dot<float, CPUMathUtil>(head_size, q_vec, K + i * head_size, attention_probs_ptr, nullptr);

          *attention_probs_ptr *= scale;
          // Apply the attention bias and mask
          if (attn_bias_data != nullptr) {
            *attention_probs_ptr += attn_bias_data[attn_bias_base_offset + past_sequence_length];
          }
          bool is_masked = (mask_index_data != nullptr) &&
                           (mask_index_data[(batch_index + 1) * total_sequence_length - 1] == 0);
          if (is_masked) {
            *attention_probs_ptr += mask_filter_value_;
          }
        }

        {
          // Calculate the rest of the attention_probs
          for (std::ptrdiff_t j = 0; j < past_sequence_length; ++j) {
            const int* beam_indices = &cache_indir_data[batch_index * max_sequence_length];
            const std::ptrdiff_t beam_offset = static_cast<std::ptrdiff_t>(beam_indices[j]) * num_heads_ *
                                               max_sequence_length * head_size;
            const std::ptrdiff_t beam_batch_offset = (beam_batch_index * beam_width * num_heads_ + head_index) *
                                                     max_sequence_length * head_size;
            const T* past_k_vec = past_key_data + beam_batch_offset + beam_offset + j * head_size;
            T* output = reinterpret_cast<T*>(attention_probs) + j + i * probs_matrix_size;
            math::Dot<float, CPUMathUtil>(head_size, q_vec, past_k_vec, output, nullptr);

            *output *= scale;
            // Apply the attention bias and mask
            if (attn_bias_data != nullptr) {
              *output += attn_bias_data[attn_bias_base_offset + j];
            }
            bool is_masked = (mask_index_data != nullptr) &&
                             (mask_index_data[batch_index * total_sequence_length + j] == 0);
            if (is_masked) {
              *output += mask_filter_value_;
            }
          }
        }

        // Append current key to present key (past_present_share_buffer_ is true)
        memcpy(present_key_data + (i * max_sequence_length + past_sequence_length) * head_size,
               K + i * head_size, head_size * sizeof(T));
      }
    });

    if (output_qk_data != nullptr) {
      // Output the scaled Q*K^T if needed.
      memcpy(output_qk_data, attention_probs,
             SafeInt<size_t>(batch_size) * num_heads_ * total_sequence_length * sizeof(T));
    }

    // attention_probs(B, N, 1, T) = Softmax(attention_probs)
    {
      const int N = batch_size * num_heads_;
      const int D = total_sequence_length;
      ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);
    }
  }

  // Used for DecoderMaskedMultiHeadAttention where sequence_length = 1
  template <typename T>
  void ComputeVxAttentionScoreWithBeams(T* output,
                                        T* tmp_buffer,
                                        const T* attention_probs,
                                        const T* V,
                                        int batch_size,
                                        int past_sequence_length,
                                        int max_sequence_length,
                                        int v_head_size,
                                        const T* past_value_data,
                                        T* present_value_data,
                                        const int32_t* cache_indir_data,
                                        int beam_width,
                                        ThreadPool* tp) const {
    const int total_sequence_length = past_sequence_length + 1;

    TensorOpCost unit_cost;
    unit_cost.compute_cycles = static_cast<double>(SafeInt<ptrdiff_t>(2) * v_head_size * total_sequence_length);
    unit_cost.bytes_loaded = static_cast<double>(SafeInt<ptrdiff_t>(3) * v_head_size * total_sequence_length * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(SafeInt<ptrdiff_t>(2) * v_head_size * total_sequence_length * sizeof(T));

    // Cost of appending current value to present value
    unit_cost.compute_cycles += static_cast<double>(v_head_size);
    unit_cost.bytes_loaded += static_cast<double>(v_head_size);

    ThreadPool::TryParallelFor(tp, SafeInt<ptrdiff_t>(batch_size) * num_heads_, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const std::ptrdiff_t batch_index = i / num_heads_;
        const std::ptrdiff_t head_index = i % num_heads_;
        const std::ptrdiff_t beam_batch_index = batch_index / beam_width;

        // Compute the attention score
        // (1, T) x (T, H_v) -> (1, H_v)
        // Decompose into T (1, 1) x (1, H_v) -> (1, H_v) operations and accumulate.
        {
          const T* attn_probs_ptr = attention_probs + (i + 1) * total_sequence_length - 1;
          math::Scale<T, CPUMathUtil>(v_head_size,
                                      static_cast<float>(*attn_probs_ptr),
                                      V + i * v_head_size,
                                      output + i * v_head_size,
                                      nullptr);
        }
        {
          for (std::ptrdiff_t j = 0; j < past_sequence_length; ++j) {
            const int* beam_indices = &cache_indir_data[batch_index * max_sequence_length];
            const std::ptrdiff_t beam_offset = static_cast<std::ptrdiff_t>(beam_indices[j]) * num_heads_ *
                                               max_sequence_length * v_head_size;
            const std::ptrdiff_t beam_batch_offset = (beam_batch_index * beam_width * num_heads_ + head_index) *
                                                     max_sequence_length * v_head_size;
            const T* past_value_vec = past_value_data + beam_offset + beam_batch_offset;
            const T* attn_probs_ptr = attention_probs + j + i * total_sequence_length;

            math::Scale<T, CPUMathUtil>(v_head_size,
                                        static_cast<float>(*attn_probs_ptr),
                                        past_value_vec + j * v_head_size,
                                        tmp_buffer + i * v_head_size,
                                        nullptr);
            math::Add<T, CPUMathUtil>(v_head_size,
                                      output + i * v_head_size,
                                      tmp_buffer + i * v_head_size,
                                      output + i * v_head_size,
                                      nullptr);
          }
        }

        // Append current value to present value (past_present_share_buffer_ is true)
        memcpy(present_value_data + (i * max_sequence_length + past_sequence_length) * v_head_size,
               V + i * v_head_size,
               v_head_size * sizeof(T));
      }
    });
  }
};

}  // namespace contrib
}  // namespace onnxruntime
