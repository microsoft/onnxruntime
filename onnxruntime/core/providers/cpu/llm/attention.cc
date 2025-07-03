// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/attention.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

using onnxruntime::attention_helper::AttentionMaskType;
using onnxruntime::attention_helper::AttentionParameters;
using onnxruntime::attention_helper::AttentionType;
using onnxruntime::attention_helper::QKMatMulOutputMode;
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {

#define REGISTER_ONNX_KERNEL_TYPED(T)                                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                     \
      Attention,                                                      \
      23,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_ONNX_KERNEL_TYPED(float)

template <typename T>
inline void ComputeAttentionSoftmaxInplace(T* score, int N, int D, ThreadPool* tp) {
  MlasComputeSoftmax(score, score, N, D, false, false, tp);
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : AttentionBase<T>(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = static_cast<QKMatMulOutputMode>(mode);
  ORT_ENFORCE(qk_matmul_output_mode_ == QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKV,
              "qk_matmul_output_mode must be 0 (None), 1 (QK), or 2 (QKV)");
  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");
}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);
  ORT_ENFORCE(Q != nullptr && K != nullptr && V != nullptr,
              "Q, K, and V inputs must not be null");
  int q_dims = onnxruntime::narrow<int>(Q->Shape().NumDimensions());
  int k_dims = onnxruntime::narrow<int>(K->Shape().NumDimensions());
  int v_dims = onnxruntime::narrow<int>(V->Shape().NumDimensions());
  ORT_ENFORCE(q_dims == 3 || q_dims == 4, "Q must be a 3D or 4D tensor");
  ORT_ENFORCE(q_dims == k_dims, "Q and K must have the same rank.");
  ORT_ENFORCE(q_dims == v_dims, "Q and V must have the same rank.");

  AttentionParameters parameters;
  parameters.is_causal = is_causal_;
  parameters.softcap = softcap_;                                              // softcap
  parameters.softmax_precision = softmax_precision_;                          // precision for softmax, 0 for float16, 1 for float32
  parameters.qk_matmul_output_mode = qk_matmul_output_mode_;                  // output mode for Q*K matmul
  parameters.mask_type = attn_mask == nullptr ? AttentionMaskType::MASK_NONE  // mask type
                                              : (attn_mask->IsDataType<bool>() ? AttentionMaskType::MASK_BOOL : AttentionMaskType::MASK_ADD);
  parameters.batch_size = onnxruntime::narrow<int>(Q->Shape()[0]);  // Q.shape[0], K.shape[0], V.shape[0] (4D)

  ORT_ENFORCE(parameters.batch_size > 0, "Batch size must be greater than 0");
  ORT_ENFORCE(parameters.mask_type == AttentionMaskType::MASK_NONE ||
                  parameters.mask_type == AttentionMaskType::MASK_BOOL ||
                  attn_mask->IsDataType<float>(),
              "Attention mask must be of type bool or float if provided.");

  std::vector<int64_t> y_shape;
  if (q_dims == 4) {
    // 4D
    parameters.kv_num_heads = kv_num_heads_ > 0 ? kv_num_heads_ : onnxruntime::narrow<int>(K->Shape()[1]);  // K.shape[1] or V.shape[1] (4D)
    parameters.q_num_heads = q_num_heads_ > 0 ? q_num_heads_ : onnxruntime::narrow<int>(Q->Shape()[1]);     // Q.shape[1] (4D)

    ORT_ENFORCE(parameters.kv_num_heads == onnxruntime::narrow<int>(K->Shape()[1]), "kv_num_heads different from K.shape[1]");
    ORT_ENFORCE(parameters.kv_num_heads == onnxruntime::narrow<int>(V->Shape()[1]), "kv_num_heads different from V.shape[1]");
    ORT_ENFORCE(parameters.q_num_heads == onnxruntime::narrow<int>(Q->Shape()[1]), "q_num_heads different from Q.shape[1]");

    // From shapes
    parameters.transpose_output = false;                                      // whether to transpose the output from BxNxSxH to BxSxNxH
    parameters.q_sequence_length = onnxruntime::narrow<int>(Q->Shape()[2]);   // Q.shape[2] (4D)
    parameters.head_size = onnxruntime::narrow<int>(Q->Shape()[3]);           // Q.shape[3] (4D)
    parameters.kv_sequence_length = onnxruntime::narrow<int>(K->Shape()[2]);  // K.shape[2] or V.shape[2] (4D)
    parameters.v_head_size = onnxruntime::narrow<int>(V->Shape()[3]);         // V.shape[3] (4D)
    parameters.past_sequence_length = past_key == nullptr                     // past_key.shape[2] or past_value.shape[2] (4D) or given by the mask
                                          ? (attn_mask == nullptr
                                                 ? 0
                                                 : onnxruntime::narrow<int>(attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 1]) - parameters.kv_sequence_length)
                                          : onnxruntime::narrow<int>(past_key->Shape()[2]);

    y_shape = {static_cast<int64_t>(parameters.batch_size),
               static_cast<int64_t>(parameters.q_num_heads),
               static_cast<int64_t>(parameters.q_sequence_length),
               static_cast<int64_t>(parameters.v_head_size)};
  } else {
    // 3D
    parameters.kv_num_heads = kv_num_heads_;
    parameters.q_num_heads = q_num_heads_;

    // From shapes
    parameters.transpose_output = true;  // whether to transpose the output from BxNxSxH to BxSxNxH
    parameters.q_sequence_length = onnxruntime::narrow<int>(Q->Shape()[1]);
    parameters.head_size = onnxruntime::narrow<int>(Q->Shape()[2]) / parameters.q_num_heads;
    parameters.kv_sequence_length = onnxruntime::narrow<int>(K->Shape()[1]);
    parameters.v_head_size = onnxruntime::narrow<int>(V->Shape()[2]) / parameters.kv_num_heads;
    parameters.past_sequence_length = past_key == nullptr
                                          ? (attn_mask == nullptr
                                                 ? 0
                                                 : onnxruntime::narrow<int>(attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 1]) - parameters.kv_sequence_length)
                                          : onnxruntime::narrow<int>(past_key->Shape()[2]);

    y_shape = {static_cast<int64_t>(parameters.batch_size),
               static_cast<int64_t>(parameters.q_sequence_length),
               static_cast<int64_t>(parameters.q_num_heads * parameters.v_head_size)};
  }
  parameters.scale = std::isnan(scale_) ? static_cast<float>(1.0 / sqrt(parameters.head_size)) : scale_;
  ORT_ENFORCE(parameters.getAttentionType() != AttentionType::kInvalid, "Invalid attention type. q_num_heads must be equal to kv_num_heads or a multiple of it.");

  std::vector<int64_t> present_key_shape = {static_cast<int64_t>(parameters.batch_size),
                                            static_cast<int64_t>(parameters.kv_num_heads),
                                            static_cast<int64_t>(parameters.past_sequence_length + parameters.kv_sequence_length),
                                            static_cast<int64_t>(parameters.head_size)};
  std::vector<int64_t> present_value_shape = {static_cast<int64_t>(parameters.batch_size),
                                              static_cast<int64_t>(parameters.kv_num_heads),
                                              static_cast<int64_t>(parameters.past_sequence_length + parameters.kv_sequence_length),
                                              static_cast<int64_t>(parameters.v_head_size)};
  std::vector<int64_t> output_qk_shape = {static_cast<int64_t>(parameters.batch_size),
                                          static_cast<int64_t>(parameters.q_num_heads),
                                          static_cast<int64_t>(parameters.q_sequence_length),
                                          static_cast<int64_t>(parameters.past_sequence_length + parameters.kv_sequence_length)};

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_key_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);

  return this->ApplyAttention(context,
                              Q->Data<T>(),   // Q
                              K->Data<T>(),   // K
                              V->Data<T>(),   // V
                              attn_mask,      // const Tensor* mask_index,  // mask, nullptr if no mask
                              nullptr,        // const Tensor* past,        // past state
                              past_key,       // past K input tensor (if not using past state)
                              past_value,     // past V input tensor (if not using past state)
                              Y,              // first output
                              present_key,    // present K output tensor (if separating present KV)
                              present_value,  // present V output tensor (if separating present KV)
                              output_qk,      // Q*K output tensor (if returning Q*K value)
                              parameters,     // attention parameters
                              nullptr,        // const Tensor* attn_bias,  // additive bias applied on scaled QK.
                              false           // bool past_present_share_buffer
  );
}

template <typename T>
void AttentionBase<T>::ComputeAttentionProbs(T* attention_probs,                       // output buffer with size BxNxSxT
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
                                             ThreadPool* tp,                           // thread pool
                                             float scale,                              // scale factor
                                             const T* attn_bias_data,                  // attention bias
                                             gsl::span<const int64_t> attn_bias_dims,  // attention bias shape
                                             bool past_present_share_buffer,
                                             int max_sequence_length) const {
  const int total_sequence_length = past_sequence_length + kv_sequence_length;               // T = P + L
  const size_t past_chunk_length = static_cast<size_t>(past_sequence_length) * head_size;    // P x H
  const size_t q_input_chunk_length = static_cast<size_t>(sequence_length) * head_size;      // S x H
  const size_t kv_input_chunk_length = static_cast<size_t>(kv_sequence_length) * head_size;  // L x H
  const size_t present_chunk_length = past_chunk_length + kv_input_chunk_length;             // T x H
  const size_t cache_chunk_length = static_cast<size_t>(max_sequence_length) * head_size;    // M x H

  {
    const int loop_len = batch_size * num_heads;
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
      double bytes_to_copy_key = (past_present_share_buffer ? kv_input_chunk_length : present_chunk_length) *
                                 static_cast<double>(sizeof(T));
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
        const int batch_index = static_cast<int>(i) / num_heads;
        const std::ptrdiff_t head_index = i % static_cast<std::ptrdiff_t>(num_heads);

        const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
        const ptrdiff_t mask_offset = SafeInt<ptrdiff_t>(batch_index) * probs_matrix_size;

        T* output = attention_probs + output_offset;

        if (attn_bias_data != nullptr) {
          // Attention bias has shape (B or 1, N or 1, S, T)
          // Here we handle the broadcast of batch_size and num_heads dimensions.
          ptrdiff_t attn_bias_offset = 0;
          if (attn_bias_dims[0] != 1) {
            attn_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attn_bias_dims[1] * probs_matrix_size;
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
          if (past_present_share_buffer) {
            k = present_key + cache_chunk_length * i;
            memcpy(const_cast<T*>(k) + past_chunk_length, K + head_size * i, head_size * sizeof(T));
          } else {
            k = ConcatStateChunk(past_key, k, present_key, past_chunk_length, present_chunk_length, i);
          }
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

  if (output_qk != nullptr) {
    // Output the scaled Q*K^T if needed.
    memcpy(output_qk, attention_probs,
           SafeInt<size_t>(batch_size) * num_heads * sequence_length * total_sequence_length * sizeof(T));
  }

  const int N = batch_size * num_heads * sequence_length;
  const int D = total_sequence_length;
  ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);
}

template <typename T>
T* AttentionBase<T>::ConcatStateChunk(const T* past,
                                      const T* chunk,
                                      T* present,
                                      size_t past_chunk_length,
                                      size_t present_chunk_length,
                                      std::ptrdiff_t i) const {
  T* start = present + i * present_chunk_length;

  T* p = start;
  if (nullptr != past) {
    const T* src_past = past + i * past_chunk_length;
    memcpy(p, src_past, past_chunk_length * sizeof(T));
    p += past_chunk_length;
  }

  memcpy(p, chunk, (present_chunk_length - past_chunk_length) * sizeof(T));
  return start;
}

template <typename T>
void AttentionBase<T>::ComputeVxAttentionScore(T* output,                 // buffer for the result with size BxSxNxH_v
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
                                               bool transpose_output,     // whether to transpose the output (0, 2, 1, 3)
                                               ThreadPool* tp,
                                               bool past_present_share_buffer,
                                               int max_sequence_length) const {
  const int total_sequence_length = past_sequence_length + kv_sequence_length;                   // T = P + L
  const ptrdiff_t past_chunk_length = SafeInt<ptrdiff_t>(past_sequence_length) * v_head_size;    // P x H_v
  const ptrdiff_t q_input_chunk_length = SafeInt<ptrdiff_t>(sequence_length) * v_head_size;      // S x H_v
  const ptrdiff_t kv_input_chunk_length = SafeInt<ptrdiff_t>(kv_sequence_length) * v_head_size;  // L x H_v
  const ptrdiff_t present_chunk_length = past_chunk_length + kv_input_chunk_length;              // T x H_v
  const ptrdiff_t cache_chunk_length = SafeInt<ptrdiff_t>(max_sequence_length) * v_head_size;    // M x H_v

  // Move the pointer of past and present to start of v values.
  if (nullptr != past) {
    past += SafeInt<ptrdiff_t>(batch_size) * num_heads * past_sequence_length * v_head_size;
  }
  if (nullptr != present) {
    present += SafeInt<ptrdiff_t>(batch_size) * num_heads * total_sequence_length * v_head_size;
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
      tp, SafeInt<ptrdiff_t>(batch_size) * num_heads, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
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

          if (transpose_output) {
            T* current_tmp_data = reinterpret_cast<T*>(tmp_buffer) + q_input_chunk_length * i;
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
            math::MatMul<T>(sequence_length, v_head_size, total_sequence_length,
                            attention_probs + attention_probs_offset, v, current_tmp_data, nullptr);

            // Transpose: out(B, S, N, H_v) -> out_tmp(B, N, S, H_v)
            const int batch_index = static_cast<int>(i / num_heads);
            const int head_index = static_cast<int>(i % num_heads);
            T* src = current_tmp_data;
            ptrdiff_t dest_offset =
                (SafeInt<ptrdiff_t>(batch_index) * sequence_length * num_heads + head_index) * v_head_size;
            T* dest = output + dest_offset;
            for (int j = 0; j < sequence_length; j++) {
              memcpy(dest, src, bytes_to_copy_trans);
              src += v_head_size;
              dest += v_hidden_size;
            }
          } else {
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
            ptrdiff_t dest_offset = SafeInt<ptrdiff_t>(sequence_length) * v_head_size * i;
            T* dest = output + dest_offset;
            math::MatMul<T>(sequence_length, v_head_size, total_sequence_length,
                            attention_probs + attention_probs_offset, v, dest, nullptr);
          }
        }
      });
}

template <typename T>
Status AttentionBase<T>::ApplyAttention(OpKernelContext* context,
                                        const T* Q,                             // Q data with shape BxNxSxH
                                        const T* K,                             // K data with shape BxNxLxH
                                        const T* V,                             // V value with size BxNxLxH_v
                                        const Tensor* mask_index,               // mask index. nullptr if no mask or its size is B
                                        const Tensor* past,                     // past state
                                        const Tensor* past_key,                 // past K input tensor (if not using past state)
                                        const Tensor* past_value,               // past V input tensor (if not using past state)
                                        Tensor* output,                         // output tensor
                                        Tensor* present_key,                    // present K output tensor (if separating present KV)
                                        Tensor* present_value,                  // present V output tensor (if separating present KV)
                                        Tensor* output_qk,                      // Q*K output tensor (if returning Q*K value)
                                        const AttentionParameters& parameters,  // attention parameters
                                        const Tensor* attn_bias,                // additive bias applied on scaled QK.
                                        bool past_present_share_buffer          // memory optimization
) const {
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();

  Tensor* present = nullptr;
  int past_sequence_length;
  if (parameters.past_sequence_length == 0) {
    if (present_key == nullptr && present_value == nullptr) {
      past_sequence_length = (nullptr != past) ? static_cast<int>(past->Shape().GetDims()[3]) : 0;
      present = this->GetPresent(context,
                                 past,
                                 parameters.batch_size,
                                 parameters.v_head_size,
                                 parameters.q_num_heads,
                                 parameters.kv_sequence_length,
                                 past_sequence_length);
    } else if (past_key != nullptr && past_value != nullptr) {
      past_sequence_length = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
  }

  // Total sequence length including that of past state: T = P + L
  const int total_sequence_length = parameters.past_sequence_length + parameters.kv_sequence_length;

  // Merge causal mask with padding mask, and convert values from 0/1 to -inf/0, then broadcast to 3D (BxSxT).
  void* mask_data = nullptr;
  bool causal = parameters.is_causal && parameters.q_sequence_length > 1;
  if (mask_index != nullptr || causal) {
    size_t mask_data_bytes = SafeInt<size_t>(parameters.batch_size) * parameters.q_sequence_length * total_sequence_length * sizeof(T);
    mask_data = allocator->Alloc(mask_data_bytes);
    memset(mask_data, 0, mask_data_bytes);
  }

  BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));
  if (mask_data != nullptr) {
    // Convert mask from boolean (0/1) to float (mask_filter_value/0.0f).
    // Merge padding mask with causal mask, and broadcast to 3D (BxSxT).
    if (mask_index == nullptr) {
      this->PrepareMask(static_cast<T*>(nullptr), gsl::span<const int64_t>{}, static_cast<T*>(mask_data),
                        causal, parameters.batch_size, parameters.q_sequence_length,
                        parameters.kv_sequence_length, parameters.past_sequence_length);
    } else if (mask_index->IsDataType<bool>()) {
      this->PrepareMask(mask_index->Data<bool>(), mask_index->Shape().GetDims(), static_cast<T*>(mask_data),
                        causal, parameters.batch_size, parameters.q_sequence_length,
                        parameters.kv_sequence_length, parameters.past_sequence_length);
    } else {
      // Convert float mask to 0/1
      this->PrepareMask(mask_index->Data<T>(), mask_index->Shape().GetDims(), static_cast<T*>(mask_data),
                        causal, parameters.batch_size, parameters.q_sequence_length,
                        parameters.kv_sequence_length, parameters.past_sequence_length);
    }
  }

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
  size_t bytes = SafeInt<size_t>(parameters.batch_size) * parameters.q_num_heads * parameters.q_sequence_length * total_sequence_length * sizeof(T);
  auto attention_probs = allocator->Alloc(bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));
  this->ComputeAttentionProbs(static_cast<T*>(attention_probs),
                              Q,
                              K,
                              static_cast<T*>(mask_data),
                              parameters.batch_size,
                              parameters.q_sequence_length,
                              parameters.kv_sequence_length,
                              parameters.past_sequence_length,
                              parameters.head_size == 0 ? parameters.v_head_size : parameters.head_size,
                              parameters.q_num_heads,
                              past_data,
                              past_key_data, present_data,
                              present_key_data,
                              output_qk_data,
                              tp,
                              parameters.scale,
                              attn_bias_data,
                              attn_bias_dims,
                              past_present_share_buffer,
                              max_sequence_length);

  void* out_tmp_data = nullptr;
  if (parameters.transpose_output) {
    // Compute the attentionScore * Value: out_tmp(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
    out_tmp_data = allocator->Alloc(SafeInt<size_t>(parameters.batch_size) * parameters.q_num_heads * parameters.q_sequence_length * parameters.v_head_size * sizeof(T));
  }
  BufferUniquePtr out_tmp_buffer(out_tmp_data, out_tmp_data == nullptr ? BufferDeleter(nullptr) : BufferDeleter(std::move(allocator)));

  int v_hidden_size = parameters.q_num_heads * parameters.v_head_size;
  this->ComputeVxAttentionScore(output->MutableData<T>(),
                                static_cast<T*>(out_tmp_data),
                                static_cast<T*>(attention_probs),
                                V,
                                parameters.batch_size,
                                parameters.q_sequence_length,
                                parameters.kv_sequence_length,
                                parameters.past_sequence_length,
                                parameters.v_head_size,
                                v_hidden_size,
                                parameters.q_num_heads,
                                past_data,
                                past_value_data,
                                present_data,
                                present_value_data,
                                parameters.transpose_output,
                                tp,
                                past_present_share_buffer,
                                max_sequence_length);

  return Status::OK();
}

template <typename T>
Tensor* AttentionBase<T>::GetPresent(OpKernelContext* context,
                                     const Tensor* past,
                                     int batch_size,
                                     int head_size,
                                     int num_heads,
                                     int kv_sequence_length,
                                     int past_sequence_length) const {
  // Input and output shapes:
  //   past        : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   present     : (2, batch_size, num_heads, past_sequence_length + kv_sequence_length, head_size)

  std::array<int64_t, 5> present_dims{2, batch_size, num_heads, static_cast<int64_t>(kv_sequence_length) + past_sequence_length, head_size};

  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);
  if (nullptr != past && nullptr == present) {
    ORT_THROW("Expect to have present state output when past state input is given");
  }

  return present;
}

template <typename T, typename U>
void make_copy(T* mask_data, const U* mask_index, size_t size);

template <>
void make_copy<float, float>(float* mask_data, const float* mask_index, size_t size) {
  memcpy(mask_data, mask_index, size * sizeof(float));
}

template <>
void make_copy<float, bool>(float* mask_data, const bool* mask_index, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = mask_index[i] ? 0.0f : std::numeric_limits<float>::lowest();
  }
}

template <typename T>
template <typename U>
void AttentionBase<T>::PrepareMask(const U* mask_index,
                                   gsl::span<const int64_t> mask_index_dims,
                                   T* mask_data,
                                   bool causal,
                                   int batch_size,
                                   int sequence_length,
                                   int kv_sequence_length,
                                   int past_sequence_length) const {
  const int all_sequence_length = past_sequence_length + kv_sequence_length;

  // mask_data has been filled with 0, and its shape is BxSxT
  T* p_mask = mask_data;

  // 4D mask in Megatron GPT2 is currently not support in CPU kernel
  if (nullptr != mask_index && mask_index_dims.size() == 4) {
    ORT_NOT_IMPLEMENTED("4D mask in attention cpu kernel is not supported");
  }

  // For 3D mask, convert values 0 to mask_filter_value, and 1 to 0.0, then apply unidirectional mask if any.
  if (nullptr != mask_index && mask_index_dims.size() == 3) {
    make_copy(p_mask, mask_index, batch_size * sequence_length * all_sequence_length);
    if (causal) {
      for (int b_i = 0; b_i < batch_size; b_i++) {
        for (int s_i = 0; s_i < sequence_length; s_i++) {
          for (int m_i = past_sequence_length + s_i + 1; m_i < all_sequence_length; m_i++) {
            p_mask[s_i * all_sequence_length + m_i] = std::numeric_limits<T>::lowest();
          }
        }
        p_mask += static_cast<size_t>(sequence_length) * all_sequence_length;
      }
    }
  } else {
    if (nullptr != mask_index) {
      make_copy(p_mask, mask_index, sequence_length * all_sequence_length);
    } else {
      memset(p_mask, 0, sequence_length * all_sequence_length * sizeof(T));
    }
    if (causal) {
      for (int s_i = 0; s_i < sequence_length; s_i++) {
        for (int m_i = past_sequence_length + s_i + 1; m_i < all_sequence_length; m_i++) {
          p_mask[s_i * all_sequence_length + m_i] = std::numeric_limits<T>::lowest();
        }
      }
    }
    for (int b_i = 1; b_i < batch_size; b_i++) {
      memcpy(p_mask + b_i * sequence_length * all_sequence_length, mask_data, sequence_length * all_sequence_length * sizeof(T));
    }
  }
}

}  // namespace onnxruntime

// TODO: rotary embedding in place