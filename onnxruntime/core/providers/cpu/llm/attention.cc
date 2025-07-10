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
inline void ComputeAttentionSoftmaxInplace(T* score, int N, int D, ThreadPool* tp) {
  MlasComputeSoftmax(score, score, N, D, false, false, tp);
}

template <typename T>
inline void ComputeAttentionSoftcapInplace(T* scores, int sequence_length, T softcap) {
  MlasComputeSoftcap(scores, scores, sequence_length, softcap);
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : AttentionBase<T>(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  // kv_num_heads, q_num_head are mandatory for 3D inputs but not used for 4D inputs.
  // The dimension is not yet known. If not specified, the inputs is assumed to be 4D.
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = static_cast<QKMatMulOutputMode>(mode);
  ORT_ENFORCE(qk_matmul_output_mode_ == QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKMask ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKSoftCap ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKSoftMax,
              "qk_matmul_output_mode must be 0, 1, 2, or 3.");
  // The default scale depends on the input dimensions. It is set to nan to indicate that it should be computed.
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

  AttentionParameters parameters;
  std::vector<int64_t> y_shape;
  std::vector<int64_t> present_key_shape;
  std::vector<int64_t> present_value_shape;
  std::vector<int64_t> output_qk_shape;

  attention_helper::ComputeOutputShapeForAttention(
      Q,
      K,
      V,
      attn_mask,
      past_key,
      past_value,
      is_causal_,
      softcap_,
      softmax_precision_,
      qk_matmul_output_mode_,
      kv_num_heads_,
      q_num_heads_,
      scale_,
      parameters,
      y_shape,
      present_key_shape,
      present_value_shape,
      output_qk_shape);

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);
  return this->ApplyAttention(context,
                              Q->Data<T>(),   // Q
                              K->Data<T>(),   // K
                              V->Data<T>(),   // V
                              attn_mask,      // const Tensor* mask_index,  // mask, nullptr if no mask
                              past_key,       // past K input tensor (if not using past state)
                              past_value,     // past V input tensor (if not using past state)
                              Y,              // first output
                              present_key,    // present K output tensor (if separating present KV)
                              present_value,  // present V output tensor (if separating present KV)
                              output_qk,      // Q*K output tensor (if returning Q*K value)
                              parameters      // attention parameters
  );
}

template <typename T>
void AttentionBase<T>::ComputeAttentionProbs(T* attention_probs,                     // output buffer with size BxNxSxT
                                             const T* Q,                             // Q data. Its size is BxNxSxH
                                             const T* K,                             // k data. Its size is BxNxLxH
                                             const Tensor* mask_index,               // mask
                                             const AttentionParameters& parameters,  // attention parameters
                                             const T* past_key,                      // past key only (if not using past state)
                                             T* present_key,                         // present key only (if not using present state)
                                             T* output_qk,                           // Q*K output
                                             ThreadPool* tp,
                                             AllocatorPtr allocator) const {
  const int total_sequence_length = parameters.past_sequence_length + parameters.kv_sequence_length;              // T = P + L
  const size_t past_chunk_length = static_cast<size_t>(parameters.past_sequence_length) * parameters.head_size;   // P x H
  const size_t q_input_chunk_length = static_cast<size_t>(parameters.q_sequence_length) * parameters.head_size;   // S x H
  const size_t k_input_chunk_length = static_cast<size_t>(parameters.kv_sequence_length) * parameters.head_size;  // L x H
  const size_t present_chunk_length = past_chunk_length + k_input_chunk_length;                                   // T x H

  const int loop_len = parameters.batch_size * parameters.q_num_heads;
  const float alpha = parameters.scale;

  TensorOpCost unit_cost;
  const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(parameters.q_sequence_length) * total_sequence_length;
  const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * parameters.head_size * probs_matrix_size);
  unit_cost.bytes_loaded = static_cast<double>((parameters.q_sequence_length + total_sequence_length) * parameters.head_size * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

  if (present_key) {
    double bytes_to_copy_key = present_chunk_length * static_cast<double>(sizeof(T));
    unit_cost.bytes_loaded += bytes_to_copy_key;
    unit_cost.bytes_stored += bytes_to_copy_key;
  }

  // Prepare mask
  // Merge causal mask with padding mask, and convert values from 0/1 to -inf/0.
  int mask_batch_size = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 4 ? 1 : mask_index->Shape().GetDims()[0]);
  int mask_num_heads = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 3
                                            ? 1
                                            : (mask_index->Shape().NumDimensions() < 4 ? mask_index->Shape().GetDims()[0] : mask_index->Shape().GetDims()[1]));

  T* mask_data = nullptr;
  bool delete_mask_data = false;
  bool causal = parameters.is_causal && parameters.q_sequence_length > 1;
  if (mask_index == nullptr) {
    // No mask = null mask.
    if (causal) {
      size_t mask_data_bytes = SafeInt<size_t>(parameters.q_sequence_length) * total_sequence_length * sizeof(T);
      mask_data = static_cast<T*>(allocator->Alloc(mask_data_bytes));
      memset(mask_data, 0, mask_data_bytes);
      for (int s_i = 0; s_i < parameters.q_sequence_length; s_i++) {
        for (int m_i = parameters.past_sequence_length + s_i + 1; m_i < total_sequence_length; m_i++) {
          mask_data[s_i * total_sequence_length + m_i] = std::numeric_limits<T>::lowest();
        }
      }
      delete_mask_data = true;
    }
  } else if (mask_index->IsDataType<bool>() || causal) {
    // We need a copy.
    size_t mask_data_bytes = SafeInt<size_t>(mask_index->Shape().Size()) * sizeof(T);
    mask_data = static_cast<T*>(allocator->Alloc(mask_data_bytes));
    delete_mask_data = true;

    if (mask_index->IsDataType<bool>()) {
      // Convert bool mask to 0/1
      make_copy(mask_data, mask_index->Data<bool>(), mask_index->Shape().Size());
    } else if (mask_index != nullptr) {
      // We make a copy because causal is True.
      make_copy(mask_data, mask_index->Data<T>(), mask_index->Shape().Size());
    }
    if (causal) {
      // This loop could be parallelized.
      int n_iter = mask_batch_size * mask_num_heads;
      for (int i = 0; i < n_iter; ++i) {
        for (int s_i = 0; s_i < parameters.q_sequence_length; s_i++) {
          for (int m_i = parameters.past_sequence_length + s_i + 1; m_i < total_sequence_length; m_i++) {
            mask_data[s_i * total_sequence_length + m_i + probs_matrix_size * i] = std::numeric_limits<T>::lowest();
          }
        }
      }
    }
  } else {
    // Nothing to do, no necessary copy.
    mask_data = const_cast<T*>(mask_index->Data<T>());
  }

  if (nullptr != present_key && parameters.kv_num_heads != parameters.q_num_heads) {
    // This is not part of the main loop because it is not needed at every iteration and
    // we cannot ensure the inner body is executed first before getting used in another iteration.
    // parameters.batch_size * parameters.q_num_heads
    for (std::ptrdiff_t batch_i = 0; batch_i < parameters.batch_size; ++batch_i) {
      for (std::ptrdiff_t head_i = 0; head_i < parameters.kv_num_heads; ++head_i) {
        std::ptrdiff_t ki = batch_i * parameters.kv_num_heads + head_i;
        const T* k = K + k_input_chunk_length * ki;
        ConcatStateChunk(past_key, k, present_key, past_chunk_length, present_chunk_length, ki);
      }
    }
  }

  // Main loop
  ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t i = begin; i != end; ++i) {
      const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
      std::ptrdiff_t batch_i = i / parameters.q_num_heads;
      std::ptrdiff_t head_i = i % parameters.q_num_heads;
      const ptrdiff_t mask_data_offset = probs_matrix_size * (head_i % mask_num_heads + (batch_i % mask_batch_size) * mask_num_heads);

      T* output = attention_probs + output_offset;
      T* out_qk = output_qk == nullptr ? nullptr : output_qk + output_offset;
      T beta;

      if (mask_data != nullptr && (out_qk == nullptr || parameters.qk_matmul_output_mode != attention_helper::QKMatMulOutputMode::kQK)) {
        // Broadcast mask data: SxT -> SxT
        memcpy(output, mask_data + mask_data_offset, probs_matrix_bytes);
        beta = 1;
      } else {
        beta = 0;
      }

      // handling GQA
      std::ptrdiff_t ki = batch_i * parameters.kv_num_heads + head_i % parameters.kv_num_heads;
      const T* k = K + k_input_chunk_length * ki;

      if (nullptr != present_key) {
        if (parameters.kv_num_heads != parameters.q_num_heads) {
          // Already done in a loop before this one.
          k = present_key + ki * present_chunk_length;
        } else {
          k = ConcatStateChunk(past_key, k, present_key, past_chunk_length, present_chunk_length, ki);
        }
      }

      // Compute Q*K' + AttentionMask
      //                     original                 transposed             each iteration
      // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
      // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
      // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
      math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, parameters.q_sequence_length, total_sequence_length, parameters.head_size, alpha,
                                Q + q_input_chunk_length * i, k,
                                beta,
                                output, nullptr);
      if (out_qk != nullptr &&
          (parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKMask ||
           parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK)) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
        if (mask_data != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK) {
          // We need to add the bias we could not add because out_qk was requested without the mask.
          // This can be optimized with vectorized add using MlasAddFloat32x4.
          for (ptrdiff_t j = 0; j < probs_matrix_size; j++) {
            // this trick does not work with infinities.
            output[j] += mask_data[mask_data_offset + j];
          }
        }
      }
      if (parameters.softcap > 0.0f) {
        ComputeAttentionSoftcapInplace(output, static_cast<int>(probs_matrix_size), parameters.softcap);
      }
      if (out_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftCap) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      }
    }
  });
  if (delete_mask_data) {
    allocator->Free(mask_data);
  }
  const int N = parameters.batch_size * parameters.q_num_heads * parameters.q_sequence_length;
  const int D = total_sequence_length;
  ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);

  if (output_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftMax) {
    memcpy(output_qk, attention_probs, SafeInt<size_t>(parameters.batch_size) * parameters.q_num_heads * parameters.q_sequence_length * total_sequence_length * sizeof(T));
  }
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
                                               int kv_num_heads,          // number of KV heads
                                               const T* past_value,       // past value only (if not using past state)
                                               T* present_value,          // present value only (if not using present state)
                                               bool transpose_output,     // whether to transpose the output (0, 2, 1, 3)
                                               ThreadPool* tp) const {
  const int total_sequence_length = past_sequence_length + kv_sequence_length;                  // T = P + L
  const ptrdiff_t past_chunk_length = SafeInt<ptrdiff_t>(past_sequence_length) * v_head_size;   // P x H_v
  const ptrdiff_t q_input_chunk_length = SafeInt<ptrdiff_t>(sequence_length) * v_head_size;     // S x H_v
  const ptrdiff_t v_input_chunk_length = SafeInt<ptrdiff_t>(kv_sequence_length) * v_head_size;  // L x H_v
  const ptrdiff_t present_chunk_length = past_chunk_length + v_input_chunk_length;              // T x H_v

  // The cost of Gemm
  TensorOpCost unit_cost;
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * v_head_size * total_sequence_length);
  unit_cost.bytes_loaded =
      static_cast<double>(SafeInt<ptrdiff_t>(sequence_length + v_head_size) * total_sequence_length * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(sequence_length * v_head_size * sizeof(T));

  const size_t bytes_to_copy_trans = SafeInt<size_t>(v_head_size) * sizeof(T);
  double bytes_to_copy_trans_all = static_cast<double>(sequence_length * bytes_to_copy_trans);
  unit_cost.bytes_loaded += bytes_to_copy_trans_all;
  unit_cost.bytes_stored += bytes_to_copy_trans_all;

  if (nullptr != present_value && kv_num_heads != num_heads) {
    // This is not part of the main loop because it is not needed at every iteration and
    // we cannot ensure the inner body is executed first before getting used in another iteration.
    // parameters.batch_size * parameters.q_num_heads
    for (std::ptrdiff_t batch_i = 0; batch_i < batch_size; ++batch_i) {
      for (std::ptrdiff_t head_i = 0; head_i < kv_num_heads; ++head_i) {
        std::ptrdiff_t vi = batch_i * kv_num_heads + head_i;
        const T* v = V + v_input_chunk_length * vi;
        ConcatStateChunk(past_value, v, present_value, past_chunk_length, present_chunk_length, vi);
      }
    }
  }

  ThreadPool::TryParallelFor(
      tp, SafeInt<ptrdiff_t>(batch_size) * num_heads, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          // handling GQA
          std::ptrdiff_t batch_i = i / num_heads;
          std::ptrdiff_t head_i = i % num_heads;
          std::ptrdiff_t vi = batch_i * kv_num_heads + head_i % kv_num_heads;
          const T* v = V + v_input_chunk_length * vi;

          if (nullptr != present_value) {
            if (kv_num_heads != num_heads) {
              // Already done in a loop before this one.
              v = present_value + vi * present_chunk_length;
            } else {
              v = ConcatStateChunk(past_value, v, present_value, past_chunk_length, present_chunk_length, vi);
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
                                        const T* Q,                            // Q data with shape BxNxSxH
                                        const T* K,                            // K data with shape BxNxLxH
                                        const T* V,                            // V value with size BxNxLxH_v
                                        const Tensor* mask_index,              // mask index. nullptr if no mask or its size is B
                                        const Tensor* past_key,                // past K input tensor (if not using past state)
                                        const Tensor* past_value,              // past V input tensor (if not using past state)
                                        Tensor* output,                        // output tensor
                                        Tensor* present_key,                   // present K output tensor (if separating present KV)
                                        Tensor* present_value,                 // present V output tensor (if separating present KV)
                                        Tensor* output_qk,                     // Q*K output tensor (if returning Q*K value)
                                        const AttentionParameters& parameters  // attention parameters
) const {
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();

  int past_sequence_length = 0;
  if (parameters.past_sequence_length > 0) {
    past_sequence_length = parameters.past_sequence_length;
  } else if (past_key != nullptr && past_value != nullptr) {
    past_sequence_length = static_cast<int>(past_key->Shape().GetDims()[2]);
  }

  // Total sequence length including that of past state: T = P + L
  const int total_sequence_length = parameters.past_sequence_length + parameters.kv_sequence_length;

  const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
  T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
  const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
  T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;
  T* output_qk_data = output_qk != nullptr ? output_qk->MutableData<T>() : nullptr;

  // Compute the attention score.
  size_t bytes = SafeInt<size_t>(parameters.batch_size) * parameters.q_num_heads * parameters.q_sequence_length * total_sequence_length * sizeof(T);
  auto attention_probs = allocator->Alloc(bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));
  this->ComputeAttentionProbs(static_cast<T*>(attention_probs),
                              Q,
                              K,
                              mask_index,
                              parameters,
                              past_key_data,
                              present_key_data,
                              output_qk_data,
                              tp,
                              allocator);

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
                                parameters.kv_num_heads,
                                past_value_data,
                                present_value_data,
                                parameters.transpose_output,
                                tp);

  return Status::OK();
}

}  // namespace onnxruntime

// TODO: rotary embedding in place