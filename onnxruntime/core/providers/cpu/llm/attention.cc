// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/attention.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"

using onnxruntime::attention_helper::AttentionParameters;
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
REGISTER_ONNX_KERNEL_TYPED(MLFloat16)

template <typename T, typename U>
void make_copy(T* mask_data, const U* mask_index, size_t size);

template <>
void make_copy<float, float>(float* mask_data, const float* mask_index, size_t size) {
  memcpy(mask_data, mask_index, size * sizeof(float));
}

template <>
void make_copy<MLFloat16, MLFloat16>(MLFloat16* mask_data, const MLFloat16* mask_index, size_t size) {
  memcpy(mask_data, mask_index, size * sizeof(MLFloat16));
}

template <>
void make_copy<float, bool>(float* mask_data, const bool* mask_index, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = mask_index[i] ? 0.0f : std::numeric_limits<float>::lowest();
  }
}

template <>
void make_copy<MLFloat16, bool>(MLFloat16* mask_data, const bool* mask_index, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = mask_index[i] ? MLFloat16(0.f) : std::numeric_limits<MLFloat16>::lowest();
  }
}

template <typename T>
inline void ComputeAttentionSoftmaxInplace(T* score, int N, int D, ThreadPool* tp, AllocatorPtr) {
  MlasComputeSoftmax(score, score, N, D, false, false, 0.0f, tp);
}

template <>
inline void ComputeAttentionSoftmaxInplace<MLFloat16>(MLFloat16* score, int N, int D, ThreadPool* tp, AllocatorPtr allocator) {
  ORT_ENFORCE(tp == nullptr, "No parallelized version of softmax for float16.");
  // Mlas Lacks kernels for fp16 softmax, we convert into float32 and call the float32 version.
  void* allocated_ptr = allocator->Alloc(static_cast<size_t>(N * D * sizeof(float)));
  BufferUniquePtr float_buffer(allocated_ptr, BufferDeleter(allocator));
  float* ptr = reinterpret_cast<float*>(allocated_ptr);
  MlasConvertHalfToFloatBuffer(score, ptr, N * D);
  MlasComputeSoftmax(ptr, ptr, N, D, false, false, 0.0f, tp);
  MlasConvertFloatToHalfBuffer(ptr, score, N * D);
}

template <typename T>
inline void ComputeAttentionSoftcapInplace(T* scores, int sequence_length, T softcap) {
  MlasComputeSoftcap(scores, scores, sequence_length, softcap);
}

template <>
inline void ComputeAttentionSoftcapInplace(MLFloat16* scores, int sequence_length, MLFloat16 softcap) {
  // Mlas Lacks kernels for fp16 softcap. The code is similar to the softcap implementation in mlas.
  float x;
  float cap = softcap.ToFloat();
  for (size_t i = 0; i < static_cast<size_t>(sequence_length); i++) {
    x = std::tanh(scores[i].ToFloat() / cap) * cap;
    scores[i] = MLFloat16(x);
  }
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : AttentionBase<T>(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  // kv_num_heads, q_num_head are mandatory for 3D inputs but not used for 4D inputs.
  // The dimension is not yet known. If not specified, the inputs is assumed to be 4D.
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = info.node().OutputDefs().size() >= 4 && info.node().OutputDefs()[3]->Exists()
                               ? static_cast<QKMatMulOutputMode>(mode)
                               : QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQK ||
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

  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
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
                  output_qk_shape)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = parameters.qk_matmul_output_mode == QKMatMulOutputMode::kNone
                          ? nullptr
                          : context->Output(3, output_qk_shape);
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
  // The case past_key != nullptr and present_key == nullptr is not supported.
  // We use the fact present_key is requested to avoid any extra allocation.
  // However, if present_key is not requested, we should avoid allocated more memory than needed but that mean
  // allocating one buffer per thread. That's why the implementation is not done.
  // The user should define a model with a present_key even if not used if past_key is not null.
  ORT_ENFORCE((past_key == nullptr) == (present_key == nullptr),
              "The implementation only supports past_key and present_key both null or both not null.");
  const size_t past_chunk_length = static_cast<size_t>(parameters.past_sequence_length) * parameters.head_size;   // P x H
  const size_t q_input_chunk_length = static_cast<size_t>(parameters.q_sequence_length) * parameters.head_size;   // S x H
  const size_t k_input_chunk_length = static_cast<size_t>(parameters.kv_sequence_length) * parameters.head_size;  // L x H
  const size_t present_chunk_length = past_chunk_length + k_input_chunk_length;                                   // T x H

  TensorOpCost unit_cost;
  const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(parameters.q_sequence_length) *
                                      parameters.total_sequence_length;
  const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * parameters.head_size * probs_matrix_size);
  unit_cost.bytes_loaded = static_cast<double>((parameters.q_sequence_length +
                                                parameters.total_sequence_length) *
                                               parameters.head_size * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

  if (present_key) {
    double bytes_to_copy_key = present_chunk_length * static_cast<double>(sizeof(T));
    unit_cost.bytes_loaded += bytes_to_copy_key;
    unit_cost.bytes_stored += bytes_to_copy_key;
  }

  // Prepare mask
  // Merge causal mask with padding mask, and convert values from 0/1 to -inf/0.
  int mask_batch_size = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 4
                                             ? 1
                                             : mask_index->Shape().GetDims()[0]);
  int mask_num_heads = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 3
                                            ? 1
                                            : (mask_index->Shape().NumDimensions() < 4
                                                   ? mask_index->Shape().GetDims()[0]
                                                   : mask_index->Shape().GetDims()[1]));

  T* mask_data = nullptr;
  bool delete_mask_data = false;
  bool causal = parameters.is_causal && parameters.q_sequence_length > 1;
  if (mask_index == nullptr) {
    // No mask = null mask.
    if (causal) {
      size_t mask_data_bytes = SafeInt<size_t>(parameters.q_sequence_length) * parameters.total_sequence_length * sizeof(T);
      void* allocated_ptr = allocator->Alloc(mask_data_bytes);
      memset(allocated_ptr, 0, mask_data_bytes);
      mask_data = static_cast<T*>(allocated_ptr);
      for (int s_i = 0; s_i < parameters.q_sequence_length; s_i++) {
        for (int m_i = parameters.past_sequence_length + s_i + 1; m_i < parameters.total_sequence_length; m_i++) {
          mask_data[s_i * parameters.total_sequence_length + m_i] = std::numeric_limits<T>::lowest();
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
      make_copy(mask_data, mask_index->Data<bool>(), SafeInt<size_t>(mask_index->Shape().Size()));
    } else if (mask_index != nullptr) {
      // We make a copy because causal is True.
      make_copy(mask_data, mask_index->Data<T>(), SafeInt<size_t>(mask_index->Shape().Size()));
    }
    if (causal) {
      // This loop could be parallelized.
      // According to the specifications, this configuration is not supported
      // as is_causal=1 or mask is not None (exclusive or).
      int n_iter = mask_batch_size * mask_num_heads;
      for (int i = 0; i < n_iter; ++i) {
        for (int s_i = 0; s_i < parameters.q_sequence_length; s_i++) {
          for (int m_i = parameters.past_sequence_length + s_i + 1; m_i < parameters.total_sequence_length; m_i++) {
            mask_data[s_i * parameters.total_sequence_length + m_i + probs_matrix_size * i] = std::numeric_limits<T>::lowest();
          }
        }
      }
    }
  } else {
    // Nothing to do, no necessary copy.
    mask_data = const_cast<T*>(mask_index->Data<T>());
  }

  bool transposed_k = parameters.transpose_output && nullptr == present_key;
  if (nullptr != present_key && parameters.kv_num_heads != parameters.q_num_heads) {
    // This is not part of the main loop because it is not needed at every iteration and
    // we cannot ensure the inner body is executed first before getting used in another iteration.
    // parameters.batch_size * parameters.q_num_heads
    for (std::ptrdiff_t batch_i = 0; batch_i < parameters.batch_size; ++batch_i) {
      for (std::ptrdiff_t head_i = 0; head_i < parameters.kv_num_heads; ++head_i) {
        ConcatStateChunk(past_key, K, present_key,
                         past_chunk_length, k_input_chunk_length, present_chunk_length,
                         parameters.kv_num_heads, parameters.head_size, batch_i, head_i,
                         parameters.transpose_output);
      }
    }
  }

  // If present_key is not null, it is already initialized to zero.
  // Main loop
  // With 3D inputs, both Q and K are transposed with permutations (0, 2, 1, 3).
  // To avoid expressing the transposition, we use GemmEx with different values for lda, ldb.
  // If past_key is not null, then we need to concatenate it with K, the concatenation is not transposed.
  const int loop_len = parameters.batch_size * parameters.q_num_heads;
  const float alpha = parameters.scale;

  ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t i = begin; i != end; ++i) {
      const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
      std::ptrdiff_t batch_i = i / parameters.q_num_heads;
      std::ptrdiff_t head_i = i % parameters.q_num_heads;
      const ptrdiff_t mask_data_offset = probs_matrix_size *
                                         (head_i % mask_num_heads + (batch_i % mask_batch_size) * mask_num_heads);

      T* output = attention_probs + output_offset;
      T* out_qk = output_qk == nullptr ? nullptr : output_qk + output_offset;
      float beta;

      if (mask_data != nullptr &&
          (out_qk == nullptr || parameters.qk_matmul_output_mode != attention_helper::QKMatMulOutputMode::kQK)) {
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
          k = ConcatStateChunk(past_key, K, present_key,
                               past_chunk_length, k_input_chunk_length, present_chunk_length,
                               parameters.kv_num_heads, parameters.head_size, batch_i, head_i,
                               parameters.transpose_output);
        }
      }

      // Compute Q*K' + AttentionMask
      //                     original                 transposed             each iteration
      // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
      // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
      // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
      if constexpr (std::is_same<T, float>::value) {
        if (parameters.transpose_output) {
          math::GemmEx<T, ThreadPool>(CblasNoTrans,
                                      CblasTrans,
                                      parameters.q_sequence_length,      // M
                                      parameters.total_sequence_length,  // N
                                      parameters.head_size,              // K
                                      alpha,
                                      Q + q_input_chunk_length * parameters.q_num_heads * batch_i + head_i * parameters.head_size,
                                      parameters.head_size * parameters.q_num_heads,  // lda
                                      transposed_k ? K + k_input_chunk_length * parameters.kv_num_heads * batch_i + head_i * parameters.head_size : k,
                                      transposed_k ? parameters.head_size * parameters.kv_num_heads : parameters.head_size,  // ldb
                                      beta,
                                      output,
                                      parameters.total_sequence_length,  // ldc
                                      nullptr);
        } else {
          math::Gemm<T, ThreadPool>(CblasNoTrans,
                                    CblasTrans,
                                    parameters.q_sequence_length,      // M
                                    parameters.total_sequence_length,  // N
                                    parameters.head_size,              // K
                                    alpha,
                                    Q + q_input_chunk_length * i,
                                    k,
                                    beta,
                                    output,
                                    nullptr);
        }
      } else if constexpr (std::is_same<T, MLFloat16>::value) {
        if (MlasHGemmSupported(CblasNoTrans, CblasTrans)) {
          MlasGemm(CblasNoTrans,
                   CblasTrans,
                   parameters.q_sequence_length,      // M
                   parameters.total_sequence_length,  // N
                   parameters.head_size,              // K
                   parameters.transpose_output
                       ? Q + q_input_chunk_length * parameters.q_num_heads * batch_i + head_i * parameters.head_size
                       : Q + q_input_chunk_length * i,
                   parameters.transpose_output
                       ? parameters.head_size * parameters.q_num_heads
                       : static_cast<int>(parameters.head_size),  // lda
                   transposed_k
                       ? K + k_input_chunk_length * parameters.kv_num_heads * batch_i + head_i * parameters.head_size
                       : k,
                   transposed_k
                       ? parameters.head_size * parameters.kv_num_heads
                       : static_cast<int>(parameters.head_size),  // ldb
                   output,
                   static_cast<int>(parameters.past_sequence_length + parameters.kv_sequence_length),  // ldc
                   MLFloat16(alpha).val, MLFloat16(beta).val, nullptr);
        } else {
          if (parameters.transpose_output) {
            math::GemmEx<T, ThreadPool>(CblasNoTrans,
                                        CblasTrans,
                                        parameters.q_sequence_length,      // M
                                        parameters.total_sequence_length,  // N
                                        parameters.head_size,              // K
                                        MLFloat16(alpha),
                                        Q + q_input_chunk_length * parameters.q_num_heads * batch_i + head_i * parameters.head_size,
                                        parameters.head_size * parameters.q_num_heads,  // lda
                                        transposed_k ? K + k_input_chunk_length * parameters.kv_num_heads * batch_i + head_i * parameters.head_size : k,
                                        transposed_k ? parameters.head_size * parameters.kv_num_heads : parameters.head_size,  // ldb
                                        MLFloat16(beta),
                                        output,
                                        parameters.total_sequence_length,  // ldc
                                        nullptr);
          } else {
            TensorShape c_shape({parameters.q_sequence_length, parameters.total_sequence_length});
            Gemm_MLFloat16(CblasNoTrans, CblasTrans,
                           static_cast<ptrdiff_t>(parameters.q_sequence_length),      // M
                           static_cast<ptrdiff_t>(parameters.total_sequence_length),  // N
                           static_cast<ptrdiff_t>(parameters.head_size),              // K
                           MLFloat16(alpha),
                           Q + q_input_chunk_length * i,
                           k,
                           MLFloat16(beta),
                           output,
                           &c_shape,
                           output,
                           nullptr);
          }
        }
      } else {
        ORT_THROW("Unsupported data type for attention Q*K multiplication: ", DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
      }
      if (out_qk != nullptr &&
          (parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKMask ||
           parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK)) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
        if (mask_data != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK) {
          // We need to add the bias we could not add because out_qk was requested without the mask.
          // This can be optimized with vectorized add using MlasAddFloat32x4.
          MlasEltwiseAdd(output, mask_data + mask_data_offset, output, probs_matrix_size);
        }
      }
      if (parameters.softcap > 0.0f) {
        if constexpr (std::is_same<T, float>::value) {
          ComputeAttentionSoftcapInplace(output, static_cast<int>(probs_matrix_size), parameters.softcap);
        } else if constexpr (std::is_same<T, MLFloat16>::value) {
          ComputeAttentionSoftcapInplace(output, static_cast<int>(probs_matrix_size), MLFloat16(parameters.softcap));
        } else {
          ORT_THROW("Unsupported data type for ComputeAttentionSoftcapInplace: ",
                    DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
        }
      }
      if (out_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftCap) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      }
      ComputeAttentionSoftmaxInplace(output, parameters.q_sequence_length, parameters.total_sequence_length, nullptr, allocator);

      if (output_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftMax) {
        memcpy(output_qk + output_offset, output,
               SafeInt<size_t>(parameters.q_sequence_length) * parameters.total_sequence_length * sizeof(T));
      }
    }
  });
  if (delete_mask_data) {
    allocator->Free(mask_data);
  }
}

template <typename T>
T* AttentionBase<T>::ConcatStateChunk(const T* past,
                                      const T* base_chunk,  // chunk is K or V, it can be transposed or not
                                      T* present,
                                      size_t past_chunk_length,
                                      size_t input_chunk_length,  // chunk length of K or V
                                      size_t present_chunk_length,
                                      size_t num_heads,
                                      size_t head_size,
                                      std::ptrdiff_t batch_i,
                                      std::ptrdiff_t head_i,
                                      bool transposed) const {
  std::ptrdiff_t i = batch_i * num_heads + head_i % num_heads;

  T* start = present + i * present_chunk_length;

  T* p = start;
  if (nullptr != past) {
    const T* src_past = past + i * past_chunk_length;
    memcpy(p, src_past, past_chunk_length * sizeof(T));
    p += past_chunk_length;
  }

  if (transposed) {
    ORT_ENFORCE(head_size > 0 && num_heads > 0 && batch_i >= 0 && head_i >= 0,
                "Invalid parameters for ConcatStateChunk: head_size=", head_size, ", batch_i=", batch_i, ", head_i=", head_i);
    size_t sequence_length = SafeInt<size_t>(input_chunk_length / head_size);
    const T* chunk = base_chunk + head_i * head_size + input_chunk_length * num_heads * batch_i;
    for (size_t j = 0; j < sequence_length; ++j) {
      memcpy(p, chunk, head_size * sizeof(T));
      p += head_size;
      chunk += num_heads * head_size;
    }
  } else {
    const T* chunk = base_chunk + input_chunk_length * i;
    memcpy(p, chunk, (present_chunk_length - past_chunk_length) * sizeof(T));
  }
  return start;
}

template <typename T>
void AttentionBase<T>::ComputeVxAttentionScore(T* output,                  // buffer for the result with size BxSxNxH_v
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
                                               bool transpose_output,      // whether to transpose the output (0, 2, 1, 3)
                                               ThreadPool* tp) const {
  ORT_ENFORCE((past_value == nullptr) == (present_value == nullptr),
              "The implementation only supports past_value and present_value both null or both not null.");
  const ptrdiff_t past_chunk_length = SafeInt<ptrdiff_t>(past_sequence_length) * v_head_size;   // P x H_v
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

  bool transposed_v = transpose_output && nullptr == present_value;
  if (nullptr != present_value && kv_num_heads != num_heads) {
    // This is not part of the main loop because it is not needed at every iteration and
    // we cannot ensure the inner body is executed first before getting used in another iteration.
    // parameters.batch_size * parameters.q_num_heads
    for (std::ptrdiff_t batch_i = 0; batch_i < batch_size; ++batch_i) {
      for (std::ptrdiff_t head_i = 0; head_i < kv_num_heads; ++head_i) {
        ConcatStateChunk(past_value, V, present_value,
                         past_chunk_length, v_input_chunk_length, present_chunk_length,
                         kv_num_heads, v_head_size, batch_i, head_i,
                         transpose_output);
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
              // transposed_v is false here.
              v = ConcatStateChunk(past_value, V, present_value,
                                   past_chunk_length, v_input_chunk_length, present_chunk_length,
                                   kv_num_heads, v_head_size, batch_i, head_i,
                                   transpose_output);
            }
          }

          if (transpose_output) {
            // transpose_output is false
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;

            if constexpr (std::is_same<T, float>::value) {
              // V is transposed but not QK. We use GemmEx with a different value for ldb.
              math::GemmEx<T, ThreadPool>(CblasNoTrans,
                                          CblasNoTrans,
                                          sequence_length,                                                                              // M
                                          v_head_size,                                                                                  // N
                                          total_sequence_length,                                                                        // K
                                          1.f,                                                                                          // alpha
                                          attention_probs + attention_probs_offset,                                                     // QK
                                          total_sequence_length,                                                                        // lda
                                          transposed_v ? V + head_i * v_head_size + v_input_chunk_length * kv_num_heads * batch_i : v,  // V
                                          transposed_v ? v_head_size * kv_num_heads : v_head_size,                                      // ldb
                                          0.f,                                                                                          // beta
                                          output + ((batch_i * sequence_length * num_heads + head_i) * v_head_size),
                                          v_head_size * num_heads,  // ldc
                                          nullptr);
            } else if constexpr (std::is_same<T, MLFloat16>::value) {
              // This switch should probably be moved to math_cpu.h.
              if (MlasHGemmSupported(CblasNoTrans, CblasNoTrans)) {
                MlasGemm(CblasNoTrans,
                         CblasNoTrans,
                         sequence_length,        // M
                         v_head_size,            // N
                         total_sequence_length,  // K
                         attention_probs + attention_probs_offset,
                         total_sequence_length,  // lda
                         transposed_v ? V + head_i * v_head_size + v_input_chunk_length * kv_num_heads * batch_i : v,
                         transposed_v ? static_cast<int>(v_head_size * kv_num_heads) : static_cast<int>(v_head_size),  // ldb
                         output + ((batch_i * sequence_length * num_heads + head_i) * v_head_size),
                         v_head_size * num_heads,  // ldc
                         MLFloat16(1.f).val, MLFloat16(0.f).val, nullptr);
              } else {
                math::GemmEx<T, ThreadPool>(CblasNoTrans,
                                            CblasNoTrans,
                                            sequence_length,                                                                              // M
                                            v_head_size,                                                                                  // N
                                            total_sequence_length,                                                                        // K
                                            MLFloat16(1.f),                                                                               // alpha
                                            attention_probs + attention_probs_offset,                                                     // QK
                                            total_sequence_length,                                                                        // lda
                                            transposed_v ? V + head_i * v_head_size + v_input_chunk_length * kv_num_heads * batch_i : v,  // V
                                            transposed_v ? v_head_size * kv_num_heads : v_head_size,                                      // ldb
                                            MLFloat16(0.f),                                                                               // beta
                                            output + ((batch_i * sequence_length * num_heads + head_i) * v_head_size),
                                            v_head_size * num_heads,  // ldc
                                            nullptr);
              }
            } else {
              ORT_THROW("Unsupported data type for attention QK*V multiplication: ",
                        DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
            }
          } else {
            // transpose_output is false
            ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
            ptrdiff_t dest_offset = SafeInt<ptrdiff_t>(sequence_length) * v_head_size * i;
            T* dest = output + dest_offset;

            if constexpr (std::is_same<T, float>::value) {
              math::MatMul<T>(sequence_length, v_head_size, total_sequence_length,
                              attention_probs + attention_probs_offset, v, dest, nullptr);
            } else if constexpr (std::is_same<T, MLFloat16>::value) {
              if (MlasHGemmSupported(CblasNoTrans, CblasNoTrans)) {
                MlasGemm(CblasNoTrans,
                         CblasNoTrans,
                         sequence_length,        // M
                         v_head_size,            // N
                         total_sequence_length,  // K
                         attention_probs + attention_probs_offset,
                         total_sequence_length,  // lda
                         v,
                         static_cast<int>(v_head_size),  // ldb
                         dest,
                         static_cast<int>(v_head_size),  // ldc
                         MLFloat16(1.f).val, MLFloat16(0.f).val, nullptr);
              } else {
                Gemm_MLFloat16(CblasNoTrans,
                               CblasNoTrans,
                               static_cast<ptrdiff_t>(sequence_length),        // M
                               static_cast<ptrdiff_t>(v_head_size),            // N
                               static_cast<ptrdiff_t>(total_sequence_length),  // K
                               MLFloat16(1.f),                                 // alpha
                               attention_probs + attention_probs_offset,
                               v,
                               MLFloat16(0.f),  // beta
                               nullptr,
                               nullptr,
                               dest,
                               nullptr);
              }
            } else {
              ORT_THROW("Unsupported data type for attention QK*V multiplication: ",
                        DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
            }
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

  const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
  T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
  const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
  T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;
  T* output_qk_data = output_qk != nullptr ? output_qk->MutableData<T>() : nullptr;

  // Compute the attention score.
  size_t bytes = SafeInt<size_t>(parameters.batch_size) * parameters.q_num_heads *
                 parameters.q_sequence_length * parameters.total_sequence_length * sizeof(T);
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

  this->ComputeVxAttentionScore(output->MutableData<T>(),
                                static_cast<T*>(attention_probs),
                                V,
                                parameters.batch_size,
                                parameters.q_sequence_length,
                                parameters.kv_sequence_length,
                                parameters.past_sequence_length,
                                parameters.total_sequence_length,
                                parameters.v_head_size,
                                parameters.q_num_heads,
                                parameters.kv_num_heads,
                                past_value_data,
                                present_value_data,
                                parameters.transpose_output,
                                tp);

  return Status::OK();
}

}  // namespace onnxruntime
