// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/attention.h"
#include "core/providers/cpu/llm/attention_helper.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"

#include <algorithm>
#include <vector>

using onnxruntime::attention_helper::AttentionParameters;
using onnxruntime::attention_helper::QKMatMulOutputMode;
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {

#define REGISTER_ONNX_KERNEL_TYPED(T)                                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                     \
      Attention,                                                      \
      24,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_ONNX_KERNEL_TYPED(float)
REGISTER_ONNX_KERNEL_TYPED(MLFloat16)

#define REGISTER_ONNX_KERNEL_VERSIONED_TYPED(T)                       \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      Attention,                                                      \
      23,                                                             \
      23,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_ONNX_KERNEL_VERSIONED_TYPED(float)
REGISTER_ONNX_KERNEL_VERSIONED_TYPED(MLFloat16)

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
    mask_data[i] = mask_index[i] ? 0.0f : mask_filter_value<float>();
  }
}

template <>
void make_copy<MLFloat16, bool>(MLFloat16* mask_data, const bool* mask_index, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = mask_index[i] ? MLFloat16(0.f) : mask_filter_value<MLFloat16>();
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

// Dispatches a GEMM operation across float and MLFloat16 types.
//   C = alpha * op(A) * op(B) + beta * C
//
// For float: delegates to math::GemmEx which calls MlasGemm (optimized SGEMM).
// For MLFloat16:
//   - If the hardware supports native fp16 GEMM for the given transpose combo
//     (checked via MlasHGemmSupported), uses MlasGemm directly.
//   - Otherwise, upcasts A/B/C to fp32, runs math::GemmEx (SGEMM), and downcasts
//     the result back to fp16.  This avoids Eigen's unoptimized fp16 codepath.
//
// The fp32 fallback handles strided C carefully: when ldc > N (e.g. 3D interleaved
// heads where multiple heads share a row), conversion is done row-by-row (N elements
// per row) to avoid overwriting adjacent heads' data.  When ldc == N (contiguous,
// the common 4D case), a single bulk conversion is used for efficiency.
//
// TODO(xadupre): Consider adding a MlasFlashAttention fast path for float32 when no masks, KV cache,
// softcap, or nonpad_kv_seqlen are active. This fuses Q*K, softmax, and QK*V into a single
// L2-cache-tiled pass. See MultiHeadAttention (contrib_ops/cpu/bert/multihead_attention.cc).
template <typename T>
inline void AttentionGemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                          int M, int N, int K,
                          float alpha,
                          const T* A, int lda,
                          const T* B, int ldb,
                          float beta,
                          T* C, int ldc) {
  if constexpr (std::is_same<T, float>::value) {
    math::GemmEx<T, ThreadPool>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, nullptr);
  } else if constexpr (std::is_same<T, MLFloat16>::value) {
    if (MlasHGemmSupported(transA, transB)) {
      MlasGemm(transA, transB, M, N, K, A, lda, B, ldb, C, ldc,
               MLFloat16(alpha).val, MLFloat16(beta).val, nullptr);
    } else {
      // fp16 fallback: upcast to fp32, run optimized SGEMM, downcast result.
      // Compute the exact contiguous span each matrix occupies: (rows-1)*stride + cols.
      // This is the distance from the first element to the last accessed element + 1.
      // Using rows*stride would overread when the pointer is offset into a larger
      // interleaved buffer (e.g., 3D layout where lda > K for a non-first head).
      size_t a_rows = (transA == CblasNoTrans) ? static_cast<size_t>(M) : static_cast<size_t>(K);
      size_t a_cols = (transA == CblasNoTrans) ? static_cast<size_t>(K) : static_cast<size_t>(M);
      size_t b_rows = (transB == CblasNoTrans) ? static_cast<size_t>(K) : static_cast<size_t>(N);
      size_t b_cols = (transB == CblasNoTrans) ? static_cast<size_t>(N) : static_cast<size_t>(K);
      size_t a_count = (a_rows > 0) ? (a_rows - 1) * static_cast<size_t>(lda) + a_cols : 0;
      size_t b_count = (b_rows > 0) ? (b_rows - 1) * static_cast<size_t>(ldb) + b_cols : 0;
      size_t c_count = (M > 0) ? static_cast<size_t>(M - 1) * static_cast<size_t>(ldc) + static_cast<size_t>(N) : 0;

      std::vector<float> a_fp32(a_count);
      std::vector<float> b_fp32(b_count);
      std::vector<float> c_fp32(c_count);

      // Upcast A and B in bulk (contiguous within each matrix's strided span).
      MlasConvertHalfToFloatBuffer(A, a_fp32.data(), a_count);
      MlasConvertHalfToFloatBuffer(B, b_fp32.data(), b_count);
      if (beta != 0.0f) {
        // C needs upcast only when beta != 0 (GEMM accumulates into C).
        // When ldc == N the buffer is contiguous — use a single bulk conversion.
        // When ldc > N (3D interleaved heads), convert only the N valid columns
        // per row to avoid reading into adjacent heads' memory.
        if (ldc == N) {
          MlasConvertHalfToFloatBuffer(C, c_fp32.data(), c_count);
        } else {
          for (int row = 0; row < M; ++row) {
            MlasConvertHalfToFloatBuffer(C + row * ldc, c_fp32.data() + row * ldc, static_cast<size_t>(N));
          }
        }
      }

      math::GemmEx<float, ThreadPool>(transA, transB, M, N, K,
                                      alpha, a_fp32.data(), lda,
                                      b_fp32.data(), ldb,
                                      beta, c_fp32.data(), ldc, nullptr);

      // Downcast result back to fp16.
      // Same ldc == N check: bulk conversion when contiguous, row-by-row when
      // strided to avoid overwriting adjacent heads' output data.
      if (ldc == N) {
        MlasConvertFloatToHalfBuffer(c_fp32.data(), C, c_count);
      } else {
        for (int row = 0; row < M; ++row) {
          MlasConvertFloatToHalfBuffer(c_fp32.data() + row * ldc, C + row * ldc, static_cast<size_t>(N));
        }
      }
    }
  } else {
    ORT_THROW("Unsupported data type for attention GEMM: ",
              DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
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
                               ? static_cast<attention_helper::QKMatMulOutputMode>(mode)
                               : attention_helper::QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKMask ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftCap ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftMax,
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
  const Tensor* nonpad_kv_seqlen = context->Input<Tensor>(6);  // optional, Opset 24

  AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

  // ComputeOutputShapeForAttention also checks the validity of the inputs.
  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                  Q,
                  K,
                  V,
                  attn_mask,
                  past_key,
                  past_value,
                  nonpad_kv_seqlen,
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
  ORT_ENFORCE(!((past_key != nullptr) && (present_key == nullptr)),
              "The implementation does not support past_key provided and present_key being null.");
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
    // No external mask: allocate only if causal behavior needed.
    if (causal) {
      size_t mask_bytes = SafeInt<size_t>(parameters.q_sequence_length) * parameters.total_sequence_length * sizeof(T);
      void* raw = allocator->Alloc(mask_bytes);
      memset(raw, 0, mask_bytes);  // start all allowed
      mask_data = static_cast<T*>(raw);
      for (int s = 0; s < parameters.q_sequence_length; ++s) {
        for (int t = parameters.past_sequence_length + s + 1; t < parameters.total_sequence_length; ++t) {
          mask_data[s * parameters.total_sequence_length + t] = mask_filter_value<T>();
        }
      }
      delete_mask_data = true;
    }
  } else {
    const bool is_bool_mask = mask_index->IsDataType<bool>();
    const bool need_copy = is_bool_mask || causal;  // copy if we must convert or overlay causal pattern
    if (need_copy) {
      size_t mask_bytes = SafeInt<size_t>(mask_index->Shape().Size()) * sizeof(T);
      mask_data = static_cast<T*>(allocator->Alloc(mask_bytes));
      delete_mask_data = true;
      if (is_bool_mask) {
        make_copy(mask_data, mask_index->Data<bool>(), SafeInt<size_t>(mask_index->Shape().Size()));
      } else {
        make_copy(mask_data, mask_index->Data<T>(), SafeInt<size_t>(mask_index->Shape().Size()));
      }
      if (causal) {
        // Overlay causal -inf above diagonal for every broadcast slice
        int slices = mask_batch_size * mask_num_heads;
        for (int slice = 0; slice < slices; ++slice) {
          T* base = mask_data + probs_matrix_size * slice;
          for (int s = 0; s < parameters.q_sequence_length; ++s) {
            for (int t = parameters.past_sequence_length + s + 1; t < parameters.total_sequence_length; ++t) {
              base[s * parameters.total_sequence_length + t] = mask_filter_value<T>();
            }
          }
        }
      }
    } else {
      // Reuse mask memory directly (numeric, non-causal)
      mask_data = const_cast<T*>(mask_index->Data<T>());
    }
  }

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
  bool transposed_k = parameters.transpose_output && nullptr == present_key;

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
      std::ptrdiff_t head_ki = head_i * parameters.kv_num_heads / parameters.q_num_heads;
      std::ptrdiff_t ki = batch_i * parameters.kv_num_heads + head_ki;
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
      const T* q_ptr = parameters.transpose_output
                           ? Q + q_input_chunk_length * parameters.q_num_heads * batch_i + head_i * parameters.head_size
                           : Q + q_input_chunk_length * i;
      int q_lda = parameters.transpose_output
                      ? parameters.head_size * parameters.q_num_heads
                      : parameters.head_size;
      const T* k_ptr = transposed_k
                           ? K + k_input_chunk_length * parameters.kv_num_heads * batch_i + head_ki * parameters.head_size
                           : k;
      int k_ldb = transposed_k
                      ? parameters.head_size * parameters.kv_num_heads
                      : parameters.head_size;

      AttentionGemm(CblasNoTrans, CblasTrans,
                    parameters.q_sequence_length, parameters.total_sequence_length, parameters.head_size,
                    alpha, q_ptr, q_lda, k_ptr, k_ldb, beta, output, parameters.total_sequence_length);
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
      // Apply nonpad_kv_seqlen masking (Opset 24+): mask out KV positions >= valid length per batch.
      if (parameters.has_nonpad_kv_seqlen) {
        int valid_kv_len = static_cast<int>(parameters.nonpad_kv_seqlen_data[batch_i]);
        for (int s = 0; s < parameters.q_sequence_length; ++s) {
          std::fill(output + s * parameters.total_sequence_length + valid_kv_len,
                    output + (s + 1) * parameters.total_sequence_length,
                    mask_filter_value<T>());
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
  ORT_ENFORCE(!((past_value != nullptr) && (present_value == nullptr)),
              "The implementation does not support past_value provided and present_value being null.");
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
          std::ptrdiff_t head_vi = head_i * kv_num_heads / num_heads;
          std::ptrdiff_t vi = batch_i * kv_num_heads + head_vi;
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

          // Compute QK * V
          ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
          const T* gemm_B;
          int gemm_ldb;
          T* gemm_C;
          int gemm_ldc;

          if (transpose_output) {
            // 3D inputs: V may be in strided layout, use appropriate strides.
            gemm_B = transposed_v ? V + head_vi * v_head_size + v_input_chunk_length * kv_num_heads * batch_i : v;
            gemm_ldb = transposed_v ? v_head_size * kv_num_heads : v_head_size;
            gemm_C = output + ((batch_i * sequence_length * num_heads + head_i) * v_head_size);
            gemm_ldc = v_head_size * num_heads;
          } else {
            // 4D inputs: V is already in head-contiguous layout.
            gemm_B = v;
            gemm_ldb = v_head_size;
            ptrdiff_t dest_offset = SafeInt<ptrdiff_t>(sequence_length) * v_head_size * i;
            gemm_C = output + dest_offset;
            gemm_ldc = v_head_size;
          }

          AttentionGemm(CblasNoTrans, CblasNoTrans,
                        sequence_length, v_head_size, total_sequence_length,
                        1.0f, attention_probs + attention_probs_offset, total_sequence_length,
                        gemm_B, gemm_ldb, 0.0f, gemm_C, gemm_ldc);
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
