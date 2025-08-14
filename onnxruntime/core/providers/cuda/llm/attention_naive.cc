/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for attention.
*/

#include "core/providers/cuda/llm/attention_naive.h"
#include "core/providers/cuda/llm/attention_naive_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"

using namespace onnxruntime::cuda;
using onnxruntime::attention_helper::AttentionParameters;
using onnxruntime::attention_helper::QKMatMulOutputMode;

namespace onnxruntime {
namespace cuda {

template <typename T>
void ComputeAttentionProbs(cudaStream_t stream,
                           T* attention_probs,                     // output buffer with size BxNxSxT
                           const T* Q,                             // Q data. Its size is BxNxSxH
                           const T* K,                             // k data. Its size is BxNxLxH
                           const Tensor* mask_index,               // mask
                           const AttentionParameters& parameters,  // attention parameters
                           const T* past_key,                      // past key only (if not using past state)
                           T* present_key,                         // present key only (if not using present state)
                           T* output_qk                           // Q*K output
                           ) {
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

  const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(parameters.q_sequence_length) *
                                      parameters.total_sequence_length;
  const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);

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
      ORT_THROW("causal not implemented yet.");
    }
  } else if (mask_index->IsDataType<bool>() || causal) {
    ORT_THROW("boolean mask not implemented yet.");
  } else {
    // Nothing to do, no necessary copy.
    mask_data = const_cast<T*>(mask_index->Data<T>());
  }

  bool transposed_k = parameters.transpose_output && nullptr == present_key;
  if (nullptr != present_key && parameters.kv_num_heads != parameters.q_num_heads) {
    ORT_THROW("past cache and kv_num_heads != q_num_heads not implemented yet.");
  }

  // If present_key is not null, it is already initialized to zero.
  // Main loop
  // With 3D inputs, both Q and K are transposed with permutations (0, 2, 1, 3).
  // To avoid expressing the transposition, we use GemmEx with different values for lda, ldb.
  // If past_key is not null, then we need to concatenate it with K, the concatenation is not transposed.
  const int loop_len = parameters.batch_size * parameters.q_num_heads;
  const float alpha = parameters.scale;
  int dtype = utils::GetONNXTensorElementDataType<T>();

  for (std::ptrdiff_t i = 0; i != loop_len; ++i) {
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
      beta = 1;
      ORT_THROW("mask_data != nullptr and out_qk == nullptr or parameters.qk_matmul_output_mode != attention_helper::QKMatMulOutputMode::kQK not implemented yet.");
      // memcpy(output, mask_data + mask_data_offset, probs_matrix_bytes);
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
        ORT_THROW("past_key and present_key not implemented yet.");
        /*
        k = ConcatStateChunk(past_key, K, present_key,
                             past_chunk_length, k_input_chunk_length, present_chunk_length,
                             parameters.kv_num_heads, parameters.head_size, batch_i, head_i,
                             parameters.transpose_output);
        */
      }
    }

    // Compute Q*K' + AttentionMask
    //                     original                 transposed             each iteration
    // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
    // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
    // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
    GemmMatMul(
        stream,
        beta != 0,  // has_bias,
        false,      // has_scales, --> subtypes
        dtype,      // dtype_A,
        dtype,      // dtype_B,
        dtype,      // dtype_C,
        dtype,      // dtype_Y,
        false,      // trans_A,
        true,       // trans_B,
        parameters.transpose_output ? Q + q_input_chunk_length * parameters.q_num_heads * batch_i + head_i * parameters.head_size
                                    : Q + q_input_chunk_length * i,  // p_input_a
        transposed_k
            ? K + k_input_chunk_length * parameters.kv_num_heads * batch_i + head_i * parameters.head_size
            : k,                                        // p_input_b
        output,                                         // p_output_y
        beta == 0 ? nullptr : output,                   // p_input_c
        nullptr,                                        // p_scale_a
        nullptr,                                        // p_scale_b
        nullptr,                                        // p_scale_y
        parameters.q_sequence_length,                   // M
        parameters.total_sequence_length,               // N
        parameters.head_size,                           // K
        parameters.head_size * parameters.q_num_heads,  // lda
        transposed_k
            ? parameters.head_size * parameters.kv_num_heads
            : parameters.head_size,        // ldb
        parameters.total_sequence_length,  // ldc
        true,                              // row_major_compute
        -1,                                // sm_count
        CUBLASLT_EPILOGUE_DEFAULT,
        alpha,
        beta);

    if (out_qk != nullptr &&
        (parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKMask ||
         parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK)) {
      // ORT_THROW("out_qk != nullptr and parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKMask or kQK not implemented yet.");
      //  memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      if (mask_data != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK) {
        // We need to add the bias we could not add because out_qk was requested without the mask.
        // This can be optimized with vectorized add using MlasAddFloat32x4.
        ORT_THROW("mask_data != nullptr and parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK not implemented yet.");
        // MlasEltwiseAdd(output, mask_data + mask_data_offset, output, probs_matrix_size);
      } else {
        ORT_THROW("out_qk != nullptr and parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKMask or kQK not implemented yet.");
      }
    }
    if (parameters.softcap > 0.0f) {
      ORT_THROW("parameters.softcap > 0.0f not implemented yet.");
      // ComputeAttentionSoftcapInplace(output, static_cast<int>(probs_matrix_size), parameters.softcap);
    }
    if (out_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftCap) {
      ORT_THROW("out_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftCap not implemented yet.");
      // memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
    }
    typedef typename ToCudaType<T>::MappedType CudaT;

    // ComputeAttentionSoftmaxInplace(output, parameters.q_sequence_length, parameters.total_sequence_length, nullptr, allocator);
    onnxruntime::contrib::attention_softmax_cuda::ComputeSoftmax<CudaT>(
        stream,
        parameters.total_sequence_length,
        parameters.q_sequence_length,
        1,                                  // parameters.batch_size,
        1,                                  // num_heads,
        reinterpret_cast<CudaT*>(nullptr),  // data.attention_bias,
        false,                              // broadcast_attn_bias_dim_0,
        false,                              // broadcast_attn_bias_dim_1,
        reinterpret_cast<CudaT*>(output),   // input
        reinterpret_cast<CudaT*>(output),   // output
        false);                             // causal

    if (output_qk != nullptr && parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftMax) {
      ORT_THROW("output_qk != nullptr and parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQKSoftMax not implemented yet.");
      // memcpy(output_qk + output_offset, output,
      //       SafeInt<size_t>(parameters.q_sequence_length) * parameters.total_sequence_length * sizeof(T));
    }
  }

  if (delete_mask_data) {
    // allocator->Free(mask_data);
    ORT_THROW("delete_mask_data not implemented yet.");
  }
}

template <typename T>
void ComputeVxAttentionScore(cudaStream_t stream,
                             T* output,                  // buffer for the result with size BxSxNxH_v
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
                             bool transpose_output) {    // whether to transpose the output (0, 2, 1, 3)
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
    ORT_THROW("past cache and kv_num_heads != q_num_heads not implemented yet.");
  }
  int dtype = utils::GetONNXTensorElementDataType<T>();

  for (std::ptrdiff_t i = 0; i != batch_size * num_heads; ++i) {
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
        ORT_THROW("past_value and present_value not implemented yet.");
      }
    }

    if (transpose_output) {
      // transpose_output is false
      ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;

      // V is transposed but not QK. We use a different value for ldb.
      GemmMatMul(
          stream,
          false,                                     // has_bias,
          false,                                     // has_scales, --> subtypes
          dtype,                                     // dtype_A,
          dtype,                                     // dtype_B,
          dtype,                                     // dtype_C,
          dtype,                                     // dtype_Y,
          false,                                     // trans_A,
          false,                                     // trans_B,
          attention_probs + attention_probs_offset,  // QK = p_input_a
          transposed_v ? V + head_i * v_head_size + v_input_chunk_length * kv_num_heads * batch_i
                       : v,                                                           // V =p_input_b
          output + ((batch_i * sequence_length * num_heads + head_i) * v_head_size),  // p_output_y
          nullptr,                                                                    // p_input_c
          nullptr,                                                                    // p_scale_a
          nullptr,                                                                    // p_scale_b
          nullptr,                                                                    // p_scale_y
          sequence_length,                                                            // M
          v_head_size,                                                                // N
          total_sequence_length,                                                      // K
          total_sequence_length,                                                      // lda
          transposed_v ? v_head_size * kv_num_heads : v_head_size,                    // ldb
          v_head_size * num_heads,                                                    // ldc
          true,                                                                       // row_major_compute
          -1,                                                                         // sm_count
          CUBLASLT_EPILOGUE_DEFAULT,
          1.f,
          0.f);
    } else {
      // transpose_output is false
      ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
      ptrdiff_t dest_offset = SafeInt<ptrdiff_t>(sequence_length) * v_head_size * i;
      T* dest = output + dest_offset;

      GemmMatMul(
          stream,
          false,                                     // has_bias,
          false,                                     // has_scales, --> subtypes
          dtype,                                     // dtype_A,
          dtype,                                     // dtype_B,
          dtype,                                     // dtype_C,
          dtype,                                     // dtype_Y,
          false,                                     // trans_A,
          false,                                     // trans_B,
          attention_probs + attention_probs_offset,  // QK = p_input_a
          v,                                         // V =p_input_b
          dest,                                      // p_output_y
          nullptr,                                   // p_input_c
          nullptr,                                   // p_scale_a
          nullptr,                                   // p_scale_b
          nullptr,                                   // p_scale_y
          sequence_length,                           // M
          v_head_size,                               // N
          total_sequence_length,                     // K
          total_sequence_length,                     // lda
          v_head_size,                               // ldb
          v_head_size,                               // ldc
          true,                                      // row_major_compute
          -1,                                        // sm_count
          CUBLASLT_EPILOGUE_DEFAULT,
          1.0f,
          0.f);
    }
  }
}

template <typename T>
Status NaiveAttention<T>::ApplyAttention(OpKernelContext* context,
                                         cudaStream_t stream,
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
  // cudaStream_t stream = kernel->Stream(context);
  ComputeAttentionProbs(stream,
                        static_cast<T*>(attention_probs),
                        Q,
                        K,
                        mask_index,
                        parameters,
                        past_key_data,
                        present_key_data,
                        output_qk_data);

  ComputeVxAttentionScore(stream, output->MutableData<T>(),
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
                          parameters.transpose_output);
  return Status::OK();
}

#define IMPLEMENT(T)                                                                                                                         \
  template class NaiveAttention<T>;                                                                                                          \
  template void ComputeAttentionProbs<T>(cudaStream_t stream, T * attention_probs, const T* Q, const T* K, const Tensor* mask_index,         \
                                         const AttentionParameters& parameters, const T* past_key, T* present_key,                           \
                                         T* output_qk);                                                              \
  template void ComputeVxAttentionScore<T>(cudaStream_t stream, T * output, const T* attention_probs, const T* V, int batch_size,            \
                                           int sequence_length, int kv_sequence_length, int past_sequence_length, int total_sequence_length, \
                                           int v_head_size, int num_heads, int kv_num_heads, const T* past_value,                            \
                                           T* present_value, bool transpose_output);

IMPLEMENT(float);
IMPLEMENT(MLFloat16);
// IMPLEMENT(BFloat16);

}  // namespace cuda
}  // namespace onnxruntime
