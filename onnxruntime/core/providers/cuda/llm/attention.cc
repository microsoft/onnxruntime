// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "core/providers/cuda/llm/attention_mask_impl.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                            \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      23,                                                             \
      23,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#define REGISTER_KERNEL_TYPED_24(T)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      24,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED_24(float)
REGISTER_KERNEL_TYPED_24(MLFloat16)
REGISTER_KERNEL_TYPED_24(BFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
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
  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");
}

// ============================================================================
// RunFlashAttention: Direct flash attention kernel call
// ============================================================================
template <typename T>
Status Attention<T>::RunFlashAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
#if USE_FLASH_ATTENTION
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
  const bool is_bf16 = std::is_same<T, BFloat16>::value;
  const bool is_bsnh = parameters.transpose_output;  // 3D inputs → BSNH

  // --- Common buffer allocation ---
  size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(
      parameters.q_sequence_length, parameters.batch_size, parameters.q_num_heads);

  auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      onnxruntime::flash::get_num_splits_and_buffer_sizes(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.total_sequence_length, parameters.q_num_heads,
          parameters.head_size, device_prop.multiProcessorCount);

  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

  if (softmax_lse_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(softmax_lse_accum_buffer.get(), 0,
                                         softmax_lse_accum_bytes, cuda_stream));
  }
  if (out_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(out_accum_buffer.get(), 0,
                                         out_accum_bytes, cuda_stream));
  }

  // --- Transpose Q from BNSH to BSNH (flash always expects Q as BSNH) ---
  const void* q_data = Q->Data<T>();
  IAllocatorUniquePtr<void> q_bsnh_buffer;
  if (!is_bsnh) {
    size_t q_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                     parameters.q_num_heads * parameters.head_size;
    q_bsnh_buffer = GetScratchBuffer<void>(q_bytes, context->GetComputeStream());
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          reinterpret_cast<const half*>(Q->Data<T>()),
          reinterpret_cast<half*>(q_bsnh_buffer.get()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          Q->Data<BFloat16>(), reinterpret_cast<BFloat16*>(q_bsnh_buffer.get()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          Q->Data<float>(), reinterpret_cast<float*>(q_bsnh_buffer.get()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
    q_data = q_bsnh_buffer.get();
  }

  // Flash outputs BSNH. If Y expects BNSH, write to scratch then transpose.
  void* out_data = Y->MutableData<T>();
  IAllocatorUniquePtr<void> out_bsnh_buffer;
  if (!is_bsnh) {
    size_t out_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                       parameters.q_num_heads * parameters.v_head_size;
    out_bsnh_buffer = GetScratchBuffer<void>(out_bytes, context->GetComputeStream());
    out_data = out_bsnh_buffer.get();
  }

  bool present_kv_already_populated = false;

  // --- Path 1: nonpad_kv_seqlen (opset 24 external KV cache) ---
  if (nonpad_kv_seqlen != nullptr) {
    ORT_ENFORCE(parameters.past_sequence_length == 0,
                "RunFlashAttention with nonpad_kv_seqlen requires K/V to be the full cache "
                "(past_sequence_length must be 0, got ",
                parameters.past_sequence_length, ").");

    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(),
        seqlens_k_buffer.get(),
        parameters.batch_size,
        parameters.total_sequence_length,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, cuda_stream,
        const_cast<void*>(q_data),
        const_cast<void*>(static_cast<const void*>(K->Data<T>())),
        const_cast<void*>(static_cast<const void*>(V->Data<T>())),
        /*k=*/nullptr, /*v=*/nullptr,
        out_data,
        softmax_lse_buffer.get(),
        const_cast<void*>(static_cast<const void*>(seqlens_k_buffer.get())),
        /*rotary_cos=*/nullptr, /*rotary_sin=*/nullptr,
        /*cache_batch_idx=*/nullptr, /*leftpad_k=*/nullptr,
        /*head_sink=*/nullptr, /*block_table=*/nullptr,
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.kv_sequence_length,
        /*seqlen_k_new=*/0, /*rotary_dim=*/0,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        /*past_bsnh=*/is_bsnh,
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        /*local_window_size=*/-1, /*is_rotary_interleaved=*/false,
        /*is_packed_qkv=*/false));
  }
  // --- Path 2: Decode with past KV cache ---
  else if (past_key != nullptr) {
    ORT_ENFORCE(past_value != nullptr, "past_key requires past_value.");
    ORT_ENFORCE(present_key != nullptr && present_value != nullptr,
                "present_key/value outputs are required when past_key is provided.");

    // Zero present buffers before strided copy to avoid stale data in positions
    // beyond past_seq that mha_fwd_kvcache might read during attention (matching GQA pattern).
    const size_t num_kv_rows = static_cast<size_t>(parameters.batch_size) * parameters.kv_num_heads;
    const size_t present_k_bytes = num_kv_rows * parameters.total_sequence_length *
                                   parameters.head_size * sizeof(T);
    const size_t present_v_bytes = num_kv_rows * parameters.total_sequence_length *
                                   parameters.v_head_size * sizeof(T);
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(present_key->MutableData<T>(), 0,
                                         present_k_bytes, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(present_value->MutableData<T>(), 0,
                                         present_v_bytes, cuda_stream));

    // Copy past KV (BNSH) into present buffers (BNSH). Strided copy because
    // past has [B, N_kv, past_seq, H] and present has [B, N_kv, total_seq, H].
    const size_t past_k_row_bytes = static_cast<size_t>(parameters.past_sequence_length) *
                                    parameters.head_size * sizeof(T);
    const size_t present_k_row_bytes = static_cast<size_t>(parameters.total_sequence_length) *
                                       parameters.head_size * sizeof(T);
    CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(
        present_key->MutableData<T>(), present_k_row_bytes,
        past_key->Data<T>(), past_k_row_bytes,
        past_k_row_bytes, num_kv_rows,
        cudaMemcpyDeviceToDevice, cuda_stream));

    const size_t past_v_row_bytes = static_cast<size_t>(parameters.past_sequence_length) *
                                    parameters.v_head_size * sizeof(T);
    const size_t present_v_row_bytes = static_cast<size_t>(parameters.total_sequence_length) *
                                       parameters.v_head_size * sizeof(T);
    CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(
        present_value->MutableData<T>(), present_v_row_bytes,
        past_value->Data<T>(), past_v_row_bytes,
        past_v_row_bytes, num_kv_rows,
        cudaMemcpyDeviceToDevice, cuda_stream));

    // seqlens_k: derive per-batch sequence lengths for the KV cache.
    // mha_fwd_kvcache expects seqlens_k = tokens already in cache BEFORE appending new ones.
    // When a bool mask is present, it encodes total valid token count (past + new).
    // Subtract kv_sequence_length to get the pre-append count.
    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    if (attn_mask != nullptr && attn_mask->IsDataType<bool>()) {
      size_t mask_dims = attn_mask->Shape().NumDimensions();
      auto dims = attn_mask->Shape().GetDims();
      int64_t mask_dim0 = dims[0];
      int64_t mask_dim1 = mask_dims >= 3 ? dims[1] : 0;
      int64_t mask_dim2 = mask_dims >= 4 ? dims[2] : 0;
      // Offset: mask gives total valid count, subtract kv_sequence_length for pre-append count.
      int seqlen_offset = -parameters.kv_sequence_length;
      ORT_RETURN_IF_ERROR(LaunchConvertMaskToFlashSeqlensK(
          attn_mask->Data<bool>(), seqlens_k_buffer.get(),
          parameters.batch_size, parameters.total_sequence_length,
          static_cast<int>(mask_dims), mask_dim0, mask_dim1, mask_dim2,
          cuda_stream, device_prop.maxThreadsPerBlock, seqlen_offset));
    } else {
      ORT_RETURN_IF_ERROR(LaunchFillInt32(seqlens_k_buffer.get(), parameters.past_sequence_length,
                                          parameters.batch_size, cuda_stream,
                                          device_prop.maxThreadsPerBlock));
    }

    // K/V new tokens: mha_fwd_kvcache expects BSNH for k_new/v_new.
    // When !is_bsnh (4D BNSH input), transpose new tokens to BSNH.
    const void* k_new = K->Data<T>();
    const void* v_new = V->Data<T>();
    IAllocatorUniquePtr<void> k_bsnh_buffer;
    IAllocatorUniquePtr<void> v_bsnh_buffer;
    if (!is_bsnh) {
      size_t k_bytes = sizeof(T) * parameters.batch_size * parameters.kv_sequence_length *
                       parameters.kv_num_heads * parameters.head_size;
      size_t v_bytes = sizeof(T) * parameters.batch_size * parameters.kv_sequence_length *
                       parameters.kv_num_heads * parameters.v_head_size;
      k_bsnh_buffer = GetScratchBuffer<void>(k_bytes, context->GetComputeStream());
      v_bsnh_buffer = GetScratchBuffer<void>(v_bytes, context->GetComputeStream());
      if constexpr (std::is_same_v<T, MLFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            reinterpret_cast<const half*>(K->Data<T>()),
            reinterpret_cast<half*>(k_bsnh_buffer.get()),
            cuda_stream, device_prop.maxThreadsPerBlock));
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            reinterpret_cast<const half*>(V->Data<T>()),
            reinterpret_cast<half*>(v_bsnh_buffer.get()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            K->Data<BFloat16>(), reinterpret_cast<BFloat16*>(k_bsnh_buffer.get()),
            cuda_stream, device_prop.maxThreadsPerBlock));
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            V->Data<BFloat16>(), reinterpret_cast<BFloat16*>(v_bsnh_buffer.get()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            K->Data<float>(), reinterpret_cast<float*>(k_bsnh_buffer.get()),
            cuda_stream, device_prop.maxThreadsPerBlock));
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            V->Data<float>(), reinterpret_cast<float*>(v_bsnh_buffer.get()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      }
      k_new = k_bsnh_buffer.get();
      v_new = v_bsnh_buffer.get();
    }

    // mha_fwd_kvcache: present_key/value as cache (BNSH), K/V as new tokens (BSNH).
    // The kernel appends new tokens at position seqlens_k[b] and attends to all.
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, cuda_stream,
        const_cast<void*>(q_data),
        static_cast<void*>(present_key->MutableData<T>()),
        static_cast<void*>(present_value->MutableData<T>()),
        const_cast<void*>(k_new), const_cast<void*>(v_new),
        out_data,
        softmax_lse_buffer.get(),
        static_cast<void*>(seqlens_k_buffer.get()),
        /*rotary_cos=*/nullptr, /*rotary_sin=*/nullptr,
        /*cache_batch_idx=*/nullptr, /*leftpad_k=*/nullptr,
        /*head_sink=*/nullptr, /*block_table=*/nullptr,
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.total_sequence_length,
        parameters.kv_sequence_length, /*rotary_dim=*/0,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        /*past_bsnh=*/false,  // present cache is BNSH
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        /*local_window_size=*/-1, /*is_rotary_interleaved=*/false,
        /*is_packed_qkv=*/false));

    present_kv_already_populated = true;
  }
  // --- Path 3: Prompt flash (no past, no mask) ---
  // Note: prompt with bool mask is handled by MEA (flash_eligible excludes it).
  else {
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
        device_prop, cuda_stream,
        const_cast<void*>(q_data),
        const_cast<void*>(static_cast<const void*>(K->Data<T>())),
        const_cast<void*>(static_cast<const void*>(V->Data<T>())),
        out_data,
        softmax_lse_buffer.get(),
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.kv_sequence_length,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        is_bsnh));
  }

  // --- Transpose output BSNH → BNSH if input was 4D (BNSH) ---
  if (!is_bsnh && out_bsnh_buffer != nullptr) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.v_head_size,
          reinterpret_cast<const half*>(out_bsnh_buffer.get()),
          reinterpret_cast<half*>(Y->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.v_head_size,
          reinterpret_cast<const BFloat16*>(out_bsnh_buffer.get()),
          Y->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.v_head_size,
          reinterpret_cast<const float*>(out_bsnh_buffer.get()),
          Y->MutableData<float>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }

  // --- Populate present_key/value (BNSH) from K/V (BSNH) ---
  // Skip for decode path where mha_fwd_kvcache already populated present buffers.
  if (!present_kv_already_populated) {
    if (present_key != nullptr && is_bsnh) {
      if constexpr (std::is_same_v<T, MLFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            reinterpret_cast<const half*>(K->Data<T>()),
            reinterpret_cast<half*>(present_key->MutableData<T>()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            K->Data<BFloat16>(), present_key->MutableData<BFloat16>(),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.head_size,
            K->Data<float>(), present_key->MutableData<float>(),
            cuda_stream, device_prop.maxThreadsPerBlock));
      }
    }
    if (present_value != nullptr && is_bsnh) {
      if constexpr (std::is_same_v<T, MLFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            reinterpret_cast<const half*>(V->Data<T>()),
            reinterpret_cast<half*>(present_value->MutableData<T>()),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            V->Data<BFloat16>(), present_value->MutableData<BFloat16>(),
            cuda_stream, device_prop.maxThreadsPerBlock));
      } else {
        ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
            parameters.batch_size, parameters.kv_sequence_length,
            parameters.kv_num_heads, parameters.v_head_size,
            V->Data<float>(), present_value->MutableData<float>(),
            cuda_stream, device_prop.maxThreadsPerBlock));
      }
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(Q);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(V);
  ORT_UNUSED_PARAMETER(attn_mask);
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  ORT_UNUSED_PARAMETER(nonpad_kv_seqlen);
  ORT_UNUSED_PARAMETER(Y);
  ORT_UNUSED_PARAMETER(present_key);
  ORT_UNUSED_PARAMETER(present_value);
  ORT_UNUSED_PARAMETER(parameters);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Flash attention is not available in this build.");
#endif
}

// ============================================================================
// RunMemoryEfficientAttention: Direct memory-efficient attention kernel call
// ============================================================================
template <typename T>
Status Attention<T>::RunMemoryEfficientAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
#if USE_MEMORY_EFFICIENT_ATTENTION
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
  const bool is_bsnh = parameters.transpose_output;
  const int sm = device_prop.major * 10 + device_prop.minor;

  // Q/K/V pointers — MEA expects BSNH format for Q
  const void* q_data = Q->Data<T>();
  const void* k_data = K->Data<T>();
  const void* v_data = V->Data<T>();

  // --- Transpose Q from BNSH to BSNH if 4D input ---
  IAllocatorUniquePtr<void> q_bsnh_buffer;
  if (!is_bsnh) {
    size_t q_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                     parameters.q_num_heads * parameters.head_size;
    q_bsnh_buffer = GetScratchBuffer<void>(q_bytes, context->GetComputeStream());
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          reinterpret_cast<const half*>(Q->Data<T>()),
          reinterpret_cast<half*>(q_bsnh_buffer.get()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          Q->Data<BFloat16>(), reinterpret_cast<BFloat16*>(q_bsnh_buffer.get()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          Q->Data<float>(), reinterpret_cast<float*>(q_bsnh_buffer.get()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
    q_data = q_bsnh_buffer.get();
  }

  // MEA output is BSNH. If Y expects BNSH, write to scratch then transpose.
  void* out_data = Y->MutableData<T>();
  IAllocatorUniquePtr<void> out_bsnh_buffer;
  if (!is_bsnh) {
    size_t out_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                       parameters.q_num_heads * parameters.v_head_size;
    out_bsnh_buffer = GetScratchBuffer<void>(out_bytes, context->GetComputeStream());
    out_data = out_bsnh_buffer.get();
  }

  // GQA head expansion: MEA requires matching num_heads for Q/K/V.
  // When q_num_heads != kv_num_heads, expand K/V via LaunchUngroup.
  const bool is_gqa = parameters.q_num_heads != parameters.kv_num_heads;
  IAllocatorUniquePtr<void> k_expand_buffer;
  IAllocatorUniquePtr<void> v_expand_buffer;

  if (is_gqa) {
    // GQA+MEA only works with fp16/bf16 (MEA doesn't support fp32).
    // Use if constexpr to avoid instantiating LaunchUngroup<float> which has no explicit
    // template instantiation in group_query_attention_impl.cu.
    if constexpr (std::is_same_v<T, float>) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "GQA with Memory Efficient Attention requires fp16 or bf16, not fp32.");
    } else {
      ORT_ENFORCE(parameters.head_size == parameters.v_head_size,
                  "GQA with MEA requires head_size == v_head_size for LaunchUngroup.");
      ORT_ENFORCE(parameters.head_size % 4 == 0,
                  "GQA with MEA requires head_size divisible by 4 for LaunchUngroup (float2 access).");
      const size_t expanded_kv_elements = static_cast<size_t>(parameters.batch_size) *
                                          static_cast<size_t>(parameters.total_sequence_length) *
                                          static_cast<size_t>(parameters.q_num_heads) *
                                          static_cast<size_t>(parameters.head_size);
      k_expand_buffer = GetScratchBuffer<void>(expanded_kv_elements * sizeof(T), context->GetComputeStream());
      v_expand_buffer = GetScratchBuffer<void>(expanded_kv_elements * sizeof(T), context->GetComputeStream());

      onnxruntime::contrib::GroupQueryAttentionParameters ungroup_params = {};
      ungroup_params.batch_size = parameters.batch_size;
      ungroup_params.num_heads = parameters.q_num_heads;
      ungroup_params.kv_num_heads = parameters.kv_num_heads;
      ungroup_params.head_size = parameters.head_size;

      using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchUngroup<NativeCudaT>(
          ungroup_params,
          reinterpret_cast<float2*>(k_expand_buffer.get()),
          reinterpret_cast<float2*>(v_expand_buffer.get()),
          reinterpret_cast<const float2*>(k_data),
          reinterpret_cast<const float2*>(v_data),
          parameters.total_sequence_length,
          parameters.total_sequence_length,
          is_bsnh,
          cuda_stream,
          device_prop.maxThreadsPerBlock));

      k_data = k_expand_buffer.get();
      v_data = v_expand_buffer.get();
    }
  }

  // Note: MEA with past_key/value is handled by the unfused fallback.
  // The cascade in ComputeInternal ensures past_key == nullptr when we reach here.

  // Handle attention mask → attention_bias conversion
  IAllocatorUniquePtr<void> converted_mask_buffer;
  IAllocatorUniquePtr<void> nonpad_bias_buffer;
  const void* attn_bias_data = nullptr;
  bool broadcast_bias_dim_0 = false;
  bool broadcast_bias_dim_1 = false;

  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to seqlens_k for custom right padding.
    // MEA expects actual token count (not count-1), so use FlashSeqlensK variant.
    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(),
        seqlens_k_buffer.get(),
        parameters.batch_size,
        parameters.total_sequence_length,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    onnxruntime::contrib::cuda::MemoryEfficientAttentionParams p;
    p.sm = sm;
    p.is_half = std::is_same<T, MLFloat16>::value;
    p.is_bf16 = std::is_same<T, BFloat16>::value;
    p.is_kv_bsnh = is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    p.scale = parameters.scale;
    p.seqlen_k_ptr = seqlens_k_buffer.get();
    p.has_custom_right_padding = true;
    p.query = q_data;
    p.key = k_data;
    p.value = v_data;
    p.attn_bias = nullptr;
    p.stream = cuda_stream;
    p.output = out_data;

    IAllocatorUniquePtr<void> workspace_buffer;
    if (onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
            parameters.v_head_size, sizeof(T) == sizeof(float))) {
      size_t workspace_bytes = sizeof(float) * parameters.batch_size * parameters.q_sequence_length *
                               parameters.q_num_heads * parameters.v_head_size;
      workspace_buffer = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
      p.workspace = workspace_buffer.get();
    } else {
      p.workspace = nullptr;
    }
    onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
  }
  // Standard MEA path: float attention bias, bool mask (converted to bias), or no mask.
  // Bool masks are converted to additive attention bias (true→0, false→mask_filter_value)
  // which correctly handles all-false masks (uniform softmax weights) unlike the
  // custom_right_padding seqlens approach which would produce NaN.
  else {
    if (attn_mask != nullptr) {
      if (attn_mask->IsDataType<bool>()) {
        // Convert bool mask to additive attention bias (true→0.0, false→mask_filter_value).
        // This handles all-false masks correctly (uniform softmax weights from extreme bias).
        using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
        int64_t num_elements = attn_mask->Shape().Size();
        converted_mask_buffer = GetScratchBuffer<void>(
            num_elements * sizeof(NativeCudaT), context->GetComputeStream());
        float mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
        ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
            attn_mask->Data<bool>(),
            reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
            num_elements, mask_filter_value, cuda_stream,
            device_prop.maxThreadsPerBlock));
        attn_bias_data = converted_mask_buffer.get();
      } else {
        attn_bias_data = attn_mask->Data<T>();
      }

      // Determine broadcast flags based on bias logical shape [B, num_heads, q_seq, kv_seq].
      // MEA always uses bias_strideM = kv_seq, so each query row must have kv_seq elements.
      // For 2D masks [B, kv_seq]: the mask is constant across q positions, so we must
      // expand to [B, 1, q_seq, kv_seq] by repeating each row q_seq times. Without this,
      // bias_strideM would walk through batch boundaries instead of replaying the same mask.
      size_t mask_dims = attn_mask->Shape().NumDimensions();
      auto dims = attn_mask->Shape().GetDims();
      if (mask_dims == 2) {
        // Expand [B, kv_seq] → [B, 1, q_seq, kv_seq] by repeating each batch's row
        using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
        const int kv_len = parameters.total_sequence_length;
        const int q_len = parameters.q_sequence_length;
        int64_t expanded_elements = static_cast<int64_t>(parameters.batch_size) * q_len * kv_len;
        auto expanded_buffer = GetScratchBuffer<void>(
            expanded_elements * sizeof(NativeCudaT), context->GetComputeStream());
        const auto* src = (attn_mask->IsDataType<bool>())
                              ? reinterpret_cast<const NativeCudaT*>(converted_mask_buffer.get())
                              : reinterpret_cast<const NativeCudaT*>(attn_mask->Data<T>());
        auto* dst = reinterpret_cast<NativeCudaT*>(expanded_buffer.get());
        ORT_RETURN_IF_ERROR(LaunchBroadcastBias2DToQSeq<NativeCudaT>(
            src, dst, parameters.batch_size, q_len, kv_len,
            cuda_stream, device_prop.maxThreadsPerBlock));
        attn_bias_data = expanded_buffer.get();
        converted_mask_buffer = std::move(expanded_buffer);
        // Expanded shape is [B, 1, q_seq, kv_seq]
        broadcast_bias_dim_0 = false;
        broadcast_bias_dim_1 = true;
      } else if (mask_dims == 3) {
        broadcast_bias_dim_0 = true;
        broadcast_bias_dim_1 = dims[0] == 1;
      } else {
        broadcast_bias_dim_0 = dims[0] == 1;
        broadcast_bias_dim_1 = dims[1] == 1;
      }
    }

    onnxruntime::contrib::cuda::MemoryEfficientAttentionParams p;
    p.sm = sm;
    p.is_half = std::is_same<T, MLFloat16>::value;
    p.is_bf16 = std::is_same<T, BFloat16>::value;
    p.is_kv_bsnh = is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    p.scale = parameters.scale;
    p.broadcast_attn_bias_dim_0 = broadcast_bias_dim_0;
    p.broadcast_attn_bias_dim_1 = broadcast_bias_dim_1;
    p.query = q_data;
    p.key = k_data;
    p.value = v_data;
    p.attn_bias = attn_bias_data;
    p.stream = cuda_stream;
    p.output = out_data;

    if (onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
            parameters.v_head_size, sizeof(T) == sizeof(float))) {
      size_t workspace_bytes = sizeof(float) * parameters.batch_size * parameters.q_sequence_length *
                               parameters.q_num_heads * parameters.v_head_size;
      auto workspace_buffer = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
      p.workspace = workspace_buffer.get();
      onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
    } else {
      p.workspace = nullptr;
      onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
    }
  }

  // --- Transpose output BSNH → BNSH if input was 4D (BNSH) ---
  if (!is_bsnh && out_bsnh_buffer != nullptr) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.v_head_size,
          reinterpret_cast<const half*>(out_bsnh_buffer.get()),
          reinterpret_cast<half*>(Y->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.v_head_size,
          reinterpret_cast<const BFloat16*>(out_bsnh_buffer.get()),
          Y->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.q_num_heads, parameters.v_head_size,
          reinterpret_cast<const float*>(out_bsnh_buffer.get()),
          Y->MutableData<float>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }

  // Populate present_key/present_value (BNSH) if requested
  if (present_key != nullptr && is_bsnh) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          reinterpret_cast<const half*>(K->Data<T>()),
          reinterpret_cast<half*>(present_key->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<BFloat16>(), present_key->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<float>(), present_key->MutableData<float>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }
  if (present_value != nullptr && is_bsnh) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          reinterpret_cast<const half*>(V->Data<T>()),
          reinterpret_cast<half*>(present_value->MutableData<T>()),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<BFloat16>(), present_value->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<float>(), present_value->MutableData<float>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(Q);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(V);
  ORT_UNUSED_PARAMETER(attn_mask);
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  ORT_UNUSED_PARAMETER(nonpad_kv_seqlen);
  ORT_UNUSED_PARAMETER(Y);
  ORT_UNUSED_PARAMETER(present_key);
  ORT_UNUSED_PARAMETER(present_value);
  ORT_UNUSED_PARAMETER(parameters);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Memory efficient attention is not available in this build.");
#endif
}

// ============================================================================
// RunUnfusedAttention: Delegates to MHA's QkvToContext (unfused GEMM+softmax+GEMM)
// ============================================================================
template <typename T>
Status Attention<T>::RunUnfusedAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    Tensor* output_qk,
    const attention_helper::AttentionParameters& parameters) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

  // Bridge to contrib::AttentionParameters for the MHA unfused path
  onnxruntime::contrib::AttentionParameters contribop_parameters;

  if (!parameters.transpose_output) {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    contribop_parameters.is_output_bnsh = true;
  } else {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;
    contribop_parameters.is_output_bnsh = false;
  }

  contribop_parameters.batch_size = parameters.batch_size;
  contribop_parameters.sequence_length = parameters.q_sequence_length;
  contribop_parameters.kv_sequence_length = parameters.kv_sequence_length;
  contribop_parameters.past_sequence_length = parameters.past_sequence_length;
  contribop_parameters.total_sequence_length = parameters.total_sequence_length;
  contribop_parameters.max_sequence_length = parameters.total_sequence_length;
  contribop_parameters.input_hidden_size = 0;
  contribop_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
  contribop_parameters.head_size = parameters.head_size;
  contribop_parameters.v_head_size = parameters.v_head_size;
  contribop_parameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
  contribop_parameters.num_heads = parameters.q_num_heads;
  contribop_parameters.rotary_dim = 0;
  contribop_parameters.num_splits = 1;
  contribop_parameters.beam_width = 1;
  contribop_parameters.is_unidirectional = parameters.is_causal;
  contribop_parameters.past_present_share_buffer = false;
  contribop_parameters.is_packed_qkv = false;
  contribop_parameters.do_rotary = false;
  contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
  contribop_parameters.mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
  contribop_parameters.scale = parameters.scale;
  contribop_parameters.use_tf32 = UseTF32();

  // Determine broadcast flags for attention_bias
  if (attn_mask != nullptr) {
    size_t attn_mask_dims_size = attn_mask->Shape().NumDimensions();
    auto attn_mask_dims = attn_mask->Shape().GetDims();
    if (attn_mask_dims_size == 2) {
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = true;
    } else if (attn_mask_dims_size == 3) {
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[0] == 1;
    } else {
      contribop_parameters.broadcast_attn_bias_dim_0 = attn_mask_dims[0] == 1;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[1] == 1;
    }
  } else {
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = false;
  }

  // Construct AttentionData
  onnxruntime::contrib::cuda::AttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
  data.mask_index = nullptr;
  data.mask_index_dims = gsl::span<const int64_t>();
  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  if (output_qk != nullptr) {
    data.output_qk = reinterpret_cast<CudaT*>(output_qk->MutableData<T>());
  }
  data.bias = nullptr;

  // Handle attention mask / nonpad_kv_seqlen → attention_bias
  IAllocatorUniquePtr<void> converted_mask_buffer;
  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to additive attention bias: [B, q_seq, total_seq]
    using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
    int64_t bias_elements = static_cast<int64_t>(parameters.batch_size) *
                            parameters.q_sequence_length *
                            parameters.total_sequence_length;
    converted_mask_buffer = GetScratchBuffer<void>(bias_elements * sizeof(NativeCudaT), context->GetComputeStream());
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToAttentionBias<NativeCudaT>(
        nonpad_kv_seqlen->Data<int64_t>(),
        reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
        parameters.batch_size,
        parameters.q_sequence_length,
        parameters.total_sequence_length,
        contribop_parameters.mask_filter_value,
        cuda_stream,
        device_prop.maxThreadsPerBlock));
    data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
    // nonpad bias is [B, q_seq, total_seq] → broadcasts over heads but not batch
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = true;
  } else if (attn_mask != nullptr) {
    if (attn_mask->IsDataType<bool>()) {
      using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
      int64_t num_elements = attn_mask->Shape().Size();
      converted_mask_buffer = GetScratchBuffer<void>(num_elements * sizeof(NativeCudaT), context->GetComputeStream());
      ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
          attn_mask->Data<bool>(),
          reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
          num_elements,
          contribop_parameters.mask_filter_value,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
      data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
    } else {
      data.attention_bias = reinterpret_cast<const CudaT*>(attn_mask->Data<T>());
    }
  }

  data.qkv_format = contribop_parameters.qkv_format;
  data.use_flash_attention = false;
  data.use_memory_efficient_attention = false;
  data.fused_runner = nullptr;
  data.fused_cross_attention_kernel = nullptr;
  data.kernel_type = onnxruntime::contrib::AttentionKernelType::AttentionKernel_Unfused;

  // Allocate workspace
  const bool no_qkv_workspace = onnxruntime::contrib::cuda::NoQkvWorkspace(contribop_parameters, data);
  size_t workspace_bytes = onnxruntime::contrib::cuda::GetAttentionWorkspaceSize(
      sizeof(T),
      contribop_parameters.batch_size,
      contribop_parameters.num_heads,
      contribop_parameters.head_size,
      contribop_parameters.v_head_size,
      contribop_parameters.sequence_length,
      contribop_parameters.kv_sequence_length,
      contribop_parameters.total_sequence_length,
      nullptr, false, false, false, false, false,
      no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;

  cublasHandle_t cublas = GetCublasHandle(context);
  cudnnHandle_t cudnn = GetCudnnHandle(context);

  return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
      device_prop, cublas, cudnn, context->GetComputeStream(), contribop_parameters, data);
}

// ============================================================================
// ComputeInternal: Dispatch to appropriate attention kernel
// ============================================================================
// MHA path (q_num_heads == kv_num_heads): uses direct kernel dispatch cascade
//   flash → memory efficient → unfused
// GQA path (q_num_heads != kv_num_heads): routes through GQA dispatch (kept for now)
// ============================================================================
template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);
  const Tensor* nonpad_kv_seqlen = context->Input<Tensor>(6);  // optional, Opset 24

  ORT_ENFORCE(nonpad_kv_seqlen == nullptr || attn_mask == nullptr,
              "nonpad_kv_seqlen and attn_mask cannot both be provided.");

  attention_helper::AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                  Q, K, V, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                  is_causal_, softcap_, softmax_precision_,
                  qk_matmul_output_mode_, kv_num_heads_, q_num_heads_, scale_,
                  parameters, y_shape, present_key_shape, present_value_shape, output_qk_shape,
                  true /* skip_nonpad_data_validation: data is on GPU */)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);

  const bool is_gqa = parameters.kv_num_heads != parameters.q_num_heads;

  // === KERNEL SELECTION CASCADE ===
  // Priority: flash attention > memory efficient attention > unfused attention
  const bool has_output_qk = (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone);

#if USE_FLASH_ATTENTION
  {
    auto& device_prop = GetDeviceProp();
    bool flash_eligible =
        !std::is_same<T, float>::value &&
        onnxruntime::flash::is_supported<T>(device_prop, parameters.head_size,
                                            parameters.q_num_heads, parameters.kv_num_heads) &&
        parameters.head_size == parameters.v_head_size &&
        !has_output_qk &&
        parameters.softcap == 0.0f &&
        parameters.softmax_precision == 0 &&
        // Bool masks without past_key (prompt) can't use flash because mha_fwd_kvcache's
        // causal semantics are decode-oriented (window offset by seqlens_k). For causal
        // prompt with padding, MEA handles it correctly via attention bias conversion.
        // Flash handles: no mask, decode with past (±mask), nonpad_kv_seqlen.
        (attn_mask == nullptr || (attn_mask->IsDataType<bool>() && past_key != nullptr));

    if (flash_eligible) {
      return RunFlashAttention(context, Q, K, V, attn_mask, past_key, past_value,
                               nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  {
    auto& device_prop = GetDeviceProp();
    int sm = device_prop.major * 10 + device_prop.minor;
    bool mea_eligible =
        onnxruntime::contrib::cuda::has_memory_efficient_attention(
            sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value,
            parameters.head_size, parameters.v_head_size) &&
        !has_output_qk &&
        parameters.softcap == 0.0f &&
        parameters.softmax_precision == 0 &&
        past_key == nullptr;

    // Cutlass FMHA requires bias strides to satisfy minimum alignment even in the
    // "unaligned" kernel path. When an attention mask is present without nonpad_kv_seqlen,
    // it becomes an additive bias with bias_strideM = total_sequence_length. Skip MEA if
    // this stride can't satisfy the kernel's minimum alignment requirement.
    if (mea_eligible && attn_mask != nullptr && nonpad_kv_seqlen == nullptr) {
      int min_bias_align = 1;
      if ((std::is_same<T, float>::value && sm >= 80) ||
          (!std::is_same<T, float>::value && sm >= 75)) {
        min_bias_align = 4;  // TensorOp on Sm80+ (float) or Sm75+ (fp16/bf16)
      } else if (!std::is_same<T, float>::value && sm >= 70) {
        min_bias_align = 2;  // TensorOp on Volta (fp16)
      }
      if (parameters.total_sequence_length % min_bias_align != 0) {
        mea_eligible = false;
      }
    }

    if (mea_eligible) {
      return RunMemoryEfficientAttention(context, Q, K, V, attn_mask, past_key, past_value,
                                         nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

  // Fallback: unfused attention
  // TODO: Support softcap and softmax_precision on CUDA kernels.
  // Currently rejected by all three kernel paths (flash, MEA, unfused).
  if (parameters.softcap != 0.0f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "softcap is not supported yet in Attention op (CUDA).");
  }
  if (parameters.softmax_precision != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "softmax_precision is not supported yet in Attention op (CUDA).");
  }
  // TODO: Support additional output_qk modes beyond kNone and kQK.
  // Currently only unfused handles output_qk, and only kNone/kQK modes.
  if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone &&
      qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kQK) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "qk_matmul_output_mode other than kNone and kQK is not supported yet "
                           "in Attention op (CUDA).");
  }

  if (is_gqa) {
    // TODO: Support GQA in unfused attention path for fp32/old-GPU fallback.
    // Requires ~160 lines: ExpandKVHeads kernel to replicate KV heads, wiring in unfused dispatch.
    // See issue #27516 for tracking.
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "GQA (q_num_heads != kv_num_heads) requires flash or memory efficient attention, "
                           "but neither is eligible. Ensure fp16/bf16 on Ampere+ GPU, or check head_size constraints.");
  }

  return RunUnfusedAttention(context, Q, K, V, attn_mask, past_key, past_value,
                             nonpad_kv_seqlen, Y, present_key, present_value, output_qk, parameters);
}

}  // namespace cuda
}  // namespace onnxruntime
