// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
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

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

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

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
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

#if USE_FLASH_ATTENTION
// Runs flash attention directly on an external KV cache (e.g., assembled by TensorScatter).
// Bypasses the contrib GQA kernel's PrepareQKV/ConcatNewToPastKV since the cache is already
// fully assembled. Converts nonpad_kv_seqlen to int32 seqlens_k internally and uses it for
// per-batch masking instead of attention_bias.
//
// Prerequisites:
//   - K/V are the full cache in BSNH format: [B, total_kv_seq, kv_heads, head_size]
//   - nonpad_kv_seqlen is provided (int64, GPU, shape [batch_size])
//   - Flash attention is supported on this GPU (SM >= 8.0, fp16/bf16)
//   - T is NOT float (flash attention requires fp16/bf16)
template <typename T>
Status Attention<T>::FlashAttentionForExternalKVCache(
    const cudaDeviceProp& device_prop,
    cudaStream_t cuda_stream,
    const Tensor* Q,
    const Tensor* K,
    const Tensor* V,
    Tensor* Y,
    Tensor* present_key,
    Tensor* present_value,
    const Tensor* nonpad_kv_seqlen,
    const attention_helper::AttentionParameters& parameters,
    bool is_bf16,
    onnxruntime::Stream* ort_stream) const {

  // The full KV cache must be passed as K/V directly (no past_key concatenation).
  // If past_sequence_length != 0, K/V would be partial and flash would read OOB.
  if (parameters.past_sequence_length != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "FlashAttentionForExternalKVCache requires K/V to be the full cache "
                           "(past_sequence_length must be 0, got ", parameters.past_sequence_length, ").");
  }

  // Convert nonpad_kv_seqlen (int64 count) to int32 seqlens_k for flash attention.
  // Flash's mha_fwd_kvcache expects the actual token count, not the GQA count-1 convention.
  auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, ort_stream);
  ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
      nonpad_kv_seqlen->Data<int64_t>(),
      seqlens_k_buffer.get(),
      parameters.batch_size,
      parameters.total_sequence_length,
      cuda_stream,
      device_prop.maxThreadsPerBlock));

  // Allocate softmax_lse buffer (required by flash attention)
  size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(
      parameters.q_sequence_length, parameters.batch_size, parameters.q_num_heads);

  // Compute num_splits and accumulation buffer sizes
  auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      onnxruntime::flash::get_num_splits_and_buffer_sizes(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.kv_sequence_length,
          parameters.q_num_heads, parameters.head_size,
          device_prop.multiProcessorCount);

  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, ort_stream);
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, ort_stream);
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, ort_stream);

  if (softmax_lse_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(softmax_lse_accum_buffer.get(), 0,
                                         softmax_lse_accum_bytes, cuda_stream));
  }
  if (out_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(out_accum_buffer.get(), 0,
                                         out_accum_bytes, cuda_stream));
  }

  // Call mha_fwd_kvcache with the full external KV cache.
  // K/V are BSNH: [B, total_kv_seq, kv_heads, head_size]
  // Q is BSNH: [B, q_seq, q_heads, head_size]
  // No new tokens to append (k=nullptr, v=nullptr) since the cache is fully assembled.
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
      device_prop,
      cuda_stream,
      const_cast<void*>(static_cast<const void*>(Q->Data<T>())),
      const_cast<void*>(static_cast<const void*>(K->Data<T>())),
      const_cast<void*>(static_cast<const void*>(V->Data<T>())),
      /*k=*/nullptr,
      /*v=*/nullptr,
      static_cast<void*>(Y->MutableData<T>()),
      /*softmax_lse=*/softmax_lse_buffer.get(),
      /*seqlens_k=*/const_cast<void*>(static_cast<const void*>(seqlens_k_buffer.get())),
      /*rotary_cos=*/nullptr,
      /*rotary_sin=*/nullptr,
      /*cache_batch_idx=*/nullptr,
      /*leftpad_k=*/nullptr,
      /*head_sink=*/nullptr,
      /*block_table=*/nullptr,
      parameters.batch_size,
      parameters.q_num_heads,
      parameters.kv_num_heads,
      parameters.head_size,
      /*seqlen_q=*/parameters.q_sequence_length,
      /*seqlen_k=*/parameters.kv_sequence_length,
      /*seqlen_k_new=*/0,
      /*rotary_dim=*/0,
      /*softmax_scale=*/parameters.scale,
      /*softcap=*/parameters.softcap,
      parameters.is_causal,
      is_bf16,
      /*use_smooth_softmax=*/false,
      /*past_bsnh=*/true,
      static_cast<int>(num_splits),
      softmax_lse_accum_buffer.get(),
      out_accum_buffer.get(),
      /*local_window_size=*/-1,
      /*is_rotary_interleaved=*/false,
      /*is_packed_qkv=*/false));

  // Populate present_key/present_value outputs if requested.
  // K/V are BSNH: [B, S, kv_heads, head_size]
  // present_key/value are BNSH: [B, kv_heads, S, head_size]
  if (present_key != nullptr) {
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
          K->Data<BFloat16>(),
          present_key->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }
  if (present_value != nullptr) {
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
          V->Data<BFloat16>(),
          present_value->MutableData<BFloat16>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    }
  }

  return Status::OK();
}
#endif  // USE_FLASH_ATTENTION

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);
  const Tensor* nonpad_kv_seqlen = context->Input<Tensor>(6);  // optional, Opset 24

  attention_helper::AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

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
                  output_qk_shape,
                  true /* skip_nonpad_data_validation: data is on GPU */)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  // Note: parameters.nonpad_kv_seqlen_data is set by ComputeOutputShapeForAttention but is a
  // device pointer on CUDA — it must not be dereferenced on host. The CUDA path reads the tensor
  // directly via nonpad_kv_seqlen->Data<int64_t>() when launching GPU kernels.
  // Only the CPU path uses parameters.nonpad_kv_seqlen_data for per-element masking.

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);

  // To reuse the existing attention-cuda implementation in contrib ops,
  // map the parameters to contribop_parameters (MHA).
  onnxruntime::contrib::AttentionParameters contribop_parameters;

  // QKV format: Determine based on input dimensions
  // 3D inputs (B, S, D): Q_K_V_BSNH - will be transposed by PrepareQkv to BNSH
  // transpose_output is true for 3D inputs, false for 4D inputs
  if (!parameters.transpose_output) {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    contribop_parameters.is_output_bnsh = true;
  } else {
    // 3D inputs in BSNH format (will be transposed)
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;
    contribop_parameters.is_output_bnsh = false;
  }

  // Check if this is Group Query Attention (GQA)
  const bool is_gqa = parameters.kv_num_heads != parameters.q_num_heads;

  if (is_gqa) {
    // Use GQA path with Flash Attention or Memory Efficient Attention
    // GQA only supports float16 and bfloat16 types
    if (std::is_same<T, float>::value) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GQA in Attention op (CUDA) does not support float32. "
                             "Please use float16 or bfloat16.");
    }
    // GQA only supports 3D inputs (B, S, D) in BSNH format, not 4D inputs (B, num_heads, S, head_size) in BNSH format
    if (!parameters.transpose_output) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "4D QKV inputs (BNSH format) are not supported yet in GQA path of Attention op (CUDA). "
                             "Please use 3D inputs (B, S, hidden_size) instead.");
    }

    // External KV cache path: when nonpad_kv_seqlen is provided and kv_seq > q_seq,
    // the full KV cache is passed directly (e.g., from TensorScatter).
    // Bypass the GQA kernel's ConcatNewToPastKV and call flash attention directly.
    // This check is before the is_causal/qk_matmul_output_mode/softmax_precision guards
    // because our flash path supports non-causal attention and doesn't use those features.
    if (parameters.kv_sequence_length != parameters.q_sequence_length) {
      if (!parameters.has_nonpad_kv_seqlen) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Cross-attention (kv_sequence_length != q_sequence_length) without "
                               "nonpad_kv_seqlen is not supported in GQA path of Attention op (CUDA). "
                               "kv_sequence_length=", parameters.kv_sequence_length,
                               ", q_sequence_length=", parameters.q_sequence_length);
      }
#if USE_FLASH_ATTENTION
      auto& device_prop_ext = GetDeviceProp();
      if (!onnxruntime::flash::is_supported<T>(device_prop_ext, parameters.head_size,
                                                parameters.q_num_heads, parameters.kv_num_heads)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Flash attention is not supported on this device for the external KV cache "
                               "GQA path. Requires SM >= 8.0, fp16/bf16, head_size <= 256.");
      }

      bool is_bf16 = std::is_same<T, BFloat16>::value;
      auto cuda_stream_ext = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
      return FlashAttentionForExternalKVCache(
          device_prop_ext, cuda_stream_ext,
          Q, K, V, Y, present_key, present_value,
          nonpad_kv_seqlen, parameters, is_bf16,
          context->GetComputeStream());
#else
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Flash attention is required for external KV cache GQA path but is not available.");
#endif
    }

    // Standard GQA path (self-attention: kv_seq == q_seq) requires these constraints
    // For now, GQA doesn't support qk_matmul_output_mode other than kNone
    if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "qk_matmul_output_mode is not supported yet in GQA path of Attention op (CUDA).");
    }
    // GQA doesn't support softmax_precision yet
    if (parameters.softmax_precision != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "softmax_precision is not supported yet in GQA path of Attention op (CUDA).");
    }
    // causal attention is required for GQA
    if (!parameters.is_causal) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Non-causal attention is not supported yet in GQA path of Attention op (CUDA).");
    }

    auto& device_prop = GetDeviceProp();

    // Bridge parameters to GroupQueryAttentionParameters
    onnxruntime::contrib::GroupQueryAttentionParameters gqa_parameters;
    gqa_parameters.batch_size = parameters.batch_size;
    gqa_parameters.sequence_length = parameters.q_sequence_length;
    gqa_parameters.seqlen_past_kv_cache = parameters.past_sequence_length;
    gqa_parameters.seqlen_present_kv_cache = parameters.total_sequence_length;
    gqa_parameters.total_sequence_length = parameters.total_sequence_length;
    gqa_parameters.kv_sequence_length = parameters.kv_sequence_length;
    gqa_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
    gqa_parameters.num_heads = parameters.q_num_heads;
    gqa_parameters.head_size = parameters.head_size;
    gqa_parameters.v_head_size = parameters.v_head_size;
    gqa_parameters.kv_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
    gqa_parameters.kv_num_heads = parameters.kv_num_heads;
    gqa_parameters.scale = parameters.scale;
    gqa_parameters.softcap = parameters.softcap;
    gqa_parameters.qkv_format = contribop_parameters.qkv_format;

    // Unset or set to default values for GQA-specific fields
    gqa_parameters.rotary_dim = 0;            // New Attention op doesn't use rotary embeddings directly
    gqa_parameters.is_unidirectional = true;  // GQA requires causal attention
    gqa_parameters.is_packed_qkv = false;     // New Attention op has separate Q, K, V inputs
    gqa_parameters.is_subsequent_prompt = false;
    gqa_parameters.is_first_prompt = parameters.past_sequence_length == 0;
    gqa_parameters.do_rotary = false;  // New Attention op doesn't use rotary embeddings
    gqa_parameters.rotary_interleaved = false;
    gqa_parameters.use_smooth_softmax = false;
    gqa_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
    gqa_parameters.past_kv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    gqa_parameters.local_window_size = -1;  // No local window for standard attention
    gqa_parameters.zeros_count = 0;
    gqa_parameters.zero_ptr = nullptr;
    gqa_parameters.num_splits = 1;

    // Construct GroupQueryAttentionData
    typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
    onnxruntime::contrib::cuda::GroupQueryAttentionData<CudaT, CudaT> gqa_data;

    // Scratch buffers for flash/memory efficient attention
    IAllocatorUniquePtr<void> k_buffer;
    IAllocatorUniquePtr<void> v_buffer;
    IAllocatorUniquePtr<void> fmha_buffer;
    IAllocatorUniquePtr<void> unpacked_qkv_buffer;
    IAllocatorUniquePtr<int> seq_lens_buffer;
    IAllocatorUniquePtr<int> seqlens_k_buffer;

    // Present KV cache buffers - GQA kernel uses these as working buffers
    // If outputs are not provided, we allocate scratch buffers
    IAllocatorUniquePtr<void> present_key_scratch;
    IAllocatorUniquePtr<void> present_value_scratch;

    // Set input pointers
    gqa_data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
    gqa_data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
    gqa_data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
    gqa_data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    gqa_data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());

    // Set output pointers
    gqa_data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());

    // GQA kernel requires present_key/present_value buffers as working storage for KV cache
    // Allocate scratch buffers if outputs are not provided
    size_t present_kv_size = static_cast<size_t>(parameters.batch_size) *
                             static_cast<size_t>(parameters.kv_num_heads) *
                             static_cast<size_t>(parameters.total_sequence_length) *
                             static_cast<size_t>(parameters.head_size) * sizeof(CudaT);
    if (present_key != nullptr) {
      gqa_data.present_key = reinterpret_cast<CudaT*>(present_key->MutableData<T>());
    } else {
      present_key_scratch = GetScratchBuffer<void>(present_kv_size, context->GetComputeStream());
      gqa_data.present_key = reinterpret_cast<CudaT*>(present_key_scratch.get());
    }
    if (present_value != nullptr) {
      gqa_data.present_value = reinterpret_cast<CudaT*>(present_value->MutableData<T>());
    } else {
      present_value_scratch = GetScratchBuffer<void>(present_kv_size, context->GetComputeStream());
      gqa_data.present_value = reinterpret_cast<CudaT*>(present_value_scratch.get());
    }

    // Compute past_present_share_buffer early since it's needed for flash attention path selection
    gqa_parameters.past_present_share_buffer = (gqa_data.past_key == gqa_data.present_key);

    // Flash Attention buffers
    IAllocatorUniquePtr<void> softmax_lse_buffer;
    IAllocatorUniquePtr<void> softmax_lse_accum_buffer;
    IAllocatorUniquePtr<void> out_accum_buffer;

    // Check Flash Attention support
#if USE_FLASH_ATTENTION
    bool use_flash_attention = onnxruntime::flash::is_supported<T>(device_prop,
                                                                   gqa_parameters.head_size,
                                                                   gqa_parameters.num_heads,
                                                                   gqa_parameters.kv_num_heads);

    gqa_data.use_flash_attention = use_flash_attention;
    gqa_data.use_flash_attention_fast_decode = use_flash_attention &&
                                               !gqa_parameters.is_first_prompt &&
                                               gqa_parameters.past_present_share_buffer;

    if (use_flash_attention) {
      // Allocate Flash specific buffers (Softmax LSE, Accum)
      size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(
          gqa_parameters.sequence_length, gqa_parameters.batch_size, gqa_parameters.num_heads);

      int num_heads_for_split = gqa_data.use_flash_attention_fast_decode
                                    ? gqa_parameters.kv_num_heads
                                    : gqa_parameters.num_heads;
      auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
          onnxruntime::flash::get_num_splits_and_buffer_sizes(
              gqa_parameters.batch_size, gqa_parameters.sequence_length,
              gqa_parameters.total_sequence_length, num_heads_for_split,
              gqa_parameters.head_size, device_prop.multiProcessorCount);

      gqa_parameters.num_splits = static_cast<int>(num_splits);

      if (gqa_data.use_flash_attention_fast_decode && num_splits > 1) {
        // The heuristic used kv_num_heads to maximize occupancy for the GQA-aware kernel.
        // However, the LSE and Accum buffers must store results for ALL num_heads.
        softmax_lse_accum_bytes = onnxruntime::flash::get_softmax_lse_accum_size(
            num_splits, gqa_parameters.batch_size, gqa_parameters.num_heads, gqa_parameters.sequence_length);
        auto round_multiple = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
        out_accum_bytes = onnxruntime::flash::get_out_accum_size(
            num_splits, gqa_parameters.batch_size, gqa_parameters.num_heads, gqa_parameters.sequence_length,
            round_multiple(gqa_parameters.head_size, 32));
      }

      softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
      softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
      out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

      gqa_data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
      gqa_data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
      gqa_data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
    } else {
      gqa_data.softmax_lse = nullptr;
      gqa_data.softmax_lse_accum = nullptr;
      gqa_data.out_accum = nullptr;
    }
#else
    gqa_data.use_flash_attention = false;
    gqa_data.use_flash_attention_fast_decode = false;
    gqa_data.softmax_lse = nullptr;
    gqa_data.softmax_lse_accum = nullptr;
    gqa_data.out_accum = nullptr;
#endif

    // Check Memory Efficient Attention support (fallback if flash attention not available)
#if USE_MEMORY_EFFICIENT_ATTENTION
    if (!gqa_data.use_flash_attention) {
      int sm = (device_prop.major * 10) + device_prop.minor;
      bool use_memory_efficient_attention =
          onnxruntime::contrib::cuda::has_memory_efficient_attention(
              sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value,
              gqa_parameters.head_size, gqa_parameters.head_size);
      gqa_data.use_memory_efficient_attention = use_memory_efficient_attention;

      // KV buffer for head expansion (when num_heads != kv_num_heads)
      size_t kv_buffer_bytes = (use_memory_efficient_attention &&
                                (gqa_parameters.num_heads != gqa_parameters.kv_num_heads))
                                   ? (sizeof(T) * gqa_parameters.batch_size * gqa_parameters.num_heads *
                                      gqa_parameters.seqlen_present_kv_cache * gqa_parameters.head_size)
                                   : 0;
      // FMHA workspace
      size_t fmha_buffer_bytes =
          (use_memory_efficient_attention &&
           onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
               gqa_parameters.head_size, sizeof(T) == sizeof(float)))
              ? (sizeof(float) * gqa_parameters.batch_size * gqa_parameters.sequence_length *
                 gqa_parameters.num_heads * gqa_parameters.head_size)
              : 0;

      k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
      v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
      fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());

      gqa_data.k = reinterpret_cast<CudaT*>(k_buffer.get());
      gqa_data.v = reinterpret_cast<CudaT*>(v_buffer.get());
      gqa_data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
    } else {
      gqa_data.use_memory_efficient_attention = false;
      gqa_data.k = nullptr;
      gqa_data.v = nullptr;
      gqa_data.fmha_buffer = nullptr;
    }
#else
    gqa_data.use_memory_efficient_attention = false;
    gqa_data.k = nullptr;
    gqa_data.v = nullptr;
    gqa_data.fmha_buffer = nullptr;
#endif

    // Centralized scratch buffer allocation using GQABufferRequirements
    auto buffer_req = onnxruntime::contrib::cuda::GQABufferRequirements::Compute<T>(
        gqa_parameters,
        false,  // use_xqa
        gqa_data.use_flash_attention,
        gqa_data.use_flash_attention_fast_decode,
        gqa_data.use_memory_efficient_attention);

    if (buffer_req.qkv_buffer_bytes > 0) {
      unpacked_qkv_buffer = GetScratchBuffer<void>(buffer_req.qkv_buffer_bytes, context->GetComputeStream());
      gqa_data.qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
    } else {
      gqa_data.qkv_buffer = nullptr;
    }

    // Allocate GPU buffer for seqlens_k (total_sequence_length - 1) for GQA compatibility
    // The GQA kernel expects sequence length information for flash/memory efficient attention
    seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
    auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

    // GQA only supports masking, not additive bias.
    // For bool mask, we need to convert it to sequence lengths on GPU.
    // Note: The GQA path interprets 2D bool masks as (batch_size, total_seq_len) since it converts
    // masks to seqlens_k directly (bypassing ONNX right-aligned broadcasting). This differs from
    // the MHA path below, where 2D masks follow ONNX broadcasting: [A, B] → [1, 1, A, B], so
    // 2D = (q_seq_len, total_seq_len) with both batch and heads broadcast.
    if (parameters.has_nonpad_kv_seqlen) {
      if (gqa_parameters.is_first_prompt) {
        // GQA prompt mode does not support nonpad_kv_seqlen masking. The GQA flash/efficient
        // attention kernels use padded_seq_lens (hardcoded to sequence_length) instead of
        // total_seq_lens in prompt mode, so seqlens_k doesn't affect masking.
        // Reject this combination to prevent silent incorrect behavior.
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "GQA prompt mode (q_sequence_length == kv_sequence_length) does not support "
                               "partial KV masking via nonpad_kv_seqlen on CUDA. The GQA kernel ignores "
                               "seqlens_k in prompt mode. Use MHA (set q_num_heads == kv_num_heads) for "
                               "partial masking, or ensure nonpad_kv_seqlen is not passed in GQA prompt mode.");
      } else {
        // Convert nonpad_kv_seqlen (int64, GPU) to seqlens_k (int32, GPU).
        // GQA convention: seqlens_k[i] = nonpad_kv_seqlen[i] - 1 (last valid index, not count).
        ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToSeqlensK(
            nonpad_kv_seqlen->Data<int64_t>(),
            seqlens_k_buffer.get(),
            parameters.batch_size,
            parameters.total_sequence_length,
            cuda_stream,
            device_prop.maxThreadsPerBlock));
      }
    } else if (attn_mask != nullptr && attn_mask->IsDataType<bool>()) {
      // Get mask dimensions for broadcasting
      // attn_mask can be 2D, 3D, or 4D and broadcasts to (batch_size, num_heads, q_seq_len, total_seq_len)
      const auto& mask_shape = attn_mask->Shape();
      int mask_dims = static_cast<int>(mask_shape.NumDimensions());
      int64_t mask_dim0 = 0, mask_dim1 = 0, mask_dim2 = 0;

      if (mask_dims == 2) {
        // Shape: (batch_size or 1, total_seq_len)
        mask_dim0 = mask_shape[0];
        mask_dim1 = 0;
        mask_dim2 = 0;
      } else if (mask_dims == 3) {
        // Shape: (num_heads or 1, q_seq_len, total_seq_len)
        mask_dim0 = mask_shape[0];
        mask_dim1 = mask_shape[1];
        mask_dim2 = 0;
      } else if (mask_dims == 4) {
        // Shape: (batch_size or 1, num_heads or 1, q_seq_len, total_seq_len)
        mask_dim0 = mask_shape[0];
        mask_dim1 = mask_shape[1];
        mask_dim2 = mask_shape[2];
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Boolean attn_mask must be 2D, 3D, or 4D. Got ", mask_dims, "D.");
      }

      // Launch CUDA kernel to convert mask to seqlens_k.
      // Mask validity (right-padding, contiguous) is checked asynchronously via CUDA_KERNEL_ASSERT.
      ORT_RETURN_IF_ERROR(LaunchConvertMaskToSeqlensK(
          attn_mask->Data<bool>(),
          seqlens_k_buffer.get(),
          parameters.batch_size,
          parameters.total_sequence_length,
          mask_dims,
          mask_dim0,
          mask_dim1,
          mask_dim2,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
    } else if (attn_mask != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Non-boolean attn_mask is not supported yet in GQA path of Attention op (CUDA).");
    } else {
      // No mask provided - use full sequence length for all batches
      // seqlens_k is total_sequence_length - 1 for historical reasons (matching GroupQueryAttention convention)
      // Fill on GPU using cudaMemset-like approach or a simple kernel
      std::vector<int> seqlens_k_host(parameters.batch_size, parameters.total_sequence_length - 1);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(seqlens_k_buffer.get(), seqlens_k_host.data(),
                                           sizeof(int) * parameters.batch_size,
                                           cudaMemcpyHostToDevice, cuda_stream));
    }

    // Process seqlens_k to compute past_seq_lens, total_seq_lens, and padded_seq_lens
    // This is always needed for flash/memory efficient attention
    seq_lens_buffer = GetScratchBuffer<int>(3 * parameters.batch_size, context->GetComputeStream());
    gqa_data.past_seq_lens = seq_lens_buffer.get();
    gqa_data.total_seq_lens = seq_lens_buffer.get() + parameters.batch_size;
    gqa_data.padded_seq_lens = gqa_data.total_seq_lens + parameters.batch_size;

    ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchGetSequenceLengths(
        seqlens_k_buffer.get(),
        gqa_data.past_seq_lens,
        gqa_data.total_seq_lens,
        gqa_data.padded_seq_lens,
        parameters.batch_size,
        parameters.q_sequence_length,
        gqa_parameters.is_first_prompt,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    // Set GQA-specific fields
    gqa_data.cos_cache = nullptr;  // No rotary embeddings
    gqa_data.sin_cache = nullptr;
    gqa_data.head_sink = nullptr;
    gqa_data.position_ids = nullptr;

    // Call GQA kernel (with flash or memory efficient attention)
    cublasHandle_t cublas = GetCublasHandle(context);

    return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
        device_prop, cublas, context->GetComputeStream(), gqa_parameters, gqa_data);
  } else {  // MHA path (kv_num_heads == q_num_heads)
    typedef typename ToCudaType<T>::MappedType CudaT;
    contribop_parameters.batch_size = parameters.batch_size;
    contribop_parameters.sequence_length = parameters.q_sequence_length;
    contribop_parameters.kv_sequence_length = parameters.kv_sequence_length;
    contribop_parameters.past_sequence_length = parameters.past_sequence_length;
    contribop_parameters.total_sequence_length = parameters.total_sequence_length;
    // max_sequence_length: For non-buffer-sharing case, this equals total_sequence_length (the present KV cache size)
    contribop_parameters.max_sequence_length = parameters.total_sequence_length;
    contribop_parameters.input_hidden_size = 0;  // Not applicable - new Attention op takes pre-projected Q/K/V
    contribop_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
    contribop_parameters.head_size = parameters.head_size;
    contribop_parameters.v_head_size = parameters.v_head_size;
    contribop_parameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
    contribop_parameters.num_heads = parameters.q_num_heads;
    contribop_parameters.rotary_dim = 0;
    contribop_parameters.num_splits = 1;
    contribop_parameters.beam_width = 1;
    contribop_parameters.is_unidirectional = parameters.is_causal;
    contribop_parameters.past_present_share_buffer = false;  // New Attention op doesn't share buffer
    contribop_parameters.is_packed_qkv = false;
    contribop_parameters.do_rotary = false;

    // The new Attention op uses attn_mask as attention_bias (additive bias), not as key_padding_mask
    // So mask_type should always be MASK_NONE since we don't have a separate padding mask input
    contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;

    // Determine broadcast flags for attention_bias (if it exists)
    // The MHA path uses attn_mask as attention_bias (additive bias added before softmax).
    // Bool masks are element-wise converted to additive bias (true → 0.0, false → -inf),
    // preserving the original shape, so the same broadcasting rules apply to both types.
    //
    // ONNX broadcasting is right-aligned to target shape (batch, heads, q_seq, total_seq):
    //   2D [A, B]       → [1, 1, A, B]    : A = q_seq_len, B = total_seq_len
    //   3D [A, B, C]    → [1, A, B, C]    : A = heads, B = q_seq_len, C = total_seq_len
    //   4D [A, B, C, D] → [A, B, C, D]    : A = batch, B = heads, C = q_seq_len, D = total_seq_len
    //
    // Note: A 2D mask cannot represent per-batch padding because the batch dimension is broadcast.
    // For per-batch boolean padding masks, use 4D shape (batch, 1, 1, total_seq_len).
    if (attn_mask != nullptr) {
      size_t attn_mask_dims_size = attn_mask->Shape().NumDimensions();
      auto attn_mask_dims = attn_mask->Shape().GetDims();
      // For 2D mask (q_seq_len, total_seq_len): both batch and heads dimensions need broadcasting
      // For 3D mask (heads_or_1, q_seq_len, total_seq_len): batch always broadcasts, heads broadcasts if dim[0]==1
      // For 4D mask (B, H, q_seq_len, total_seq_len): check if B==1 and H==1

      if (attn_mask_dims_size == 2) {
        // 2D mask: both dimensions need broadcasting
        contribop_parameters.broadcast_attn_bias_dim_0 = true;
        contribop_parameters.broadcast_attn_bias_dim_1 = true;
      } else if (attn_mask_dims_size == 3) {
        // 3D mask [A, q_seq_len, total_seq_len]: right-aligned to [_, A, q_seq, total_seq]
        // A maps to heads dimension (validated to be 1 or q_num_heads by attention_helper.h)
        // Batch dimension is missing, so always broadcasts
        contribop_parameters.broadcast_attn_bias_dim_0 = true;
        contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[0] == 1;
      } else {
        // 4D mask: check both dim 0 and dim 1 explicitly
        contribop_parameters.broadcast_attn_bias_dim_0 = attn_mask_dims[0] == 1;
        contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[1] == 1;
      }
    } else {
      contribop_parameters.broadcast_attn_bias_dim_0 = false;
      contribop_parameters.broadcast_attn_bias_dim_1 = false;
    }

    contribop_parameters.mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
    contribop_parameters.scale = parameters.scale;
    contribop_parameters.use_tf32 = UseTF32();
    // TODO(titaiwang, xadupre): qk_matmul_output_mode only supports kNone and kQK for now
    if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone &&
        qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kQK) {
      ORT_THROW("qk_matmul_output_mode other than -1 (None) and 0 (QK) is not supported yet in Attention op (CUDA).");
    }
    // TODO(titaiwang, xadupre): softcap and softmax_precision are not used yet
    if (parameters.softcap != 0.0f) {
      ORT_THROW("softcap is not supported yet in Attention op (CUDA).");
    }
    if (parameters.softmax_precision != 0) {
      ORT_THROW("softmax_precision is not supported yet in Attention op (CUDA).");
    }

    // Construct AttentionData to pass to QkvToContext
    onnxruntime::contrib::cuda::AttentionData<CudaT> data;

    // Set input pointers
    data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
    data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
    data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
    data.mask_index = nullptr;  // New Attention op doesn't have key_padding_mask
    data.mask_index_dims = gsl::span<const int64_t>();
    data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());

    // Set output pointers
    data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
    data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
    data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
    if (nullptr != output_qk) {
      data.output_qk = reinterpret_cast<CudaT*>(output_qk->MutableData<T>());
    }

    // Set additional fields
    data.bias = nullptr;  // New Attention op doesn't have bias
    IAllocatorUniquePtr<void> converted_mask_buffer;
    IAllocatorUniquePtr<void> nonpad_kv_bias_buffer;
    if (parameters.has_nonpad_kv_seqlen) {
      if (attn_mask != nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Using both nonpad_kv_seqlen and attn_mask simultaneously is not yet supported "
                               "in MHA path of Attention op (CUDA).");
      }

      // Flash attention requires BSNH (3D) inputs; 4D BNSH inputs have different strides.
      if (!parameters.transpose_output) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "4D BNSH inputs with nonpad_kv_seqlen are not supported in the flash path.");
      }

      // Try flash attention for external KV cache (fp16/bf16 only)
#if USE_FLASH_ATTENTION
      if (!std::is_same<T, float>::value) {
        auto& device_prop_mha = GetDeviceProp();
        bool flash_supported = onnxruntime::flash::is_supported<T>(
            device_prop_mha, parameters.head_size,
            parameters.q_num_heads, parameters.kv_num_heads);

        if (flash_supported) {
          bool is_bf16 = std::is_same<T, BFloat16>::value;
          auto cuda_stream_mha = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
          return FlashAttentionForExternalKVCache(
              device_prop_mha, cuda_stream_mha,
              Q, K, V, Y, present_key, present_value,
              nonpad_kv_seqlen, parameters, is_bf16,
              context->GetComputeStream());
        }
      }
#endif

      // Fallback: fp32 or flash not available — use attention_bias + unfused path.
      // Guard against excessive memory usage in unfused O(n²) attention.
      using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
      int64_t bias_elements = static_cast<int64_t>(parameters.batch_size) *
                              parameters.q_sequence_length *
                              parameters.total_sequence_length;
      constexpr int64_t kMaxUnfusedBiasBytes = static_cast<int64_t>(128) * 1024 * 1024;
      int64_t bias_bytes = bias_elements * static_cast<int64_t>(sizeof(NativeCudaT));
      if (bias_bytes > kMaxUnfusedBiasBytes) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Unfused attention fallback for nonpad_kv_seqlen would allocate ", bias_bytes,
                               " bytes for the attention bias (batch=", parameters.batch_size,
                               ", q_seq=", parameters.q_sequence_length,
                               ", total_seq=", parameters.total_sequence_length,
                               "). This exceeds the ", kMaxUnfusedBiasBytes,
                               "-byte limit and would likely cause OOM. "
                               "Use float16 or bfloat16 to enable flash attention for this workload.");
      }

      nonpad_kv_bias_buffer = GetScratchBuffer<void>(bias_bytes, context->GetComputeStream());
      auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
      ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToAttentionBias<NativeCudaT>(
          nonpad_kv_seqlen->Data<int64_t>(),
          reinterpret_cast<NativeCudaT*>(nonpad_kv_bias_buffer.get()),
          parameters.batch_size,
          parameters.q_sequence_length,
          parameters.total_sequence_length,
          contribop_parameters.mask_filter_value,
          cuda_stream,
          GetDeviceProp().maxThreadsPerBlock));
      data.attention_bias = reinterpret_cast<const CudaT*>(nonpad_kv_bias_buffer.get());
      contribop_parameters.broadcast_attn_bias_dim_0 = false;
      contribop_parameters.broadcast_attn_bias_dim_1 = true;
    } else if (nullptr != attn_mask) {
      if (attn_mask->IsDataType<bool>()) {
        // Convert boolean mask to additive attention bias: true -> 0.0, false -> mask_filter_value.
        // The conversion is element-wise and preserves the original shape, so the broadcast flags
        // set above apply identically to the converted float buffer.
        using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
        int64_t num_elements = attn_mask->Shape().Size();
        converted_mask_buffer = GetScratchBuffer<void>(num_elements * sizeof(NativeCudaT), context->GetComputeStream());
        auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
        ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
            attn_mask->Data<bool>(),
            reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
            num_elements,
            contribop_parameters.mask_filter_value,
            cuda_stream,
            GetDeviceProp().maxThreadsPerBlock));
        data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
      } else {
        data.attention_bias = reinterpret_cast<const CudaT*>(attn_mask->Data<T>());
      }
    }
    data.qkv_format = contribop_parameters.qkv_format;

    // The MHA path uses unfused O(n²) attention for all remaining cases:
    //   - fp32 with nonpad_kv_seqlen (attention_bias fallback; flash requires fp16/bf16)
    //   - All cases without nonpad_kv_seqlen
    // The nonpad fp16/bf16 case now uses flash attention (dispatched above via early return).
    data.use_flash_attention = false;
    data.use_memory_efficient_attention = false;
    data.fused_runner = nullptr;
    data.fused_cross_attention_kernel = nullptr;
    data.kernel_type = onnxruntime::contrib::AttentionKernelType::AttentionKernel_Unfused;

    // Allocate workspace for Q, K, V processing and scratch buffer
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
        nullptr,  // fused_runner
        false,    // use_flash_attention
        false,    // use_lean_attention
        false,    // use_fused_cross_attention
        false,    // use_memory_efficient_attention
        false,    // use_cudnn_flash_attention
        no_qkv_workspace);
    auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

    data.has_qkv_workspace = !no_qkv_workspace;
    data.workspace = reinterpret_cast<CudaT*>(work_space.get());
    data.workspace_bytes = workspace_bytes;

    // Call QkvToContext to perform the attention computation
    auto& device_prop = GetDeviceProp();
    cublasHandle_t cublas = GetCublasHandle(context);
    cudnnHandle_t cudnn = GetCudnnHandle(context);

    // QkvToContext takes two template parameters: T for computation type, QK for output_qk type
    // For now, both are the same type (CudaT)

    return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
        device_prop, cublas, cudnn, context->GetComputeStream(), contribop_parameters, data);
  }
}
}  // namespace cuda
}  // namespace onnxruntime
