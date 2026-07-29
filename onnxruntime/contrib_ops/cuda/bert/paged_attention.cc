// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/paged_attention.h"
#include "contrib_ops/cuda/bert/paged_attention_helper.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, TCACHE)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      PagedAttention,                                                         \
      kMSDomain,                                                              \
      1,                                                                      \
      T##_##TCACHE,                                                           \
      kCudaExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("T_CACHE", DataTypeImpl::GetTensorType<TCACHE>())   \
          .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()),       \
      PagedAttention<T, TCACHE>);

REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16, BFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, int8_t)
REGISTER_KERNEL_TYPED(BFloat16, int8_t)
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
REGISTER_KERNEL_TYPED(MLFloat16, Float8E4M3FN)
REGISTER_KERNEL_TYPED(BFloat16, Float8E4M3FN)
#endif

// True when TCACHE stores quantized values that need a scale on read/write.
template <typename TCACHE>
constexpr bool IsQuantizedCacheType() {
  if constexpr (std::is_same<TCACHE, int8_t>::value) {
    return true;
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
  } else if constexpr (std::is_same<TCACHE, Float8E4M3FN>::value) {
    return true;
#endif
  } else {
    return false;
  }
}

// The element type TCACHE stores, named as a KVCacheDataType so that an explicit k_cache_dtype /
// v_cache_dtype attribute can be checked against it.
template <typename TCACHE>
constexpr KVCacheDataType CacheStorageDataType() {
  if constexpr (std::is_same<TCACHE, int8_t>::value) {
    return KVCacheDataType::INT8;
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
  } else if constexpr (std::is_same<TCACHE, Float8E4M3FN>::value) {
    return KVCacheDataType::FLOAT8E4M3FN;
#endif
  } else if constexpr (std::is_same<TCACHE, BFloat16>::value) {
    return KVCacheDataType::BFLOAT16;
  } else {
    return KVCacheDataType::FLOAT16;
  }
}

template <typename T, typename TCACHE>
PagedAttention<T, TCACHE>::PagedAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  // scale == 0 selects the 1/sqrt(head_size) default. MLA derives its softmax scale from the
  // pre-absorption head width, so that default is silently wrong there and validation requires a
  // real value (see docs/contrib_ops/cuda/paged_attention.md §12.6).
  has_explicit_scale_ = scale_ != 0.0f;
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  ORT_ENFORCE(std::isfinite(qk_norm_epsilon_) && qk_norm_epsilon_ > 0.0f,
              "qk_norm_epsilon must be a positive finite number");
  k_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("k_quant_type", "NONE"));
  v_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("v_quant_type", "NONE"));
  // Empty (the default) means the cache tensor's own element type is the logical type, which covers
  // every format this operator stores today. A non-empty value names a sub-byte logical type packed
  // into a uint8 cache, which no build supports yet and is rejected during validation. The string is
  // parsed once here; everything downstream compares the enum.
  k_cache_dtype_ = StringToKVCacheDataType(info.GetAttrOrDefault<std::string>("k_cache_dtype", ""));
  v_cache_dtype_ = StringToKVCacheDataType(info.GetAttrOrDefault<std::string>("v_cache_dtype", ""));

  // Multi-head Latent Attention. "SEPARATE" (the default) is the shipped two-cache layout;
  // "LATENT" makes value/value_cache absent and aliases V onto the leading v_head_size channels of
  // key_cache. Anything else is rejected here rather than silently treated as SEPARATE.
  const std::string kv_cache_layout = info.GetAttrOrDefault<std::string>("kv_cache_layout", "SEPARATE");
  ORT_ENFORCE(kv_cache_layout == "SEPARATE" || kv_cache_layout == "LATENT",
              "'kv_cache_layout' must be 'SEPARATE' or 'LATENT', got '", kv_cache_layout, "'");
  is_latent_kv_ = kv_cache_layout == "LATENT";
  v_head_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("v_head_size", 0));
  ORT_ENFORCE(v_head_size_ >= 0, "'v_head_size' must be non-negative, got ", v_head_size_);
  rotary_offset_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_offset", 0));
  ORT_ENFORCE(rotary_offset_ >= 0, "'rotary_offset' must be non-negative, got ", rotary_offset_);

  kernel_options_ = this->GetAttentionKernelOptions();
  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();
  disable_memory_efficient_attention_ = sizeof(T) != 2 || !kernel_options_->UseEfficientAttention();
  disable_paged_decode_ = sizeof(T) != 2 || !kernel_options_->UseDecoderAttention();
}

template <typename T, typename TCACHE>
Status PagedAttention<T, TCACHE>::ComputeInternal(OpKernelContext* context) const {
  auto ort_stream = GetOrtStream(context);

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_cache = context->Input<Tensor>(3);
  const Tensor* value_cache = context->Input<Tensor>(4);
  const Tensor* cumulative_seqlens_q = context->Input<Tensor>(5);
  const Tensor* past_seqlens = context->Input<Tensor>(6);
  const Tensor* block_table = context->Input<Tensor>(7);
  const Tensor* cos_cache = context->Input<Tensor>(8);
  const Tensor* sin_cache = context->Input<Tensor>(9);
  const Tensor* slot_mapping = context->Input<Tensor>(10);
  const Tensor* head_sink = context->Input<Tensor>(11);
  const Tensor* q_norm_weight = context->Input<Tensor>(12);
  const Tensor* k_norm_weight = context->Input<Tensor>(13);
  const Tensor* k_scale = context->Input<Tensor>(14);
  const Tensor* v_scale = context->Input<Tensor>(15);

  auto& device_prop = GetDeviceProp();
  PagedAttentionParameters parameters;
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<TCACHE>::MappedType CudaTCache;
  constexpr bool kIsQuantizedCache = IsQuantizedCacheType<TCACHE>();
  PagedAttentionData<CudaT, CudaTCache> data;

  // Check shapes of inputs to op and set parameters
  ORT_RETURN_IF_ERROR(paged_attention_helper::CheckInputs(query,
                                                          key,
                                                          value,
                                                          key_cache,
                                                          value_cache,
                                                          cumulative_seqlens_q,
                                                          past_seqlens,
                                                          block_table,
                                                          cos_cache,
                                                          sin_cache,
                                                          slot_mapping,
                                                          head_sink,
                                                          q_norm_weight,
                                                          k_norm_weight,
                                                          k_scale,
                                                          v_scale,
                                                          &parameters,
                                                          num_heads_,
                                                          kv_num_heads_,
                                                          scale_,
                                                          softcap_,
                                                          qk_norm_epsilon_,
                                                          k_quant_type_,
                                                          v_quant_type_,
                                                          k_cache_dtype_,
                                                          v_cache_dtype_,
                                                          CacheStorageDataType<TCACHE>(),
                                                          is_latent_kv_,
                                                          v_head_size_,
                                                          rotary_offset_,
                                                          has_explicit_scale_,
                                                          device_prop.maxThreadsPerBlock));
  parameters.local_window_size = local_window_size_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  DUMP_STRING_INIT();
  DUMP_STRING("Batch size = ", parameters.batch_size);
  DUMP_STRING("Token count = ", parameters.token_count);
  DUMP_STRING("Q hidden size = ", parameters.hidden_size);
  DUMP_STRING("KV hidden size = ", parameters.kv_hidden_size);
  DUMP_STRING("Q num heads = ", parameters.num_heads);
  DUMP_STRING("KV num heads = ", parameters.kv_num_heads);
  DUMP_STRING("Head size = ", parameters.head_size);
  DUMP_STRING("V head size = ", parameters.v_head_size);
  DUMP_STRING("Latent (MLA) KV layout = ", parameters.is_latent_kv);
  DUMP_STRING("Rotary offset = ", parameters.rotary_offset);
  DUMP_STRING("Num blocks = ", parameters.num_blocks);
  DUMP_STRING("Block size = ", parameters.block_size);
  DUMP_STRING("Max num blocks per sequence = ", parameters.max_num_blocks_per_seq);
  DUMP_STRING("Rotary dimension = ", parameters.rotary_dim);
  DUMP_STRING("Is packed QKV = ", parameters.is_packed_qkv);

  // Check rotary
  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to PagedAttention when do_rotary = 1");
  }

  // Set output tensor shapes. In LATENT mode the output head width is v_head_size, so it is
  // narrower than the query head width (512 vs. 576 for DeepSeek-V3). v_hidden_size equals
  // hidden_size in every SEPARATE-mode model.
  TensorShapeVector output_shape(2);
  output_shape[0] = static_cast<int64_t>(parameters.token_count);
  output_shape[1] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  TensorShapeVector key_cache_out_shape(4);
  key_cache_out_shape[0] = static_cast<int64_t>(parameters.num_blocks);
  key_cache_out_shape[1] = static_cast<int64_t>(parameters.block_size);
  key_cache_out_shape[2] = static_cast<int64_t>(parameters.kv_num_heads);
  key_cache_out_shape[3] = static_cast<int64_t>(parameters.head_size);
  Tensor* key_cache_out = context->Output(1, key_cache_out_shape);

  // LATENT has a single physical cache, so there is no value_cache_out to produce.
  Tensor* value_cache_out = nullptr;
  if (!parameters.is_latent_kv) {
    TensorShapeVector value_cache_out_shape(4);
    value_cache_out_shape[0] = static_cast<int64_t>(parameters.num_blocks);
    value_cache_out_shape[1] = static_cast<int64_t>(parameters.block_size);
    value_cache_out_shape[2] = static_cast<int64_t>(parameters.kv_num_heads);
    value_cache_out_shape[3] = static_cast<int64_t>(parameters.head_size);
    value_cache_out = context->Output(2, value_cache_out_shape);
  }

  if (key_cache_out != nullptr && key_cache->Data<TCACHE>() != key_cache_out->MutableData<TCACHE>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "key_cache and key_cache_out must be the same buffer");
  } else if (value_cache_out != nullptr && value_cache->Data<TCACHE>() != value_cache_out->MutableData<TCACHE>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "value_cache and value_cache_out must be the same buffer");
  }

  // Empty query input: output is already shaped [0, hidden_size], and the cache outputs
  // alias the input caches (verified above), so no backend kernel or cache update is needed.
  if (parameters.token_count == 0) {
    return Status::OK();
  }

  // Kernel backend selection — three backends, decided partly after a D->H sync because the choice
  // depends on the actual max new-query length.
  //
  //   * FlashAttention (preferred for prefill / mixed batches).
  //   * PagedDecode: a flash-decoding style kernel that reads the paged cache in place and
  //     dequantizes in registers. Only handles one new token per sequence.
  //   * MemoryEfficientAttention: the general fallback; gathers pages into a dense buffer.
  //
  // The vendored FlashAttention paged kernel loads a whole kBlockN x head_size K/V tile using a
  // single (page, offset) pair, so a tile must never straddle a page boundary: block_size has to be
  // a multiple of kBlockN. kBlockN is fixed by head_size in run_mha_fwd_splitkv_dispatch. When the
  // model uses a smaller page than that, we fall back to another backend (both of which accept any
  // block_size) rather than rejecting the model.
  //
  // A quantized cache is exempt: FlashAttention cannot read a quantized page at all, so that path
  // dequantizes the live context into a dense buffer and uses the non-paged varlen entry point,
  // which has no page-alignment requirement.
  const int flash_min_block_size =
      parameters.head_size <= 64 ? 256 : (parameters.head_size <= 128 ? 128 : 64);
  const bool flash_block_size_ok = kIsQuantizedCache || (parameters.block_size % flash_min_block_size) == 0;

  // LATENT (absorbed MLA) has exactly one eligible backend: neither FlashAttention nor the CUTLASS
  // fMHA wrapper supports v_head_size != head_size or a head_size of 576, and the paged decode
  // kernel assumes a separate value cache. See docs/contrib_ops/cuda/paged_attention.md §12.7.
  const bool use_latent_attention = parameters.is_latent_kv;
  if (use_latent_attention) {
    const size_t latent_smem = GetPagedLatentSharedMemoryBytes(parameters.head_size, parameters.v_head_size);
    if (latent_smem > static_cast<size_t>(device_prop.sharedMemPerBlock)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention: the unfused MLA backend needs ", latent_smem,
                             " bytes of shared memory for head_size=", parameters.head_size,
                             " and v_head_size=", parameters.v_head_size, ", but the device provides ",
                             device_prop.sharedMemPerBlock, " bytes per block.");
    }
  }

#if USE_FLASH_ATTENTION
  const bool flash_eligible = !use_latent_attention &&
                              !disable_flash_attention_ &&
                              flash_block_size_ok &&
                              onnxruntime::flash::is_supported<T>(device_prop,
                                                                  parameters.head_size,
                                                                  parameters.num_heads,
                                                                  parameters.kv_num_heads);
#else
  const bool flash_eligible = false;
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  const int sm = device_prop.major * 10 + device_prop.minor;
  const bool is_half = std::is_same<T, MLFloat16>::value;
  const bool is_bf16 = std::is_same<T, BFloat16>::value;
  const bool mea_eligible =
      !use_latent_attention &&
      !flash_eligible &&
      !disable_memory_efficient_attention_ &&
      has_memory_efficient_attention(sm, is_half, is_bf16,
                                     parameters.head_size, parameters.head_size);
#else
  const bool mea_eligible = false;
#endif

  // The decode kernel keeps Q, the running accumulator and one KV tile in shared memory, which
  // bounds the head size it can serve.
  const bool decode_eligible =
      !use_latent_attention &&
      !disable_paged_decode_ &&
      GetPagedDecodeSharedMemoryBytes(parameters.head_size) <= static_cast<size_t>(device_prop.sharedMemPerBlock);

  size_t cumulative_seqlens_kv_bytes = sizeof(int) * (parameters.batch_size + 1);
  auto cumulative_seqlens_kv_buffer = GetScratchBuffer<void>(cumulative_seqlens_kv_bytes, GetComputeStream(context));
  int* cumulative_seqlens_kv_ptr = reinterpret_cast<int*>(cumulative_seqlens_kv_buffer.get());

  // The fused prologue (QK-Norm and/or rotary) writes densified Q and K into the workspace, so it
  // needs room for both. Plain packed-QKV only needs to densify Q.
  const bool needs_qk_prologue = do_rotary_ || parameters.use_qk_norm;
  size_t workspace_buffer_bytes = 0;
  if (needs_qk_prologue) {
    workspace_buffer_bytes = sizeof(T) * parameters.token_count * (parameters.hidden_size + parameters.kv_hidden_size);
  } else if (parameters.is_packed_qkv) {
    workspace_buffer_bytes = sizeof(T) * parameters.token_count * parameters.hidden_size;
  }
  auto workspace_buffer = GetScratchBuffer<void>(workspace_buffer_bytes, GetComputeStream(context));

  // Populate cumulative_seqlens_kv for all backends. The host then reads both cumulative arrays
  // back to size the gather buffers and to learn whether this is a pure decode step.
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(ort_stream.get()->GetHandle());
  ORT_RETURN_IF_ERROR(LaunchGetCumulativeSeqlensKV(
      cumulative_seqlens_kv_ptr,
      reinterpret_cast<const int*>(cumulative_seqlens_q->Data<int>()),
      reinterpret_cast<const int*>(past_seqlens->Data<int>()),
      parameters.batch_size, cuda_stream));

  int total_kv_tokens = 0;
  int max_query_len = 0;
  int max_kv_len = 0;
  IAllocatorUniquePtr<void> gathered_key_buffer;
  IAllocatorUniquePtr<void> gathered_value_buffer;
  IAllocatorUniquePtr<void> fmha_buffer;

  // Compute max_query_len on the host. The previous `token_count - batch_size + 1` heuristic
  // underestimates (or goes non-positive) when batches have zero new tokens. max_kv_len and
  // total_kv_tokens size the dense staging buffers and the decode split count.
  {
    const int kCumulativeCount = parameters.batch_size + 1;
    auto cum_q_pinned = this->AllocateBufferOnCPUPinned<int>(kCumulativeCount);
    auto cum_kv_pinned = this->AllocateBufferOnCPUPinned<int>(kCumulativeCount);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cum_q_pinned.get(),
                                         reinterpret_cast<const int*>(cumulative_seqlens_q->Data<int>()),
                                         sizeof(int) * kCumulativeCount, cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cum_kv_pinned.get(), cumulative_seqlens_kv_ptr,
                                         sizeof(int) * kCumulativeCount, cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
    for (int i = 0; i < parameters.batch_size; ++i) {
      const int q_len_i = cum_q_pinned.get()[i + 1] - cum_q_pinned.get()[i];
      if (q_len_i > max_query_len) {
        max_query_len = q_len_i;
      }
      const int kv_len_i = cum_kv_pinned.get()[i + 1] - cum_kv_pinned.get()[i];
      if (kv_len_i > max_kv_len) {
        max_kv_len = kv_len_i;
      }
    }
    total_kv_tokens = cum_kv_pinned.get()[parameters.batch_size];
    if (total_kv_tokens <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "PagedAttention: total_kv_tokens is not positive (", total_kv_tokens,
                             ") for non-empty input.");
    }
  }

  // Prefer the decode kernel only where it is a clear win: a quantized cache (it avoids
  // materializing and dequantizing the whole live context) or when FlashAttention is unavailable
  // (it beats the dense-gather MemoryEfficientAttention fallback). Unquantized FlashAttention-
  // eligible shapes keep using FlashAttention.
  const bool use_paged_decode = decode_eligible && max_query_len == 1 && (kIsQuantizedCache || !flash_eligible);
  const bool use_flash_attention = flash_eligible && !use_paged_decode;
  const bool use_memory_efficient_attention = mea_eligible && !use_paged_decode;

  if (!use_latent_attention && !use_paged_decode && !use_flash_attention && !use_memory_efficient_attention) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention requires FlashAttention (sm>=80, fp16/bf16, block_size a multiple of ",
                           flash_min_block_size, " for head_size ", parameters.head_size,
                           "), MemoryEfficientAttention (fp16 sm>=53, bf16 sm>=80, head_size<=1024 and %8==0), "
                           "or the paged decode kernel (fp16/bf16, one new token per sequence; this step has "
                           "max_query_len=",
                           max_query_len,
                           ") to be available. Check ORT_DISABLE_FLASH_ATTENTION / "
                           "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION / ORT_DISABLE_DECODER_ATTENTION env vars and "
                           "dtype/head_size/block_size.");
  }

  // The attention-sink epilogue rescales the output using FlashAttention's log-sum-exp, which the
  // CUTLASS memory-efficient kernel does not expose. The decode kernel folds the sink straight into
  // its softmax denominator, so only the MEA path has to fail loudly here.
  if (parameters.use_smooth_softmax && use_memory_efficient_attention) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention: 'head_sink' is only supported by the FlashAttention "
                           "and paged decode backends, but the MemoryEfficientAttention backend was selected "
                           "(head_size=",
                           parameters.head_size, ", block_size=", parameters.block_size,
                           "). FlashAttention requires sm>=80, fp16/bf16 and block_size a multiple of ",
                           flash_min_block_size, ".");
  }

  size_t softmax_lse_bytes = 0;
#if USE_FLASH_ATTENTION
  if (use_flash_attention) {
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.token_count,
                                                                 parameters.num_heads);
  }
#endif
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, GetComputeStream(context));

  // Both gather-based backends need a dense KV staging buffer when the cache is quantized
  // (FlashAttention cannot read a quantized page, and the CUTLASS kernel is not paged at all).
  const bool needs_dense_kv = use_memory_efficient_attention || (use_flash_attention && kIsQuantizedCache);
  // The dense buffer keeps the grouped layout for FlashAttention (it does GQA internally) and is
  // GQA-expanded for the CUTLASS kernel.
  const int gathered_num_heads = use_memory_efficient_attention ? parameters.num_heads : parameters.kv_num_heads;

  if (needs_dense_kv) {
    const size_t gather_elems = static_cast<size_t>(total_kv_tokens) *
                                gathered_num_heads * parameters.head_size;
    gathered_key_buffer = GetScratchBuffer<void>(sizeof(T) * gather_elems, GetComputeStream(context));
    gathered_value_buffer = GetScratchBuffer<void>(sizeof(T) * gather_elems, GetComputeStream(context));
  }

  // Split-KV workspaces for the decode kernel: one partial (accumulator, max, denominator) per
  // split. Splitting only pays off when there are too few (sequence, head) pairs to fill the GPU.
  int num_splits = 1;
  IAllocatorUniquePtr<void> decode_partial_out_buffer;
  IAllocatorUniquePtr<void> decode_partial_max_buffer;
  IAllocatorUniquePtr<void> decode_partial_sum_buffer;
  if (use_paged_decode) {
    num_splits = ComputePagedDecodeSplits(parameters.batch_size, parameters.num_heads, max_kv_len,
                                          device_prop.multiProcessorCount);
    const size_t rows = static_cast<size_t>(num_splits) * parameters.token_count * parameters.num_heads;
    decode_partial_out_buffer =
        GetScratchBuffer<void>(sizeof(float) * rows * parameters.head_size, GetComputeStream(context));
    decode_partial_max_buffer = GetScratchBuffer<void>(sizeof(float) * rows, GetComputeStream(context));
    decode_partial_sum_buffer = GetScratchBuffer<void>(sizeof(float) * rows, GetComputeStream(context));
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (use_memory_efficient_attention) {
    if (MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float))) {
      // MEA output accumulator is float32 regardless of input dtype (see GQA pattern at
      // group_query_attention.cc:482); use sizeof(float), not sizeof(T).
      const size_t fmha_elems = static_cast<size_t>(parameters.token_count) *
                                parameters.num_heads * parameters.head_size;
      fmha_buffer = GetScratchBuffer<void>(sizeof(float) * fmha_elems, GetComputeStream(context));
    }
  }
#endif

  // Print debug info
  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_efficient_attention = use_memory_efficient_attention;
    debug_info.use_decoder_attention = use_paged_decode;

    debug_info.Print("PagedAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  // Set up data struct for kernel launch
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.key_cache = reinterpret_cast<CudaTCache*>(const_cast<TCACHE*>(key_cache->Data<TCACHE>()));
  // Absent in LATENT mode, where V is a slice of key_cache and ReshapeAndCache skips the V store.
  data.value_cache = value_cache == nullptr
                         ? nullptr
                         : reinterpret_cast<CudaTCache*>(const_cast<TCACHE*>(value_cache->Data<TCACHE>()));
  data.k_scale = k_scale == nullptr ? nullptr : k_scale->Data<float>();
  data.v_scale = v_scale == nullptr ? nullptr : v_scale->Data<float>();
  data.cumulative_seqlens_q = reinterpret_cast<const int*>(cumulative_seqlens_q->Data<int>());
  data.past_seqlens = reinterpret_cast<const int*>(past_seqlens->Data<int>());
  data.cumulative_seqlens_kv = cumulative_seqlens_kv_ptr;
  data.block_table = reinterpret_cast<const int*>(block_table->Data<int>());
  data.slot_mapping = slot_mapping == nullptr ? nullptr : reinterpret_cast<const int*>(slot_mapping->Data<int>());
  data.head_sink = head_sink == nullptr ? nullptr : reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  data.q_norm_weight = q_norm_weight == nullptr ? nullptr : reinterpret_cast<const CudaT*>(q_norm_weight->Data<T>());
  data.k_norm_weight = k_norm_weight == nullptr ? nullptr : reinterpret_cast<const CudaT*>(k_norm_weight->Data<T>());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  data.use_paged_decode = use_paged_decode;
  if (softmax_lse_buffer != nullptr) {
    // FlashAttention always writes fp32 log-sum-exp, independent of T.
    data.softmax_lse = reinterpret_cast<float*>(softmax_lse_buffer.get());
  }
  if (workspace_buffer != nullptr) {
    data.workspace_buffer = reinterpret_cast<CudaT*>(workspace_buffer.get());
  }
  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }
  data.max_query_len = max_query_len;  // consumed by all backends
  if (needs_dense_kv) {
    data.gathered_key = reinterpret_cast<CudaT*>(gathered_key_buffer.get());
    data.gathered_value = reinterpret_cast<CudaT*>(gathered_value_buffer.get());
    data.total_kv_tokens = total_kv_tokens;
    data.max_kv_len = max_kv_len;
  }
  if (use_paged_decode) {
    data.decode_partial_out = reinterpret_cast<float*>(decode_partial_out_buffer.get());
    data.decode_partial_max = reinterpret_cast<float*>(decode_partial_max_buffer.get());
    data.decode_partial_sum = reinterpret_cast<float*>(decode_partial_sum_buffer.get());
    data.num_splits = num_splits;
  }
  if (use_memory_efficient_attention && fmha_buffer != nullptr) {
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT, CudaTCache>(
      device_prop, cublas, ort_stream.get(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
