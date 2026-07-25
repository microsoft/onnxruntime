// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
#include "contrib_ops/cuda/bert/unfused_attention.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
// Map string attribute to quantization type enum
KVQuantizationType StringToKVQuantizationType(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  if (s == "NONE") {
    return KVQuantizationType::NONE;
  }
  if (s == "PER_TENSOR") {
    return KVQuantizationType::PER_TENSOR;
  }

  if (s == "PER_CHANNEL") {
    return KVQuantizationType::PER_CHANNEL;
  }
  return KVQuantizationType::NONE;
}
}  // namespace

#define REGISTER_KERNEL_TYPED(T, U)                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                      \
      GroupQueryAttention,                                                            \
      kMSDomain,                                                                      \
      1,                                                                              \
      T##_##U,                                                                        \
      kCudaExecutionProvider,                                                         \
      (*KernelDefBuilder::Create())                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                      \
          .TypeConstraint("T_CACHE", DataTypeImpl::GetTensorType<U>())                \
          .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>())         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()})              \
          .MayInplace(3, 1)                        /* past_key and present_key */     \
          .MayInplace(4, 2)                        /* past_value and present_value */ \
          .InputMemoryType(OrtMemTypeCPUInput, 6), /* total_sequence_length */        \
      GroupQueryAttention<T, U>);

REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16, BFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, int8_t)
REGISTER_KERNEL_TYPED(BFloat16, int8_t)
#ifdef USE_FP8_KV_CACHE
REGISTER_KERNEL_TYPED(MLFloat16, Float8E4M3FN)
REGISTER_KERNEL_TYPED(BFloat16, Float8E4M3FN)
#endif
#ifdef USE_INT4_KV_CACHE
REGISTER_KERNEL_TYPED(MLFloat16, uint8_t)
REGISTER_KERNEL_TYPED(BFloat16, uint8_t)
#endif

constexpr const char* kDisableFlashDecode = "ORT_DISABLE_FLASH_DECODE";
constexpr int kHeadSinkInputIndex = 11;

// Group Query Attention (GQA) Operator
//
// This operator implements Group Query Attention, a variation of Multi-Head Attention (MHA)
// where the number of key/value heads is smaller than the number of query heads.
// It supports:
// - Rotary Positional Embeddings (RoPE)
// - KV Cache (past/present key/value)
// - Quantized KV Cache (Int8/Int4) via GroupQueryAttentionData
// - Flash Attention and Memory Efficient Attention backends
//
template <typename T, typename U>
GroupQueryAttention<T, U>::GroupQueryAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;
  is_unidirectional_ = true;
  const int64_t local_window_size_attr = info.GetAttrOrDefault<int64_t>("local_window_size", -1);
  // Validate before narrowing to int so an out-of-range attribute cannot wrap to a valid-looking
  // small window (e.g. 2^32 + 128) and silently run a different window than the model specifies.
  ORT_ENFORCE(local_window_size_attr == -1 || (local_window_size_attr > 0 && local_window_size_attr <= std::numeric_limits<int>::max()),
              "local_window_size must be -1 or greater than 0 (and not exceed INT_MAX).");
  local_window_size_ = static_cast<int>(local_window_size_attr);
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  ORT_ENFORCE(std::isfinite(qk_norm_epsilon_) && qk_norm_epsilon_ > 0.0f,
              "GroupQueryAttention (CUDA): qk_norm_epsilon must be finite and positive.");

  k_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("k_quant_type", "NONE"));
  v_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("v_quant_type", "NONE"));
  kv_cache_bit_width_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_cache_bit_width", 0));

  constexpr bool kIsFp16OrBf16 = std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>;
  // XQA defaults on for fp16/bf16; ORT_ENABLE_XQA=0 disables it explicitly.
  enable_xqa_ = kIsFp16OrBf16 && (ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_XQA", 1) != 0);

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

  // Memory efficient attention supports float and float16. BFloat16 support added for SM80+.
  disable_memory_efficient_attention_ = !kernel_options_->UseEfficientAttention();

  // cuDNN SDPA (cudnn_frontend) supports FP16 and BF16 and is auto-preferred on SM>=90.
  enable_cudnn_flash_attention_ = kIsFp16OrBf16 && kernel_options_->UseCudnnFlashAttention();
  auto_enable_cudnn_flash_attention_ = kIsFp16OrBf16 && kernel_options_->AllowCudnnFlashAttentionAuto();

  if (!disable_flash_attention_) {
    zeros_ = this->GetScratchBuffer<int>(kZerosCount, nullptr);
    CUDA_CALL_THROW(cudaMemset(zeros_.get(), 0, kZerosCount * sizeof(int)));
  }

  disable_flash_decode_ = ParseEnvironmentVariableWithDefault<bool>(kDisableFlashDecode, false);
}

template <typename T, typename U>
Status GroupQueryAttention<T, U>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                          bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  // Keep is_packed=false so the original fp16/bf16 head_sink remains available to the Flash/fallback
  // paths (which are used when XQA is disabled or ineligible). We only cache an extra FP32 copy for XQA.
  is_packed = false;

  if (input_idx != kHeadSinkInputIndex) {
    return Status::OK();
  }

  // XQA consumes the attention sink as FP32. When head_sink is a constant initializer, convert it once
  // here into a cached device buffer (xqa_head_sink_) to avoid a per-launch conversion. Dynamic /
  // non-initializer head_sink inputs are not prepacked and fall back to the per-launch scratch path.
  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    const auto& shape = tensor.Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 1,
                      "head_sink must be a 1D tensor, got ", shape.NumDimensions(), " dimensions");
    ORT_RETURN_IF_NOT(shape[0] == num_heads_,
                      "head_sink dimension 0 must be equal to the num heads, got ", shape[0]);
    ORT_RETURN_IF_NOT(tensor.IsDataType<T>(), "head_sink type must match GroupQueryAttention input type");

    // Derive the element count from the tensor itself (one sink per head) rather than num_heads_.
    const int head_sink_count = static_cast<int>(shape.Size());
    const size_t head_sink_bytes = tensor.SizeInBytes();
    const void* head_sink_data = tensor.DataRaw();
    IAllocatorUniquePtr<void> head_sink_gpu;
    cudaStream_t stream = cudaStreamLegacy;

    if (tensor.Location().device.Type() == OrtDevice::CPU) {
      head_sink_gpu = IAllocator::MakeUniquePtr<void>(alloc, head_sink_bytes, true);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(head_sink_gpu.get(), head_sink_data, head_sink_bytes,
                                           cudaMemcpyHostToDevice, stream));
      head_sink_data = head_sink_gpu.get();
    }

    xqa_head_sink_ = IAllocator::MakeUniquePtr<float>(alloc, static_cast<size_t>(head_sink_count), true);
    using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
    ORT_RETURN_IF_ERROR(LaunchConvertHeadSinkToFloat<CudaT>(
        reinterpret_cast<const CudaT*>(head_sink_data), xqa_head_sink_.get(), head_sink_count, stream,
        GetDeviceProp().maxThreadsPerBlock));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    xqa_head_sink_count_ = head_sink_count;
  }

  return Status::OK();
}

// ComputeInternal executes the GQA kernel.
//
// Inputs:
// 0. query             (Tensor) [batch, sequence_length, hidden_size]
// 1. key               (Tensor) [batch, sequence_length, kv_hidden_size] (Optional)
// 2. value             (Tensor) [batch, sequence_length, kv_hidden_size] (Optional)
// 3. past_key          (Tensor) [batch, num_kv_heads, max_seq_len, head_size] (Optional)
// 4. past_value        (Tensor) [batch, num_kv_heads, max_seq_len, head_size] (Optional)
// 5. seqlens_k         (Tensor) [batch] - Total sequence length minus 1 (for historical compatibility)
// 6. total_seqlen      (Tensor) - Max total sequence length
// 7. cos_cache         (Tensor) - Precomputed cosine table for RoPE
// 8. sin_cache         (Tensor) - Precomputed sine table for RoPE
// 9. position_ids      (Tensor) - Position indices for RoPE
// 10. attention_bias   (Tensor) [batch or 1, num_heads or 1, sequence_length, total_sequence_length] (Optional)
//                      Additive bias to QxK'; dispatches to the unfused fallback path.
// 11. head_sink        (Tensor) - Attention sink for GPT-OSS
template <typename T, typename U>
Status GroupQueryAttention<T, U>::ComputeInternal(OpKernelContext* context) const {
  auto ort_stream = GetOrtStream(context);

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);

  // The input seqlens_k is total sequence length - 1 for historical reasons.
  // Rename it to total_seq_lens_minus_one in cuda kernel to avoid confusion.
  const Tensor* total_seq_lens_minus_one = context->Input<Tensor>(5);

  // The max of total sequence lengths. The content of this tensor is a scalar stored in CPU memory.
  const Tensor* total_seqlen = context->Input<Tensor>(6);

  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);
  const Tensor* position_ids = context->Input<Tensor>(9);
  const Tensor* attention_bias = context->Input<Tensor>(10);
  const Tensor* head_sink = context->Input<Tensor>(11);
  const Tensor* k_scale = context->Input<Tensor>(12);
  const Tensor* v_scale = context->Input<Tensor>(13);

  // q_norm_weight (input 14) / k_norm_weight (input 15) carry the per-head Q/K RMSNorm (QK-Norm)
  // prologue weights, each of shape (head_size,) and shared across heads. They are populated by the
  // GroupQueryAttentionPreNormFusion optimizer pass for Qwen3 / Gemma 2-3 / OLMo2 / SmolLM3 style
  // models. Both must be present together; shape validation and wiring happen after CheckInputs,
  // where head_size is known.
  const Tensor* q_norm_weight = (context->InputCount() > 14) ? context->Input<Tensor>(14) : nullptr;
  const Tensor* k_norm_weight = (context->InputCount() > 15) ? context->Input<Tensor>(15) : nullptr;
  if ((q_norm_weight != nullptr) != (k_norm_weight != nullptr)) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "GroupQueryAttention (CUDA): q_norm_weight and k_norm_weight must be provided together.");
  }

  if (k_quant_type_ != KVQuantizationType::NONE) {
    if (k_scale == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "k_scale must be provided when k_quant_type is not NONE");
    }

    if (k_scale->DataType() != DataTypeImpl::GetType<float>()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "k_scale must be float tensor");
    }
  }

  if (v_quant_type_ != KVQuantizationType::NONE) {
    if (v_scale == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "v_scale must be provided when v_quant_type is not NONE");
    }
    if (v_scale->DataType() != DataTypeImpl::GetType<float>()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "v_scale must be float tensor");
    }
  }

  // attention_bias runs on the unfused fallback path (the fused kernels below have no bias input),
  // so the feature combinations that path excludes stay NOT_IMPLEMENTED with the bias present.
  const bool has_attention_bias = attention_bias != nullptr;
  if (has_attention_bias) {
    if (!attention_bias->IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_bias type must match GroupQueryAttention input type T.");
    }
    if (k_quant_type_ != KVQuantizationType::NONE || v_quant_type_ != KVQuantizationType::NONE) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "attention_bias with quantized KV cache is not implemented in GroupQueryAttention cuda kernel.");
    }
    if (use_smooth_softmax_ || head_sink != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "attention_bias with smooth softmax or head_sink is not implemented in GroupQueryAttention cuda kernel.");
    }
  }

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;

  typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
  typedef typename onnxruntime::cuda::OrtToCudaType<U>::type CudaU;
  GroupQueryAttentionData<CudaT, CudaU> data;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                total_seq_lens_minus_one,
                                                                total_seqlen,
                                                                scale_,
                                                                softcap_,
                                                                kv_cache_bit_width_,
                                                                device_prop.maxThreadsPerBlock));
#ifndef USE_INT4_KV_CACHE
  if (kv_cache_bit_width_ == 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv_cache_bit_width==4 is not enabled in this build.");
  }
#endif

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               parameters));

  if (has_attention_bias) {
    const auto& bias_dims = attention_bias->Shape().GetDims();
    parameters.broadcast_attn_bias_dim_0 = bias_dims[0] == 1;
    parameters.broadcast_attn_bias_dim_1 = bias_dims[1] == 1;
  }

  // Validate and enable the per-head Q/K RMSNorm (QK-Norm) prologue (inputs 14/15). Both weights
  // must be 1D tensors of shape (head_size) with element type T (shared across all heads).
  if (q_norm_weight != nullptr) {
    const auto& q_norm_shape = q_norm_weight->Shape();
    const auto& k_norm_shape = k_norm_weight->Shape();
    if (q_norm_shape.NumDimensions() != 1 || q_norm_shape[0] != parameters.head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "q_norm_weight must be a 1D tensor of shape (head_size=", parameters.head_size, ")");
    }
    if (k_norm_shape.NumDimensions() != 1 || k_norm_shape[0] != parameters.head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "k_norm_weight must be a 1D tensor of shape (head_size=", parameters.head_size, ")");
    }
    if (!q_norm_weight->IsDataType<T>() || !k_norm_weight->IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "q_norm_weight/k_norm_weight type must match GroupQueryAttention input type T.");
    }
    parameters.use_qk_norm = true;
  }
  parameters.qk_norm_epsilon = qk_norm_epsilon_;

  parameters.local_window_size = local_window_size_;
  parameters.is_unidirectional = is_unidirectional_;
  parameters.use_smooth_softmax = use_smooth_softmax_ || head_sink != nullptr;
  parameters.zeros_count = kZerosCount;
  parameters.zero_ptr = zeros_.get();
  parameters.k_quant_type = k_quant_type_;
  parameters.v_quant_type = v_quant_type_;
  parameters.kv_cache_bit_width = kv_cache_bit_width_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  // The current GQA CUDA implementation will never be able to have a QK output.
  // GQA CUDA uses either flash attention or memory efficient attention. Neither kernel supports returning the QK output.
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckNoQKOutput(
      context->OutputCount(),
      static_cast<int>(Info().template GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)))));

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to GroupQueryAttention when do_rotary = 1");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Set up present KV output shapes
  // For 4-bit quantization, we pack two 4-bit values into one uint8 byte.
  // Therefore, the dense head size in the tensor shape is halved (rounded up).
  int dense_head_size = (parameters.kv_cache_bit_width == 4) ? (parameters.head_size + 1) / 2 : parameters.head_size;
  std::vector<int64_t> present_dims = {
      parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, dense_head_size};

  TensorShape present_shape(present_dims);
  Tensor* present_key_output = context->Output(1, present_shape);    // present_key
  Tensor* present_value_output = context->Output(2, present_shape);  // present_value

  IAllocatorUniquePtr<void> k_buffer;
  IAllocatorUniquePtr<void> v_buffer;
  IAllocatorUniquePtr<void> rotary_buffer;
  IAllocatorUniquePtr<void> fmha_buffer;
  IAllocatorUniquePtr<void> unpacked_qkv_buffer;
  IAllocatorUniquePtr<int> seq_lens_buffer;

  // Flash Attention buffers
  IAllocatorUniquePtr<void> softmax_lse_buffer;
  IAllocatorUniquePtr<void> softmax_lse_accum_buffer;
  IAllocatorUniquePtr<void> out_accum_buffer;

  data.position_ids = (position_ids != nullptr) ? position_ids->Data<int64_t>() : nullptr;

  // Input pointers for both paths
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());

  // Handle Past/Present pointers
  data.k_scale = k_scale == nullptr ? nullptr : reinterpret_cast<const float*>(k_scale->DataRaw());
  data.v_scale = v_scale == nullptr ? nullptr : reinterpret_cast<const float*>(v_scale->DataRaw());

  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaU*>(past_key->Data<U>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaU*>(past_value->Data<U>());
  data.present_key = reinterpret_cast<CudaU*>(present_key_output->MutableData<U>());
  data.present_value = reinterpret_cast<CudaU*>(present_value_output->MutableData<U>());
  // Compute past_present_share_buffer early since it's needed for flash attention path selection.
  bool past_key_shared = (data.past_key != nullptr && data.past_key == data.present_key);
  bool past_value_shared = (data.past_value != nullptr && data.past_value == data.present_value);
  ORT_ENFORCE(past_key_shared == past_value_shared,
              "past_key/present_key and past_value/present_value must be both shared or both separate.");
  parameters.past_present_share_buffer = past_key_shared;

  bool is_inputs_quantized = (k_quant_type_ != KVQuantizationType::NONE) || (v_quant_type_ != KVQuantizationType::NONE);
  constexpr bool is_int8 = std::is_same<U, int8_t>::value;
  constexpr bool is_fp8 = std::is_same<U, Float8E4M3FN>::value;

  // Allocate XQA scratch if needed (only for Flash Decoding path)
  IAllocatorUniquePtr<void> xqa_scratch_buffer;
  // Check conditions to enable XQA (Extreme Query Attention) kernel for optimized decoding.
  // XQA is a highly optimized kernel for generation phase (seq_len=1).
  // Constraints:
  // 1. Compute Capability SM 8.0+ (Ampere or newer).
  // 2. Not the first prompt (decoding phase).
  // 3. Sequence length is 1.
  // 4. Past and Present KV cache share the same buffer (required for XQA specific memory access).
  // 5. No Softcap (XQA doesn't support softcap).
  // 6. Standard Softmax, or smooth softmax represented by a head_sink tensor.
  // 7. Local window (sliding window) attention is supported on both the non-quantized and the
  //    quantized (INT8/FP8) XQA paths (local_window_size == -1 means global attention).
  // QK-Norm can use XQA for the non-quantized KV-cache path: ExtremeDecoding runs the same
  // UnpackRoPEAppend preprocess before XQA, so Q/K can be normalized before the XQA kernel consumes
  // Q and the appended cache. Keep quantized QK-Norm off the XQA route until scale correctness is
  // validated for normalized K before quantized-cache append.
  const bool xqa_qk_norm_ok = !parameters.use_qk_norm || !is_inputs_quantized;
  const bool use_xqa_attention_sinks = head_sink != nullptr && !is_inputs_quantized;
  const bool is_xqa_smooth_softmax_supported = !parameters.use_smooth_softmax || use_xqa_attention_sinks;
  // XQA is enabled when enable_xqa_=true; ineligible shapes/group sizes fall back via data.use_xqa below.
  // The XQA kernel has no attention_bias input.
  if (enable_xqa_ &&
      !has_attention_bias &&
      (device_prop.major >= 8) &&
      !parameters.is_first_prompt &&
      parameters.sequence_length == 1 &&
      parameters.kv_sequence_length > 0 &&  // Shared KV (kv_seq=0) has no new K/V to append
      parameters.past_present_share_buffer &&
      parameters.softcap == 0.0f &&
      xqa_qk_norm_ok &&
      is_xqa_smooth_softmax_supported) {
    int group_size = parameters.num_heads / parameters.kv_num_heads;

    // Sliding window (local_window_size > 0) is wired through to the quantized XQA kernels as well,
    // so the INT8/FP8 variants no longer need to be restricted to global attention.
    bool is_int8_quantized_supported = is_int8 &&
                                       (k_quant_type_ == KVQuantizationType::PER_TENSOR &&
                                        v_quant_type_ == KVQuantizationType::PER_TENSOR &&
                                        data.k_scale == data.v_scale &&  // XQA requires k_scale and v_scale to be the same. Here requires k_scale and v_scale are same tensor.
                                        (parameters.head_size == 256 || parameters.head_size == 128 || parameters.head_size == 64) &&
                                        (group_size == 4 || group_size == 8 || group_size == 16 || group_size == 32));

#ifdef USE_FP8_KV_CACHE
    bool is_fp8_quantized_supported = is_fp8 &&
                                      (k_quant_type_ == KVQuantizationType::PER_TENSOR &&
                                       v_quant_type_ == KVQuantizationType::PER_TENSOR &&
                                       data.k_scale == data.v_scale &&
                                       (parameters.head_size == 256 || parameters.head_size == 128 || parameters.head_size == 64) &&
                                       (group_size == 4 || group_size == 8 || group_size == 16 || group_size == 32) &&
                                       (device_prop.major >= 9 || (device_prop.major == 8 && device_prop.minor == 9)));  // FP8 requires SM89+ (Ada Lovelace)
#else
    constexpr bool is_fp8_quantized_supported = false;
#endif

    bool is_non_quantized_supported = !is_inputs_quantized &&
                                      (parameters.head_size == 256 || parameters.head_size == 128 || parameters.head_size == 64) &&
                                      (group_size == 1 || group_size == 2 || group_size == 4 || group_size == 5 ||
                                       group_size == 8 || group_size == 16 || group_size == 32);

    data.use_xqa = (is_non_quantized_supported || is_int8_quantized_supported || is_fp8_quantized_supported);

    if (data.use_xqa) {
      // Consumer Blackwell (sm_120) and some other devices expose a smaller per-block opt-in
      // shared-memory limit than the SM the XQA kernel was compiled / JIT-compiled for (sm_80 and
      // sm_90 use a larger K/V-tile layout, ~140 KB for head_size 128/256). Launching XQA there
      // fails with cudaErrorInvalidValue because cudaFuncSetAttribute requests more shared memory
      // than the device allows. Query the actual kernel requirement (read from the device symbol,
      // so it is accurate even for a PTX kernel JIT-compiled for this SM) and fall back to cuDNN
      // SDPA / Flash when it does not fit. Cached per node because head_size and group size are
      // constant for a given GQA node (avoids a device->host copy on every decode step).
      int xqa_smem_ok = xqa_shared_memory_ok_.load(std::memory_order_relaxed);
      if (xqa_smem_ok < 0) {
        // GetXQARequiredSharedMemoryBytes issues a cudaMemcpyFromSymbol, which synchronizes and is
        // therefore illegal during CUDA graph capture (it would invalidate the capture). The result
        // is invariant for a node (fixed head_size / group size / device), and ORT performs at least
        // one non-captured warm-up run before capturing, so this normally resolves and caches during
        // warm-up. Guard defensively: only issue the synchronizing query when the compute stream is
        // not capturing. If we somehow reach here for the first time while capturing, conservatively
        // skip XQA for this run instead of breaking the capture, and leave the cache unresolved so a
        // later non-capturing run can determine it.
        if (!onnxruntime::llm::common::isCapturing(Stream(context))) {
          const size_t required_smem = onnxruntime::contrib::cuda::GetXQARequiredSharedMemoryBytes(
              device_prop, parameters.head_size, parameters.num_heads, parameters.kv_num_heads);
          xqa_smem_ok = (required_smem == 0 || required_smem <= device_prop.sharedMemPerBlockOptin) ? 1 : 0;
          xqa_shared_memory_ok_.store(xqa_smem_ok, std::memory_order_relaxed);
        } else {
          xqa_smem_ok = 0;  // capturing (or status query failed) and not yet resolved -> fall back
        }
      }
      data.use_xqa = (xqa_smem_ok != 0);
    }

    if (data.use_xqa) {
      size_t xqa_internal_bytes = onnxruntime::contrib::cuda::GetXQAScratchSize(
          GetDeviceProp(),
          parameters.batch_size,
          parameters.num_heads,
          parameters.kv_num_heads,
          parameters.head_size,
          parameters.seqlen_present_kv_cache,
          parameters.k_quant_type != KVQuantizationType::NONE ? (is_fp8 ? XqaQuantType::kFp8 : XqaQuantType::kInt8) : XqaQuantType::kNone,
          std::is_same<T, BFloat16>::value);
      assert(xqa_internal_bytes > 0);
      // Calculate additional scratch needed for manual RoPE/Append in ExtremeDecoding
      size_t xqa_total_bytes = xqa_internal_bytes;
      size_t q_bytes = 0;
      size_t k_bytes = 0;
      if (parameters.do_rotary) {
        // 1. Q_rotated buffer: B * N * H * sizeof(T) (if rotary)
        // 2. K_rotated buffer: B * Nk * H * sizeof(T) (if rotary)
        size_t element_size = sizeof(CudaT);
        q_bytes = parameters.batch_size * parameters.num_heads * parameters.head_size * element_size;
        k_bytes = parameters.batch_size * parameters.kv_num_heads * parameters.head_size * element_size;
        q_bytes = (q_bytes + 255) / 256 * 256;
        k_bytes = (k_bytes + 255) / 256 * 256;
        xqa_total_bytes += q_bytes + k_bytes;
      }
      const bool use_prepacked_xqa_head_sink =
          use_xqa_attention_sinks && xqa_head_sink_ != nullptr && xqa_head_sink_count_ == parameters.num_heads;
      const bool convert_xqa_head_sink = use_xqa_attention_sinks && !use_prepacked_xqa_head_sink;
      size_t xqa_head_sink_bytes = 0;
      if (convert_xqa_head_sink) {
        // No prepacked FP32 head_sink (dynamic input): reserve scratch for the per-launch conversion.
        xqa_head_sink_bytes = parameters.num_heads * sizeof(float);
        xqa_head_sink_bytes = (xqa_head_sink_bytes + 255) / 256 * 256;
        xqa_total_bytes += xqa_head_sink_bytes;
      }

      xqa_scratch_buffer = this->GetScratchBuffer<void>(xqa_total_bytes, GetComputeStream(context));
      data.xqa_buffer = xqa_scratch_buffer.get();
      data.xqa_buffer_bytes = xqa_internal_bytes;

      char* xqa_extra_buffer = reinterpret_cast<char*>(data.xqa_buffer) + xqa_internal_bytes;
      if (parameters.do_rotary) {
        data.qkv_buffer = reinterpret_cast<CudaT*>(xqa_extra_buffer);
        xqa_extra_buffer += q_bytes + k_bytes;
      }
      if (use_prepacked_xqa_head_sink) {
        data.xqa_head_sink = xqa_head_sink_.get();
      } else if (convert_xqa_head_sink) {
        data.xqa_head_sink = reinterpret_cast<float*>(xqa_extra_buffer);
        data.xqa_head_sink_needs_conversion = true;
      }
    }
  }

  // === cuDNN SDPA eligibility (preferred on SM>=90, Hopper/Blackwell) ===
  // Constrained to the well-supported causal path: non-quantized FP16/BF16 KV cache, no softcap,
  // no smooth-softmax / head sink, and no sliding window. Rotary and packed QKV are handled by
  // PrepareQKV before the kernel runs; cuDNN handles grouped-query attention natively.
  bool use_cudnn_sdpa = !data.use_xqa &&
                        !has_attention_bias &&  // GQA's cuDNN path is bottom-right causal, which cuDNN doesn't compose with a bias
                        !is_inputs_quantized &&
                        std::is_same<T, U>::value &&
                        parameters.softcap == 0.0f &&
                        !parameters.use_smooth_softmax &&
                        head_sink == nullptr &&
                        parameters.local_window_size == -1 &&
                        parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH &&
                        (enable_cudnn_flash_attention_ ||
                         (auto_enable_cudnn_flash_attention_ && device_prop.major >= 9)) &&
                        onnxruntime::cudnn_sdpa::is_stable() &&
                        onnxruntime::cudnn_sdpa::is_supported(device_prop,
                                                              parameters.num_heads,
                                                              parameters.kv_num_heads,
                                                              parameters.head_size,
                                                              parameters.head_size,
                                                              parameters.sequence_length,          // seq_len_q
                                                              parameters.seqlen_present_kv_cache,  // seq_len_kv (capacity)
                                                              /*is_causal=*/true);
  data.use_cudnn_sdpa = use_cudnn_sdpa;

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !data.use_xqa &&
                             !data.use_cudnn_sdpa &&
                             !has_attention_bias &&  // flash_api.h has no bias parameter
                             !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported<T>(device_prop,
                                                                 parameters.head_size,
                                                                 parameters.num_heads,
                                                                 parameters.kv_num_heads);

  data.use_flash_attention = use_flash_attention;
  // The fast-decode path lets the flash kernel perform RoPE and KV-append internally, bypassing
  // PrepareQKV (and therefore the fused QK-Norm prologue). Disable it when q/k norm weights are
  // present so the regular FlashAttention path (which normalizes via PrepareQKV) is used instead.
  data.use_flash_attention_fast_decode = use_flash_attention && !disable_flash_decode_ && !parameters.is_first_prompt && parameters.kv_sequence_length > 0 && parameters.past_present_share_buffer && !is_inputs_quantized && !parameters.use_qk_norm;

  if (use_flash_attention) {
    // Allocate Flash specific buffers (Softmax LSE, Accum)
    size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);

    int num_heads_for_split = data.use_flash_attention_fast_decode ? parameters.kv_num_heads : parameters.num_heads;
    size_t sequence_length_for_split = static_cast<size_t>(parameters.total_sequence_length);
    if (data.use_flash_attention_fast_decode && parameters.local_window_size > 0) {
      sequence_length_for_split = std::min(sequence_length_for_split, static_cast<size_t>(parameters.local_window_size));
    }

    auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, sequence_length_for_split, num_heads_for_split,
        parameters.head_size, device_prop.multiProcessorCount);

    parameters.num_splits = static_cast<int>(num_splits);

    if (data.use_flash_attention_fast_decode && num_splits > 1) {
      // The heuristic used kv_num_heads to maximize occupancy for the GQA-aware kernel.
      // However, the LSE and Accum buffers must store results for ALL num_heads.
      softmax_lse_accum_bytes = onnxruntime::flash::get_softmax_lse_accum_size(num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length);
      auto round_multiple = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
      out_accum_bytes = onnxruntime::flash::get_out_accum_size(num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length, round_multiple(parameters.head_size, 32));
    }

    softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, GetComputeStream(context));
    softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, GetComputeStream(context));
    out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, GetComputeStream(context));

    auto cuda_stream = Stream(context);
    if (softmax_lse_accum_bytes > 0) {
      // Initialize to 0 is fine because Flash kernel will write -inf to it if needed.
      // However, the standard Flash kernel often doesn't zero it globally.
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(softmax_lse_accum_buffer.get(), 0, softmax_lse_accum_bytes, cuda_stream));
    }
    if (out_accum_bytes > 0) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(out_accum_buffer.get(), 0, out_accum_bytes, cuda_stream));
    }

    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }
#endif

  if (data.use_flash_attention_fast_decode && parameters.sequence_length == 1) {
    // FlashDecoding Fast Path:
    // - Uses Flash Attention's internal KV append logic, so total_seq_lens and padded_seq_lens are not needed.
    // - The input seqlens_k from ONNX graph is (total_len - 1), which equals past_seq_len when seq_len == 1.
    // - This optimization avoids launching GetSequenceLengths kernel for single-token decoding.
    data.past_seq_lens = const_cast<int*>(total_seq_lens_minus_one->Data<int>());
  } else {
    // Compute sequence length buffers (past_seq_lens and total_seq_lens).
    // Allocate buffer for both: first half is past_seq_lens, second half is total_seq_lens.
    seq_lens_buffer = GetScratchBuffer<int>(3 * parameters.batch_size, GetComputeStream(context));
    auto cuda_stream = Stream(context);
    data.past_seq_lens = seq_lens_buffer.get();
    data.total_seq_lens = seq_lens_buffer.get() + parameters.batch_size;
    data.padded_seq_lens = data.total_seq_lens + parameters.batch_size;
    ORT_RETURN_IF_ERROR(LaunchGetSequenceLengths(total_seq_lens_minus_one->Data<int>(),
                                                 data.past_seq_lens,
                                                 data.total_seq_lens,
                                                 data.padded_seq_lens,
                                                 parameters.batch_size,
                                                 parameters.sequence_length,
                                                 parameters.is_first_prompt,
                                                 cuda_stream,
                                                 device_prop.maxThreadsPerBlock));
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("total_seq_lens", data.total_seq_lens, parameters.batch_size, 1);
    DUMP_TENSOR("past_seq_lens", data.past_seq_lens, parameters.batch_size, 1);
    DUMP_TENSOR("padded_seq_lens", data.padded_seq_lens, parameters.batch_size, 1);
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (!data.use_xqa && !data.use_cudnn_sdpa && !data.use_flash_attention) {
    // Fall back to memory efficient attention.
    int sm = (device_prop.major * 10) + device_prop.minor;
    // With attention_bias, MEA is skipped: the cutlass wrapper computes the bias row stride from
    // kv_sequence_length, which GQA sets to the KV-cache capacity (seqlen_present_kv_cache), not
    // the bias row length (total_sequence_length) — mismatched under past/present buffer sharing.
    // Bias-carrying nodes take the unfused fallback below instead.
    bool use_memory_efficient_attention =
        !disable_memory_efficient_attention_ &&
        !has_attention_bias &&
        has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value, parameters.head_size, parameters.head_size);
    data.use_memory_efficient_attention = use_memory_efficient_attention;

    // KV buffer for head expansion (when num_heads != kv_num_heads)
    size_t kv_buffer_bytes = (use_memory_efficient_attention && (parameters.num_heads != parameters.kv_num_heads))
                                 ? (sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.seqlen_present_kv_cache * parameters.head_size)
                                 : 0;
    // FMHA workspace
    size_t fmha_buffer_bytes = (use_memory_efficient_attention && MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float)))
                                   ? (sizeof(float) * parameters.batch_size * parameters.sequence_length * parameters.num_heads * parameters.head_size)
                                   : 0;

    k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, GetComputeStream(context));
    v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, GetComputeStream(context));
    fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, GetComputeStream(context));

    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
  }
#endif

  // -------------
  // Centralized scratch buffer allocation using GQABufferRequirements
  // This ensures allocation logic stays in sync with kernel usage
  auto buffer_req = GQABufferRequirements::Compute<T>(
      parameters,
      data.use_xqa,
      data.use_flash_attention,
      data.use_flash_attention_fast_decode,
      data.use_memory_efficient_attention);

  if (buffer_req.qkv_buffer_bytes > 0) {
    unpacked_qkv_buffer = GetScratchBuffer<void>(buffer_req.qkv_buffer_bytes, GetComputeStream(context));
    data.qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
  }

  // ---------------------------------------------------------------------
  // GQA-capable unfused fallback (issue #28195).
  // Activates when Flash / MEA / XQA are all ineligible and KV is not quantized.
  // Supports any head_size (FP32 QK accumulation), GQA, sliding window, softcap.
  // See LaunchUnfusedAttention in contrib_ops/cuda/bert/unfused_attention.h.
  // ---------------------------------------------------------------------
  IAllocatorUniquePtr<void> unfused_scratch;
  if (!data.use_xqa && !data.use_cudnn_sdpa && !data.use_flash_attention && !data.use_memory_efficient_attention &&
      !is_inputs_quantized && !parameters.use_smooth_softmax && head_sink == nullptr &&
      parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH) {
    data.use_unfused = true;

    const size_t B = static_cast<size_t>(parameters.batch_size);
    const size_t N_q = static_cast<size_t>(parameters.num_heads);
    const size_t S_q = static_cast<size_t>(parameters.sequence_length);
    const size_t H = static_cast<size_t>(parameters.head_size);
    // GQA guarantees head_size == v_head_size; use H_v for the Y output buffer
    // so the allocation stays correct if a distinct v_head_size is ever exposed.
    const size_t H_v = (parameters.v_head_size > 0)
                           ? static_cast<size_t>(parameters.v_head_size)
                           : H;
    const size_t S_kv = static_cast<size_t>(parameters.total_sequence_length);

    auto align = [](SafeInt<size_t> v) -> SafeInt<size_t> {
      return ((v + SafeInt<size_t>(255)) / SafeInt<size_t>(256)) * SafeInt<size_t>(256);
    };
    const SafeInt<size_t> q_bnsh_bytes = align(SafeInt<size_t>(B) * N_q * S_q * H * sizeof(T));
    const SafeInt<size_t> y_bnsh_bytes = align(SafeInt<size_t>(B) * N_q * S_q * H_v * sizeof(T));
    const SafeInt<size_t> ws_bytes = SafeInt<size_t>(
        onnxruntime::contrib::cuda::GetUnfusedAttentionWorkspaceSize(
            static_cast<int>(B), static_cast<int>(N_q), static_cast<int>(S_q), static_cast<int>(S_kv)));
    const SafeInt<size_t> workspace_offset = q_bnsh_bytes + y_bnsh_bytes;

    unfused_scratch = GetScratchBuffer<void>(static_cast<size_t>(q_bnsh_bytes + y_bnsh_bytes + ws_bytes),
                                             GetComputeStream(context));
    auto* base = reinterpret_cast<uint8_t*>(unfused_scratch.get());
    data.unfused_q_bnsh = reinterpret_cast<CudaT*>(base);
    data.unfused_y_bnsh = reinterpret_cast<CudaT*>(base + static_cast<size_t>(q_bnsh_bytes));
    data.unfused_workspace = reinterpret_cast<void*>(base + static_cast<size_t>(workspace_offset));
  }

  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_xqa = data.use_xqa;
    debug_info.use_flash_attention = data.use_flash_attention;
    debug_info.use_efficient_attention = data.use_memory_efficient_attention;
    debug_info.use_cudnn_flash_attention = data.use_cudnn_sdpa;

    debug_info.Print("GroupQueryAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }

  if (head_sink != nullptr) {
    data.head_sink = reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  }

  if (has_attention_bias) {
    data.attention_bias = reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
  }

  if (parameters.use_qk_norm) {
    data.q_norm_weight = reinterpret_cast<const CudaT*>(q_norm_weight->Data<T>());
    data.k_norm_weight = reinterpret_cast<const CudaT*>(k_norm_weight->Data<T>());
    data.qk_norm_epsilon = qk_norm_epsilon_;
  }

#if DUMP_TENSOR_LEVEL > 0
  DUMP_TENSOR_INIT();
  // Dump Scales
  if (data.k_scale) {
    if (parameters.k_quant_type == KVQuantizationType::PER_TENSOR) {
      DUMP_TENSOR("k_scale", data.k_scale, 1, 1);
    } else if (parameters.k_quant_type == KVQuantizationType::PER_CHANNEL) {
      DUMP_TENSOR("k_scale", data.k_scale, parameters.kv_num_heads, 1, parameters.head_size);
    }
  }
  if (data.v_scale) {
    if (parameters.v_quant_type == KVQuantizationType::PER_TENSOR) {
      DUMP_TENSOR("v_scale", data.v_scale, 1, 1);
    } else if (parameters.v_quant_type == KVQuantizationType::PER_CHANNEL) {
      DUMP_TENSOR("v_scale", data.v_scale, parameters.kv_num_heads, 1, parameters.head_size);
    }
  }
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  if (data.use_cudnn_sdpa) {
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&data.allocator));
    data.cudnn_handle = static_cast<void*>(GetCudnnHandle(context));
  }

  ORT_RETURN_IF_ERROR((QkvToContext<CudaT, CudaU>(
      device_prop, cublas, ort_stream.get(), parameters, data)));
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
