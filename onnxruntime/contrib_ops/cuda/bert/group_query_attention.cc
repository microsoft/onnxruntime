// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
// Map string attribute to quantization type enum
KVQuantizationType StringToKVQuantizationType(const std::string& s) {
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

// Helper struct to manage buffers and pointers for quantization
template <typename T>
struct QuantizationData {
  using CudaT = typename ToCudaType<T>::MappedType;

  IAllocatorUniquePtr<T> dequantized_past_key_buffer;
  IAllocatorUniquePtr<T> dequantized_past_value_buffer;
  IAllocatorUniquePtr<T> dequantized_present_key_buffer;
  IAllocatorUniquePtr<T> dequantized_present_value_buffer;

  const CudaT* k_scale_ptr = nullptr;
  const CudaT* v_scale_ptr = nullptr;

  bool is_k_quantized = false;
  bool is_v_quantized = false;
};

// Helper function to handle dequantization of past_key and past_value
template <typename T>
Status HandleDequantization(
    const GroupQueryAttention<T>* kernel,
    OpKernelContext* context,
    const GroupQueryAttentionParameters& params,
    GroupQueryAttentionData<typename ToCudaType<T>::MappedType>& data,
    QuantizationData<T>& quant_data) {
  using CudaT = typename ToCudaType<T>::MappedType;

  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  Tensor* present_key = context->Output<Tensor>(1);
  Tensor* present_value = context->Output<Tensor>(2);
  const Tensor* k_scale = context->Input<Tensor>(12);
  const Tensor* v_scale = context->Input<Tensor>(13);

  auto stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
  size_t past_kv_size = static_cast<size_t>(params.batch_size) * params.kv_num_heads * params.seqlen_past_kv_cache * params.head_size;
  size_t present_kv_size = static_cast<size_t>(params.batch_size) * params.kv_num_heads * params.seqlen_present_kv_cache * params.head_size;

  quant_data.is_k_quantized = params.k_quant_type != KVQuantizationType::NONE;
  if (quant_data.is_k_quantized) {
    ORT_ENFORCE(past_key != nullptr && k_scale != nullptr, "past_key and k_scale must be provided for quantized KV cache.");
    quant_data.k_scale_ptr = reinterpret_cast<const CudaT*>(k_scale->Data<T>());

    quant_data.dequantized_past_key_buffer = kernel->template GetScratchBuffer<T>(past_kv_size, context->GetComputeStream());
    data.past_key = reinterpret_cast<const CudaT*>(quant_data.dequantized_past_key_buffer.get());

    Status status;
    if (params.kv_cache_bit_width == 8) {
      status = LaunchDequantizeKV<CudaT, int8_t, CudaT>(
          stream, reinterpret_cast<CudaT*>(quant_data.dequantized_past_key_buffer.get()),
          past_key->Data<int8_t>(), quant_data.k_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_past_kv_cache, params.sequence_length, params.head_size,
          true /*is_past*/, params.kv_cache_bit_width, params.k_quant_type);
    } else if (params.kv_cache_bit_width == 4) {
      status = LaunchDequantizeKV<CudaT, uint8_t, CudaT>(
          stream, reinterpret_cast<CudaT*>(quant_data.dequantized_past_key_buffer.get()),
          past_key->Data<uint8_t>(), quant_data.k_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_past_kv_cache, params.sequence_length, params.head_size,
          true /*is_past*/, params.kv_cache_bit_width, params.k_quant_type);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
    }
    ORT_RETURN_IF_ERROR(status);

    quant_data.dequantized_present_key_buffer = kernel->template GetScratchBuffer<T>(present_kv_size, context->GetComputeStream());
    data.present_key = reinterpret_cast<CudaT*>(quant_data.dequantized_present_key_buffer.get());
  } else {
    data.past_key = (nullptr == past_key) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  }

  quant_data.is_v_quantized = params.v_quant_type != KVQuantizationType::NONE;
  if (quant_data.is_v_quantized) {
    ORT_ENFORCE(past_value != nullptr && v_scale != nullptr, "past_value and v_scale must be provided for quantized KV cache.");
    quant_data.v_scale_ptr = reinterpret_cast<const CudaT*>(v_scale->Data<T>());

    quant_data.dequantized_past_value_buffer = kernel->template GetScratchBuffer<T>(past_kv_size, context->GetComputeStream());
    data.past_value = reinterpret_cast<const CudaT*>(quant_data.dequantized_past_value_buffer.get());

    Status status;
    if (params.kv_cache_bit_width == 8) {
      status = LaunchDequantizeKV<CudaT, int8_t, CudaT>(
          stream, reinterpret_cast<CudaT*>(quant_data.dequantized_past_value_buffer.get()),
          past_value->Data<int8_t>(), quant_data.v_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_past_kv_cache, params.sequence_length, params.head_size,
          true /*is_past*/, params.kv_cache_bit_width, params.v_quant_type);
    } else if (params.kv_cache_bit_width == 4) {
      status = LaunchDequantizeKV<CudaT, uint8_t, CudaT>(
          stream, reinterpret_cast<CudaT*>(quant_data.dequantized_past_value_buffer.get()),
          past_value->Data<uint8_t>(), quant_data.v_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_past_kv_cache, params.sequence_length, params.head_size,
          true /*is_past*/, params.kv_cache_bit_width, params.v_quant_type);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
    }
    ORT_RETURN_IF_ERROR(status);

    quant_data.dequantized_present_value_buffer = kernel->template GetScratchBuffer<T>(present_kv_size, context->GetComputeStream());
    data.present_value = reinterpret_cast<CudaT*>(quant_data.dequantized_present_value_buffer.get());
  } else {
    data.past_value = (nullptr == past_value) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
    data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  }

  return Status::OK();
}

// Helper function to handle requantization of present_key and present_value
template <typename T>
Status HandleRequantization(
    OpKernelContext* context,
    const GroupQueryAttentionParameters& params,
    const GroupQueryAttentionData<typename ToCudaType<T>::MappedType>& data,
    const QuantizationData<T>& quant_data) {
  using CudaT = typename ToCudaType<T>::MappedType;

  Tensor* present_key = context->Output<Tensor>(1);
  Tensor* present_value = context->Output<Tensor>(2);
  auto stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

  if (quant_data.is_k_quantized) {
    Status status;
    if (params.kv_cache_bit_width == 8) {
      status = LaunchQuantizeKV<CudaT, int8_t, CudaT>(
          stream, present_key->MutableData<int8_t>(), data.present_key, quant_data.k_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache, params.head_size,
          params.kv_cache_bit_width, params.k_quant_type);
    } else if (params.kv_cache_bit_width == 4) {
      status = LaunchQuantizeKV<CudaT, uint8_t, CudaT>(
          stream, present_key->MutableData<uint8_t>(), data.present_key, quant_data.k_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache, params.head_size,
          params.kv_cache_bit_width, params.k_quant_type);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
    }
    ORT_RETURN_IF_ERROR(status);
  }

  if (quant_data.is_v_quantized) {
    Status status;
    if (params.kv_cache_bit_width == 8) {
      status = LaunchQuantizeKV<CudaT, int8_t, CudaT>(
          stream, present_value->MutableData<int8_t>(), data.present_value, quant_data.v_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache, params.head_size,
          params.kv_cache_bit_width, params.v_quant_type);
    } else if (params.kv_cache_bit_width == 4) {
      status = LaunchQuantizeKV<CudaT, uint8_t, CudaT>(
          stream, present_value->MutableData<uint8_t>(), data.present_value, quant_data.v_scale_ptr, data.seqlens_k,
          params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache, params.head_size,
          params.kv_cache_bit_width, params.v_quant_type);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
    }
    ORT_RETURN_IF_ERROR(status);
  }

  return Status::OK();
}

}  // namespace

#define REGISTER_KERNEL_TYPED(T)                                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                      \
      GroupQueryAttention,                                                            \
      kMSDomain,                                                                      \
      1,                                                                              \
      T,                                                                              \
      kCudaExecutionProvider,                                                         \
      (*KernelDefBuilder::Create())                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                      \
          .TypeConstraint("T_CACHE",                                                  \
                          {DataTypeImpl::GetTensorType<T>(),                          \
                           DataTypeImpl::GetTensorType<int8_t>(),                     \
                           DataTypeImpl::GetTensorType<uint8_t>()})                   \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()})              \
          .MayInplace(3, 1)                        /* past_key and present_key */     \
          .MayInplace(4, 2)                        /* past_value and present_value */ \
          .InputMemoryType(OrtMemTypeCPUInput, 6), /* total_sequence_length */        \
      GroupQueryAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;
  is_unidirectional_ = true;
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;

  k_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("k_quant_type", "NONE"));
  v_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("v_quant_type", "NONE"));
  kv_cache_bit_width_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_cache_bit_width", 0));
  ORT_ENFORCE(kv_cache_bit_width_ == 0 || kv_cache_bit_width_ == 4 || kv_cache_bit_width_ == 8,
              "kv_cache_bit_width must be 0 (no quantization), 4 or 8.");

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

  // Memory efficient attention only supports float and float16, not bfloat16.
  disable_memory_efficient_attention_ = std::is_same<T, BFloat16>::value || !kernel_options_->UseEfficientAttention();

  if (!disable_flash_attention_) {
    zeros_ = this->GetScratchBuffer<int>(kZerosCount, nullptr);
    CUDA_CALL_THROW(cudaMemset(zeros_.get(), 0, kZerosCount * sizeof(int)));
  }
}

template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* seqlens_k = context->Input<Tensor>(5);
  const Tensor* total_seqlen = context->Input<Tensor>(6);
  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);
  const Tensor* position_ids = context->Input<Tensor>(9);
  const Tensor* attention_bias = context->Input<Tensor>(10);
  const Tensor* head_sink = context->Input<Tensor>(11);
  const Tensor* k_scale = context->Input<Tensor>(12);
  const Tensor* v_scale = context->Input<Tensor>(13);

  if (position_ids != nullptr || attention_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "position_ids and attention_bias are not supported in GroupQueryAttention cuda kernel.");
  }

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;
  typedef typename ToCudaType<T>::MappedType CudaT;
  GroupQueryAttentionData<CudaT> data;

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
                                                                seqlens_k,
                                                                total_seqlen,
                                                                scale_,
                                                                softcap_,
                                                                kv_cache_bit_width_,
                                                                device_prop.maxThreadsPerBlock));

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               parameters));
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
      static_cast<int>(Info().GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)))));

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to GroupQueryAttention when do_rotary = 1");
  }

  // ========================================================================
  // Input Validation for Quantized KV Cache
  // ========================================================================

  // Validate INT4 quantization requires even head_size
  if (kv_cache_bit_width_ == 4) {
    if (parameters.head_size % 2 != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "INT4 quantized KV cache requires head_size to be even. Got head_size=",
                             parameters.head_size,
                             ". INT4 packs 2 values per byte, so odd head_size is not supported.");
    }
  }

  // Validate quantized cache requires scale tensors
  bool has_quantized_k = (k_quant_type_ != KVQuantizationType::NONE);
  bool has_quantized_v = (v_quant_type_ != KVQuantizationType::NONE);

  if (has_quantized_k || has_quantized_v) {
    // If using quantization, kv_cache_bit_width must be set
    if (kv_cache_bit_width_ == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "When k_quant_type or v_quant_type is not NONE, kv_cache_bit_width must be 4 or 8.");
    }

    // If using quantized K cache, k_scale must be provided when past_key exists
    if (has_quantized_k && past_key != nullptr && k_scale == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "k_scale must be provided when past_key is quantized (k_quant_type is not NONE).");
    }

    // If using quantized V cache, v_scale must be provided when past_value exists
    if (has_quantized_v && past_value != nullptr && v_scale == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "v_scale must be provided when past_value is quantized (v_quant_type is not NONE).");
    }
  }

  // Validate scale tensors are only provided when quantization is enabled
  if (k_scale != nullptr && !has_quantized_k) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "k_scale is provided but k_quant_type is NONE. Set k_quant_type to enable quantization.");
  }

  if (v_scale != nullptr && !has_quantized_v) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "v_scale is provided but v_quant_type is NONE. Set v_quant_type to enable quantization.");
  }

  // ========================================================================

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Set up present KV output shapes
  std::vector<int64_t> present_dims = {
      parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, parameters.head_size};

  // Update present shape when kv cache has quantization.
  if (kv_cache_bit_width_ == 4) {
    present_dims[3] = (present_dims[3] + 1) / 2;
  }

  TensorShape present_shape(present_dims);

  context->Output(1, present_shape);  // present_key
  context->Output(2, present_shape);  // present_value

  QuantizationData<T> quant_data;
  IAllocatorUniquePtr<void> k_buffer;
  IAllocatorUniquePtr<void> v_buffer;
  IAllocatorUniquePtr<void> rotary_buffer;
  IAllocatorUniquePtr<void> fmha_buffer;
  IAllocatorUniquePtr<void> unpacked_qkv_buffer;

  // Flash Attention buffers
  IAllocatorUniquePtr<void> softmax_lse_buffer;
  IAllocatorUniquePtr<void> softmax_lse_accum_buffer;
  IAllocatorUniquePtr<void> out_accum_buffer;

  // Allocate seqlens_k_buffer here so it is available for both paths
  auto seqlens_k_buffer = GetScratchBuffer<void>(sizeof(int) * parameters.batch_size, context->GetComputeStream());
  data.seqlens_k_buff = reinterpret_cast<int*>(seqlens_k_buffer.get());

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.kv_num_heads) &&
                             onnxruntime::flash::is_supported_quantization(true /*is_causal*/,
                                                                           parameters.head_size,
                                                                           static_cast<int>(k_quant_type_),
                                                                           static_cast<int>(v_quant_type_),
                                                                           kv_cache_bit_width_);
#else
  constexpr bool use_flash_attention = false;
#endif

  if (use_flash_attention) {
    // --- FLASH ATTENTION PATH ---

    data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
    data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
    data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
    data.seqlens_k = const_cast<int*>(seqlens_k->Data<int>());

    data.k_scale = k_scale == nullptr ? nullptr : reinterpret_cast<const CudaT*>(k_scale->Data<T>());
    data.v_scale = v_scale == nullptr ? nullptr : reinterpret_cast<const CudaT*>(v_scale->Data<T>());

    // Handle Past/Present pointers handling quantization types
    if (k_quant_type_ != KVQuantizationType::NONE && past_key != nullptr) {
      if (kv_cache_bit_width_ == 8) {
        data.past_key = reinterpret_cast<const CudaT*>(past_key->Data<int8_t>());
        data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<int8_t>());
      } else {
        data.past_key = reinterpret_cast<const CudaT*>(past_key->Data<uint8_t>());
        data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<uint8_t>());
      }
    } else {
      data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
      data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<T>());
    }

    if (v_quant_type_ != KVQuantizationType::NONE && past_value != nullptr) {
      if (kv_cache_bit_width_ == 8) {
        data.past_value = reinterpret_cast<const CudaT*>(past_value->Data<int8_t>());
        data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<int8_t>());
      } else {
        data.past_value = reinterpret_cast<const CudaT*>(past_value->Data<uint8_t>());
        data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<uint8_t>());
      }
    } else {
      data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
      data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<T>());
    }

    data.use_flash_attention = true;
    data.use_memory_efficient_attention = false;

    // Allocate Flash specific buffers (Softmax LSE, Accum)
    size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);
    auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.total_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount);
    parameters.num_splits = static_cast<int>(num_splits);

    softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
    softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
    out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());

    size_t unpacked_qkv_bytes = parameters.is_packed_qkv && (parameters.num_heads != parameters.kv_num_heads)
                                    ? (static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * (parameters.num_heads + 2 * parameters.kv_num_heads) * parameters.head_size * sizeof(T))
                                    : 0;
    if (unpacked_qkv_bytes > 0) {
      unpacked_qkv_buffer = GetScratchBuffer<void>(unpacked_qkv_bytes, context->GetComputeStream());
      data.unpacked_qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
    } else {
      data.unpacked_qkv_buffer = nullptr;
    }

  } else {
    // --- FALLBACK PATH ---

#if USE_MEMORY_EFFICIENT_ATTENTION
    int sm = (device_prop.major * 10) + device_prop.minor;
    bool use_memory_efficient_attention =
        !use_flash_attention &&
        !disable_memory_efficient_attention_ &&
        has_memory_efficient_attention(sm, sizeof(T) == 2, parameters.head_size, parameters.head_size);

    size_t kv_buffer_bytes = (use_memory_efficient_attention && (parameters.num_heads != parameters.kv_num_heads))
                                 ? (sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.seqlen_present_kv_cache * parameters.head_size)
                                 : 0;
    size_t rotary_buffer_bytes = use_memory_efficient_attention && do_rotary_
                                     ? (2 * sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.sequence_length * parameters.head_size +
                                        sizeof(int64_t) * parameters.batch_size * parameters.sequence_length)
                                     : 0;
    size_t fmha_buffer_bytes = (use_memory_efficient_attention && MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float)))
                                   ? (static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads * parameters.head_size * sizeof(float))
                                   : 0;
    size_t unpacked_qkv_bytes = use_memory_efficient_attention && parameters.is_packed_qkv
                                    ? (static_cast<size_t>(parameters.batch_size) * parameters.sequence_length * (parameters.num_heads + 2 * parameters.kv_num_heads) * parameters.head_size * sizeof(T))
                                    : 0;
    k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
    v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
    rotary_buffer = GetScratchBuffer<void>(rotary_buffer_bytes, context->GetComputeStream());
    fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());
    unpacked_qkv_buffer = GetScratchBuffer<void>(unpacked_qkv_bytes, context->GetComputeStream());
#else
    bool use_memory_efficient_attention = false;
#endif
    data.use_memory_efficient_attention = use_memory_efficient_attention;
    data.use_flash_attention = false;

    data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
    data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
    data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
    data.seqlens_k = const_cast<int*>(seqlens_k->Data<int>());

    data.k_scale = k_scale == nullptr ? nullptr : reinterpret_cast<const CudaT*>(k_scale->Data<T>());
    data.v_scale = v_scale == nullptr ? nullptr : reinterpret_cast<const CudaT*>(v_scale->Data<T>());

    ORT_RETURN_IF_ERROR(HandleDequantization(this, context, parameters, data, quant_data));

    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
    data.unpacked_qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
    data.rotary_buffer = reinterpret_cast<CudaT*>(rotary_buffer.get());
  }

  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_efficient_attention = data.use_memory_efficient_attention;

    debug_info.Print("GroupQueryAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  if (data.past_key == data.present_key) {
    parameters.kv_share_buffer = true;
  } else {
    parameters.kv_share_buffer = false;
  }
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }

  if (head_sink != nullptr) {
    data.head_sink = reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  }

#if DUMP_TENSOR_LEVEL > 0
  DUMP_TENSOR_INIT();
  // Dump Scales
  if (data.k_scale) {
    // Assuming scalar or small vector, dump first few elements
    DUMP_TENSOR("k_scale", data.k_scale, 1, 1, 1, 1);
  }

  // Dump Present Key (should contain past + new)
  // Note: Dimensions might need adjustment for int4 to avoid dump crash if size mismatch
  if (parameters.k_quant_type != KVQuantizationType::NONE) {
    // Just dump raw bytes
    if (parameters.kv_cache_bit_width == 4) {
      DUMP_TENSOR("present_key_quant_packed", reinterpret_cast<const int8_t*>(data.present_key),
                  parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, parameters.head_size / 2);
    } else {
      DUMP_TENSOR("present_key_quant", reinterpret_cast<const int8_t*>(data.present_key),
                  parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, parameters.head_size);
    }
  }
#endif

  cublasHandle_t cublas = GetCublasHandle(context);
  ORT_RETURN_IF_ERROR(QkvToContext<CudaT>(device_prop, cublas, context->GetComputeStream(), parameters, data));

  if (!use_flash_attention) {
    ORT_RETURN_IF_ERROR(HandleRequantization(context, parameters, data, quant_data));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
