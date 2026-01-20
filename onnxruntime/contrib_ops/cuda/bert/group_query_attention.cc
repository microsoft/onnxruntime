// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
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

  IAllocatorUniquePtr<T> dequantized_key_buffer;
  IAllocatorUniquePtr<T> dequantized_value_buffer;

  const float* k_scale_ptr = nullptr;
  const float* v_scale_ptr = nullptr;

  bool is_k_quantized = false;
  bool is_v_quantized = false;
};

// Helper function to dequantize a single KV cache tensor (K or V)
// Returns Status::OK() on success, or an error status on failure.
template <typename T, typename CudaT>
Status DequantizeIfNeeded(
    const GroupQueryAttention<T>* kernel,
    OpKernelContext* context,
    cudaStream_t stream,
    const GroupQueryAttentionParameters& params,
    const Tensor* past_tensor,
    Tensor* present_tensor,
    const Tensor* scale_tensor,
    KVQuantizationType quant_type,
    size_t present_kv_size,
    const int* past_seq_lens,
    IAllocatorUniquePtr<T>& dequantized_buffer,
    const float*& scale_ptr,
    const void*& past_data_ptr,
    void*& present_data_ptr,
    bool& is_quantized,
    const char* tensor_name) {
  is_quantized = quant_type != KVQuantizationType::NONE;

  if (is_quantized) {
    if (params.seqlen_past_kv_cache > 0) {
      ORT_ENFORCE(past_tensor != nullptr,
                  tensor_name, " must be provided for quantized KV cache with past data.");
    }
    ORT_ENFORCE(scale_tensor != nullptr,
                tensor_name, "_scale must be provided for quantized KV cache.");

    // Enforce shared buffer constraint
    if (params.seqlen_past_kv_cache > 0) {
      ORT_ENFORCE(past_tensor->DataRaw() == present_tensor->DataRaw(),
                  "For quantized KV cache, ", tensor_name, " and ", tensor_name, " must share the same buffer.");
    }

    scale_ptr = scale_tensor->Data<float>();
    dequantized_buffer = kernel->template GetScratchBuffer<T>(present_kv_size, context->GetComputeStream());

    // Set both past and present to the same dequantized buffer
    present_data_ptr = dequantized_buffer.get();
    past_data_ptr = present_data_ptr;

    if (params.seqlen_past_kv_cache > 0) {
      bool is_input_bsnh = params.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;
      Status status;
      if (params.kv_cache_bit_width == 8) {
        status = LaunchDequantizeKV<CudaT, int8_t, float>(
            stream, reinterpret_cast<CudaT*>(dequantized_buffer.get()),
            static_cast<const int8_t*>(past_tensor->DataRaw()), scale_ptr, past_seq_lens,
            params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache, params.head_size,
            params.kv_cache_bit_width, quant_type, is_input_bsnh);
      } else if (params.kv_cache_bit_width == 4) {
        status = LaunchDequantizeKV<CudaT, uint8_t, float>(
            stream, reinterpret_cast<CudaT*>(dequantized_buffer.get()),
            static_cast<const uint8_t*>(past_tensor->DataRaw()), scale_ptr, past_seq_lens,
            params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache, params.head_size,
            params.kv_cache_bit_width, quant_type, is_input_bsnh);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
      }
      ORT_RETURN_IF_ERROR(status);
    }
  } else {
    past_data_ptr = (past_tensor == nullptr) ? nullptr : past_tensor->DataRaw();
    present_data_ptr = (present_tensor == nullptr) ? nullptr : present_tensor->MutableDataRaw();
  }

  return Status::OK();
}

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

  // TODO: we only need allocate params.total_sequence_length instead of params.seqlen_present_kv_cache.
  //       however, that change the shape of present_key and present_value so it might need some change in the downstream
  //                that need set the sequence length properly
  // It's fine to use seqlen_present_kv_cache right now.
  // It simulates the input shape of fp16 case that past_key and present_key share buffers.
  size_t present_kv_size = static_cast<size_t>(params.batch_size) * params.kv_num_heads * params.seqlen_present_kv_cache * params.head_size;

  // Dequantize Key
  {
    Status status = DequantizeIfNeeded<T, CudaT>(
        kernel, context, stream, params,
        past_key, present_key, k_scale,
        params.k_quant_type, present_kv_size, data.past_seq_lens,
        quant_data.dequantized_key_buffer, quant_data.k_scale_ptr,
        data.past_key, data.present_key,
        quant_data.is_k_quantized, "past_key");
    ORT_RETURN_IF_ERROR(status);
  }

  // Dequantize Value
  {
    Status status = DequantizeIfNeeded<T, CudaT>(
        kernel, context, stream, params,
        past_value, present_value, v_scale,
        params.v_quant_type, present_kv_size, data.past_seq_lens,
        quant_data.dequantized_value_buffer, quant_data.v_scale_ptr,
        data.past_value, data.present_value,
        quant_data.is_v_quantized, "past_value");
    ORT_RETURN_IF_ERROR(status);
  }

  return Status::OK();
}

// Helper function to quantize a single KV cache tensor (K or V)
// Returns Status::OK() on success, or an error status on failure.
template <typename CudaT>
Status QuantizeIfNeeded(
    cudaStream_t stream,
    Tensor* present_tensor,
    const void* dequantized_data,
    const float* scale_ptr,
    const int* past_seq_lens,
    const int* total_seq_lens,
    const GroupQueryAttentionParameters& params,
    KVQuantizationType quant_type,
    bool is_quantized,
    bool is_output_bsnh) {
  if (!is_quantized) {
    return Status::OK();
  }

  Status status;
  if (params.kv_cache_bit_width == 8) {
    status = LaunchQuantizeKV<CudaT, int8_t, float>(
        stream, present_tensor->MutableData<int8_t>(),
        reinterpret_cast<const CudaT*>(dequantized_data), scale_ptr, past_seq_lens, total_seq_lens,
        params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache,
        params.seqlen_present_kv_cache, params.head_size,
        params.kv_cache_bit_width, quant_type, false, is_output_bsnh);
  } else if (params.kv_cache_bit_width == 4) {
    status = LaunchQuantizeKV<CudaT, uint8_t, float>(
        stream, present_tensor->MutableData<uint8_t>(),
        reinterpret_cast<const CudaT*>(dequantized_data), scale_ptr, past_seq_lens, total_seq_lens,
        params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache,
        params.seqlen_present_kv_cache, params.head_size,
        params.kv_cache_bit_width, quant_type, false, is_output_bsnh);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
  }
  return status;
}

// Helper function to append and quantize a single KV cache tensor (K or V)
// Returns Status::OK() on success, or an error status on failure.
// template <typename CudaT>
// Status QuantizeAppendIfNeeded(
//     cudaStream_t stream,
//     Tensor* present_tensor,
//     const CudaT* new_data,
//     const float* scale_ptr,
//     const int* total_seq_lens,
//     const GroupQueryAttentionParameters& params,
//     KVQuantizationType quant_type,
//     bool is_quantized,
//     bool is_output_bsnh) {
//   if (!is_quantized) {
//     return Status::OK();
//   }

//   Status status;
//   if (params.kv_cache_bit_width == 8) {
//     status = LaunchQuantizeAppendKV<CudaT, int8_t, float>(
//         stream, present_tensor->MutableData<int8_t>(),
//         new_data, scale_ptr, total_seq_lens,
//         params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache,
//         params.head_size, params.kv_cache_bit_width,
//         params.sequence_length, quant_type, false, is_output_bsnh);
//   } else if (params.kv_cache_bit_width == 4) {
//     status = LaunchQuantizeAppendKV<CudaT, uint8_t, float>(
//         stream, present_tensor->MutableData<uint8_t>(),
//         new_data, scale_ptr, total_seq_lens,
//         params.batch_size, params.kv_num_heads, params.seqlen_present_kv_cache,
//         params.head_size, params.kv_cache_bit_width,
//         params.sequence_length, quant_type, false, is_output_bsnh);
//   } else {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Quantized KV cache requires kv_cache_bit_width to be 4 or 8.");
//   }

//   return status;
// }

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

  bool is_output_bsnh = params.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH;

  // Quantize Key
  ORT_RETURN_IF_ERROR(QuantizeIfNeeded<CudaT>(
      stream, present_key, data.present_key, quant_data.k_scale_ptr,
      params.past_present_share_buffer ? data.past_seq_lens : nullptr,
      data.total_seq_lens, params, params.k_quant_type,
      quant_data.is_k_quantized, is_output_bsnh));

  // Quantize Value
  ORT_RETURN_IF_ERROR(QuantizeIfNeeded<CudaT>(
      stream, present_value, data.present_value, quant_data.v_scale_ptr,
      params.past_present_share_buffer ? data.past_seq_lens : nullptr,
      data.total_seq_lens, params, params.v_quant_type,
      quant_data.is_v_quantized, is_output_bsnh));

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
          .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>())         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()})              \
          .MayInplace(3, 1)                        /* past_key and present_key */     \
          .MayInplace(4, 2)                        /* past_value and present_value */ \
          .InputMemoryType(OrtMemTypeCPUInput, 6), /* total_sequence_length */        \
      GroupQueryAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

constexpr const char* kDisableFlashDecode = "ORT_DISABLE_FLASH_DECODE";

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

  query_dynamic_quant_ = ParseEnvironmentVariableWithDefault<int>("ORT_FLASH_ATTENTION_QUERY_DYNAMIC_QUANT", 0) != 0;

  enable_xqa_ = std::is_same_v<T, MLFloat16> && ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_XQA", 0) != 0;

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

  // Memory efficient attention supports float and float16. BFloat16 support added for SM80+.
  disable_memory_efficient_attention_ = !kernel_options_->UseEfficientAttention();

  if (!disable_flash_attention_) {
    zeros_ = this->GetScratchBuffer<int>(kZerosCount, nullptr);
    CUDA_CALL_THROW(cudaMemset(zeros_.get(), 0, kZerosCount * sizeof(int)));
  }

  disable_flash_decode_ = ParseEnvironmentVariableWithDefault<bool>(kDisableFlashDecode, false);
}

template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* context) const {
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

  if (k_quant_type_ != KVQuantizationType::NONE && k_scale == nullptr) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "k_scale must be provided when k_quant_type is not NONE");
  }
  if (k_quant_type_ != KVQuantizationType::NONE) {
    if (k_scale->DataType() != DataTypeImpl::GetType<float>()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "k_scale must be float tensor");
    }
  }

  if (v_quant_type_ != KVQuantizationType::NONE && v_scale == nullptr) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "v_scale must be provided when v_quant_type is not NONE");
  }
  if (v_quant_type_ != KVQuantizationType::NONE) {
    if (v_scale->DataType() != DataTypeImpl::GetType<float>()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "v_scale must be float tensor");
    }
  }

  if (attention_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "attention_bias is not supported in GroupQueryAttention cuda kernel.");
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
                                                                total_seq_lens_minus_one,
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
  parameters.query_dynamic_quant = query_dynamic_quant_;

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
    present_dims[3] = present_dims[3] / 2;
  }

  TensorShape present_shape(present_dims);

  context->Output(1, present_shape);  // present_key
  context->Output(2, present_shape);  // present_value

  QuantizationData<T> quant_data;
  quant_data.is_k_quantized = (k_quant_type_ != KVQuantizationType::NONE);
  quant_data.is_v_quantized = (v_quant_type_ != KVQuantizationType::NONE);

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
  quant_data.k_scale_ptr = data.k_scale;
  data.v_scale = v_scale == nullptr ? nullptr : reinterpret_cast<const float*>(v_scale->DataRaw());
  quant_data.v_scale_ptr = data.v_scale;

  // Handle Past/Present pointers handling quantization types
  // Note: For non-quantized (T) or when falling back, we use the standard T* pointers.
  if (k_quant_type_ != KVQuantizationType::NONE) {
    // Quantized Key Cache
    if (kv_cache_bit_width_ == 8) {
      data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<int8_t>());
      data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<int8_t>());
    } else {
      data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<uint8_t>());
      data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<uint8_t>());
    }
  } else {
    // Non-Quantized Key Cache
    data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<T>());
  }

  if (v_quant_type_ != KVQuantizationType::NONE) {
    // Quantized Value Cache
    if (kv_cache_bit_width_ == 8) {
      data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<int8_t>());
      data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<int8_t>());
    } else {
      data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<uint8_t>());
      data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<uint8_t>());
    }
  } else {
    // Non-Quantized Value Cache
    data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
    data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<T>());
  }

  // Compute past_present_share_buffer early since it's needed for flash attention path selection.
  // This compares the final pointer values after quantization handling.
  parameters.past_present_share_buffer = (data.past_key == data.present_key);

  bool use_quantized_kv = (k_quant_type_ != KVQuantizationType::NONE) || (v_quant_type_ != KVQuantizationType::NONE);

  // Allocate XQA scratch if needed (only for Flash Decoding path)
  IAllocatorUniquePtr<void> xqa_scratch_buffer;
  if (enable_xqa_) {
    int group_size = parameters.num_heads / parameters.kv_num_heads;
    bool is_int8_supported = (parameters.kv_cache_bit_width == 8 &&
                              k_quant_type_ == KVQuantizationType::PER_TENSOR &&
                              v_quant_type_ == KVQuantizationType::PER_TENSOR &&
                              data.k_scale == data.v_scale &&  // XQA requires k_scale and v_scale to be the same. Here requires k_scale and v_scale are same tensor.
                              parameters.head_size == 128 &&
                              (group_size == 8 || group_size == 16 || group_size == 32));
    bool is_non_quantized_supported = !use_quantized_kv &&
                                      parameters.head_size == 128 &&
                                      (64 % group_size == 0);

    data.use_xqa = !parameters.is_first_prompt &&
                   parameters.sequence_length == 1 &&
                   parameters.past_present_share_buffer &&
                   (is_non_quantized_supported || is_int8_supported) &&
                   parameters.softcap == 0.0f &&
                   !parameters.use_smooth_softmax &&
                   parameters.local_window_size == -1;

    if (data.use_xqa) {
      size_t xqa_scratch_bytes = onnxruntime::contrib::cuda::GetXQAScratchSize(
          GetDeviceProp(),
          parameters.batch_size,
          parameters.num_heads,
          parameters.kv_num_heads,
          parameters.seqlen_present_kv_cache);
      if (xqa_scratch_bytes > 0) {
        // Calculate additional scratch needed for manual RoPE/Append in ExtremeDecoding

        if (parameters.do_rotary) {
          // 1. Q_rotated buffer: B * N * H * sizeof(T) (if rotary)
          // 2. K_rotated buffer: B * Nk * H * sizeof(T) (if rotary)
          size_t element_size = sizeof(CudaT);
          size_t q_bytes = parameters.batch_size * parameters.num_heads * parameters.head_size * element_size;
          size_t k_bytes = parameters.batch_size * parameters.kv_num_heads * parameters.head_size * element_size;
          q_bytes = (q_bytes + 255) / 256 * 256;
          k_bytes = (k_bytes + 255) / 256 * 256;
          xqa_scratch_bytes += q_bytes + k_bytes;
        }

        xqa_scratch_buffer = this->GetScratchBuffer<void>(xqa_scratch_bytes, context->GetComputeStream());
        data.scratch = xqa_scratch_buffer.get();
        data.scratch_bytes = xqa_scratch_bytes;
      }
    }
  }

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !data.use_xqa &&
                             !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported<T>(device_prop,
                                                                 parameters.head_size,
                                                                 parameters.num_heads,
                                                                 parameters.kv_num_heads) &&
                             (!use_quantized_kv || (parameters.is_first_prompt && parameters.past_present_share_buffer));

  data.use_flash_attention = use_flash_attention;
  data.use_flash_attention_fast_decode = use_flash_attention && !disable_flash_decode_ && !parameters.is_first_prompt && parameters.past_present_share_buffer && (!use_quantized_kv);

  if (use_flash_attention) {
    // Allocate Flash specific buffers (Softmax LSE, Accum)
    size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);

    int num_heads_for_split = data.use_flash_attention_fast_decode ? parameters.kv_num_heads : parameters.num_heads;
    auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.total_sequence_length, num_heads_for_split,
        parameters.head_size, device_prop.multiProcessorCount);

    parameters.num_splits = static_cast<int>(num_splits);

    if (data.use_flash_attention_fast_decode && num_splits > 1) {
      // The heuristic used kv_num_heads to maximize occupancy for the GQA-aware kernel.
      // However, the LSE and Accum buffers must store results for ALL num_heads.
      softmax_lse_accum_bytes = onnxruntime::flash::get_softmax_lse_accum_size(num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length);
      auto round_multiple = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
      out_accum_bytes = onnxruntime::flash::get_out_accum_size(num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length, round_multiple(parameters.head_size, 32));
    }

    softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
    softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
    out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

    auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
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

  // Fallback to dequantize past kv + GQA + quantize present kv for paths WITHOUT optimized kernels.
  // For share_buffer=true cases, we have optimized paths:
  //   - First prompt: FlashAttentionAndQuantizeKV
  //   - Decoding: FlashAttentionWithQuantizeKV (Only supports Rotary cases currently)
  // Fallback is only needed for non-share_buffer cases (which are rare for quantized KV).
  bool use_dequantize_quantize_fallback = use_quantized_kv && !(data.use_xqa || data.use_flash_attention);
  if (use_quantized_kv && !use_dequantize_quantize_fallback) {
    const char* force_fallback_env = getenv("ORT_GQA_FORCE_FALLBACK");
    if (force_fallback_env && atoi(force_fallback_env) > 0) {
      use_dequantize_quantize_fallback = true;
    }
  }

  // use_dequantize_quantize_fallback is already computed above.
  if (data.use_flash_attention_fast_decode && parameters.sequence_length == 1 && !use_dequantize_quantize_fallback) {
    // FlashDecoding Fast Path:
    // - Uses Flash Attention's internal KV append logic, so total_seq_lens and padded_seq_lens are not needed.
    // - The input seqlens_k from ONNX graph is (total_len - 1), which equals past_seq_len when seq_len == 1.
    // - This optimization avoids launching GetSequenceLengths kernel for single-token decoding.
    data.past_seq_lens = const_cast<int*>(total_seq_lens_minus_one->Data<int>());
  } else {
    // Compute sequence length buffers (past_seq_lens and total_seq_lens).
    // Allocate buffer for both: first half is past_seq_lens, second half is total_seq_lens.
    seq_lens_buffer = GetScratchBuffer<int>(3 * parameters.batch_size, context->GetComputeStream());
    auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
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
  if (!data.use_xqa && !data.use_flash_attention) {
    // Fall back to memory efficient attention.
    int sm = (device_prop.major * 10) + device_prop.minor;
    bool use_memory_efficient_attention =
        !disable_memory_efficient_attention_ &&
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

    k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
    v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
    fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());

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

  if (buffer_req.unpacked_qkv_bytes > 0) {
    unpacked_qkv_buffer = GetScratchBuffer<void>(buffer_req.unpacked_qkv_bytes, context->GetComputeStream());
    data.unpacked_qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
  }
  if (buffer_req.rotary_buffer_bytes > 0) {
    rotary_buffer = GetScratchBuffer<void>(buffer_req.rotary_buffer_bytes, context->GetComputeStream());
    data.rotary_buffer = reinterpret_cast<CudaT*>(rotary_buffer.get());
  }

  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = data.use_flash_attention;
    debug_info.use_efficient_attention = data.use_memory_efficient_attention;

    debug_info.Print("GroupQueryAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  // Validate past_value pointer consistency (past_present_share_buffer was computed early after pointer setup)
  if (parameters.past_present_share_buffer) {
    ORT_ENFORCE(data.past_value == data.present_value, "past_value and present_value must be the same tensor when past_present_share_buffer is true");
  } else {
    ORT_ENFORCE(data.past_value != data.present_value, "past_value and present_value must be different tensors when past_present_share_buffer is false");
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

  if (use_dequantize_quantize_fallback) {
    // When falling back, we have already dequantized the KV cache to float/bfloat16.
    // We want the attention kernel to treat this as standard float/bfloat16 attention.
    // So we create a copy of parameters and disable quantization flags.
    GroupQueryAttentionParameters fallback_params = parameters;
    fallback_params.k_quant_type = KVQuantizationType::NONE;
    fallback_params.v_quant_type = KVQuantizationType::NONE;
    fallback_params.kv_cache_bit_width = 0;

    // Fallback uses a single dequantized buffer for past and present (shared),
    // simulating the behavior of share_buffer=true for the attention kernel.
    fallback_params.past_present_share_buffer = true;

    // We also need to perform the dequantization here before calling the kernel.
    // Note: data.past_key and data.present_key are updated by HandleDequantization to point to temp buffers.
    ORT_RETURN_IF_ERROR(HandleDequantization(this, context, parameters, data, quant_data));

    // Call QkvToContext with dequantized buffers and original float parameters.
    auto stream = context->GetComputeStream();
    ORT_RETURN_IF_ERROR(QkvToContext<CudaT>(device_prop, cublas, stream, fallback_params, data));

    // Requantize the present KV cache back to int4/int8 for subsequent steps.
    return HandleRequantization(context, parameters, data, quant_data);
  } else {
    ORT_RETURN_IF_ERROR(QkvToContext<CudaT>(
        device_prop, cublas, context->GetComputeStream(), parameters, data));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
