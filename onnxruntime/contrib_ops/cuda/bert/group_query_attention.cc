// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cuda/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      GroupQueryAttention,                                               \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int64_t>()}) \
          .MayInplace(3, 1)                                              \
          .MayInplace(4, 2),                                             \
      GroupQueryAttention<T>);

// REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_unidirectional_ = true;
  is_past_bsnh_ = false;  // info.GetAttrOrDefault<int64_t>("is_past_bsnh", 1) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

#if USE_FLASH_ATTENTION
  disable_flash_attention_ = sizeof(T) != 2 ||
                             ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFlashAttention, false);
#else
  disable_flash_attention_ = true;
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  disable_memory_efficient_attention_ = sizeof(T) != 2 ||
                                        ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif
}

template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* attention_mask = context->Input<Tensor>(5);

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;
  typedef typename ToCudaType<T>::MappedType CudaT;
  GroupQueryAttentionData<CudaT> data;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                attention_mask,
                                                                is_past_bsnh_,
                                                                scale_,
                                                                device_prop.maxThreadsPerBlock));
  parameters.is_unidirectional = is_unidirectional_;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.kv_num_heads);
  // Allocate buffers
  size_t softmax_lse_bytes = 0;
  size_t softmax_lse_accum_bytes = 0;
  size_t out_accum_bytes = 0;
  if (use_flash_attention) {
    // softmax buffer
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);
    // split kv buffer
    using namespace std;
    auto [num_splits, slse_accum_bytes, o_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount);
    parameters.num_splits = num_splits;
    softmax_lse_accum_bytes = slse_accum_bytes;
    out_accum_bytes = o_accum_bytes;
  }
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());        // nullptr
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());  // nullptr
  auto out_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());          // nullptr
#endif

  if (!use_flash_attention) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GroupQueryAttention is only supported on ampere gpu and above");
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  int sm = (device_prop.major * 10) + device_prop.minor;
  bool use_memory_efficient_attention =
      !use_flash_attention &&
      !disable_memory_efficient_attention_ &&
      (parameters.head_size & 7) == 0 &&
      parameters.sequence_length <= parameters.seqlen_past_kv_cache + parameters.sequence_length &&
      (sizeof(T) == 2 || parameters.sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32) &&
      has_memory_efficient_attention(sm, sizeof(T) == 2);
  // allocate buffers
  size_t kv_buffer_bytes = 0;
  // need a buffer if we must ungroup kv
  const bool needs_buff = (parameters.num_heads != parameters.kv_num_heads);
  if (use_memory_efficient_attention && needs_buff) {
    kv_buffer_bytes = (sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.seqlen_present_kv_cache * parameters.head_size);
  }
  size_t fmha_buffer_bytes = 0;
  if (use_memory_efficient_attention && MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float))) {
    fmha_buffer_bytes = (parameters.batch_size * parameters.sequence_length * parameters.num_heads * parameters.head_size * sizeof(float));
  }
  // seqstart pointer for memory efficient
  size_t seqstart_k_bytes = 0;
  size_t seqstart_q_bytes = 0;
  if (use_memory_efficient_attention) {
    seqstart_k_bytes = sizeof(int32_t) * (parameters.batch_size + 1);
    seqstart_q_bytes = sizeof(int32_t) * (parameters.batch_size + 1);
  }
  auto k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
  auto v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
  auto fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());
  auto seqstart_k_buffer = GetScratchBuffer<void>(seqstart_k_bytes, context->GetComputeStream());
  auto seqstart_q_buffer = GetScratchBuffer<void>(seqstart_q_bytes, context->GetComputeStream());
#else
  constexpr bool use_memory_efficient_attention = false;
  auto k_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
  auto v_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
  auto fmha_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
  auto seqstart_k_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
  auto seqstart_q_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
#endif

  // seqlens_k buffer
  size_t seqlens_k_bytes = 0;
  seqlens_k_bytes = sizeof(int) * parameters.batch_size;
  auto seqlens_k_buffer = GetScratchBuffer<void>(seqlens_k_bytes, context->GetComputeStream());

  std::vector<int64_t> present_dims;
  if (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    present_dims = {
        parameters.batch_size, parameters.seqlen_present_kv_cache, parameters.kv_num_heads, parameters.head_size};
  } else {  // BNSH
    present_dims = {
        parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, parameters.head_size};
  }
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(value->Data<T>());
  data.past_key = (nullptr == past_key) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (nullptr == past_value) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  if (softmax_lse_buffer != nullptr) {
    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
  }
  if (softmax_lse_accum_buffer != nullptr) {
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
  }
  if (out_accum_buffer != nullptr) {
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }
  data.attention_mask = const_cast<int64_t*>(attention_mask->Data<int64_t>());
  if (seqlens_k_buffer != nullptr) {
    data.seqlens_k = reinterpret_cast<int*>(seqlens_k_buffer.get());
    if (seqstart_k_buffer != nullptr) {
      data.seqstart_k = reinterpret_cast<int32_t*>(seqstart_k_buffer.get());
    }
    if (seqstart_q_buffer != nullptr) {
      data.seqstart_q = reinterpret_cast<int32_t*>(seqstart_q_buffer.get());
    }
  }
  if (k_buffer != nullptr) {
    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
  }
  if (fmha_buffer != nullptr) {
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
  }
  if (k_buffer != nullptr) {
    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
  }

  if (data.past_key == data.present_key) {
    parameters.kv_share_buffer = true;
  } else {
    parameters.kv_share_buffer = false;
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
