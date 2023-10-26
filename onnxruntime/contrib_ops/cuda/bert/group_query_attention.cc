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

#define REGISTER_KERNEL_TYPED(T)                                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                 \
      GroupQueryAttention,                                                                                       \
      kMSDomain,                                                                                                 \
      1,                                                                                                         \
      T,                                                                                                         \
      kCudaExecutionProvider,                                                                                    \
      (*KernelDefBuilder::Create())                                                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                                 \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}) \
          .MayInplace(3, 1)                                                                                      \
          .MayInplace(4, 2)                                                                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 5),                                                               \
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
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 1) == 1;
  local_window_size_ = info.GetAttrOrDefault<int64_t>("local_window_size", -1);
  is_past_bsnh_ = info.GetAttrOrDefault<int64_t>("is_past_bsnh", 1) == 1;
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
  const Tensor* past_seq_len = context->Input<Tensor>(5);

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
                                                                past_seq_len,
                                                                is_past_bsnh_,
                                                                scale_,
                                                                device_prop.maxThreadsPerBlock));
  if (local_window_size_ > 0 && !is_unidirectional_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "unidirectional must be true when using local (sliding window) attention.");
  }
  parameters.is_unidirectional = is_unidirectional_;
  parameters.local_window_size = local_window_size_;
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
  size_t seqlens_k_bytes = 0;
  if (use_flash_attention) {
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);
    // split kv buffers
    parameters.num_splits = onnxruntime::flash::num_splits_heuristic(
        parameters.batch_size, parameters.sequence_length, parameters.kv_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount, 128, false,
        device_prop.major == 8 && device_prop.minor > 0);
    if (parameters.num_splits > 1) {
      // softmax_lse_accum buffer
      softmax_lse_accum_bytes = onnxruntime::flash::get_softmax_lse_accum_size(
          parameters.num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length);
      // out_accum buffer
      auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int head_size_rounded = round_multiple(parameters.head_size, 32);
      out_accum_bytes = onnxruntime::flash::get_out_accum_size(
          parameters.num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length, head_size_rounded);
    }
    // seqlens_k buffer
    if (past_key != nullptr) {
      seqlens_k_bytes = sizeof(int) * parameters.batch_size;
    }
  }
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());
  auto seqlens_k_buffer = GetScratchBuffer<void>(seqlens_k_bytes, context->GetComputeStream());
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());        // nullptr
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());  // nullptr
  auto out_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());          // nullptr
  auto seqlens_k_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());          // nullptr
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  int sm = (device_prop.major * 10) + device_prop.minor;
  bool use_memory_efficient_attention =
      !use_flash_attention &&
      !disable_memory_efficient_attention_ &&
      (parameters.head_size & 7) == 0 &&
      parameters.sequence_length <= parameters.past_sequence_length + parameters.kv_sequence_length &&
      (sizeof(T) == 2 || parameters.sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32) &&
      has_memory_efficient_attention(sm, sizeof(T) == 2);
  // allocate buffers
  size_t kv_buffer_bytes = 0;
  // need a buffer if we must ungroup kv or if kv-cache is present
  const bool needs_buff = ((parameters.num_heads != parameters.kv_num_heads) ||
                           (past_key != nullptr && parameters.present_sequence_length != parameters.max_sequence_length));
  if (use_memory_efficient_attention && needs_buff) {
    kv_buffer_bytes = (sizeof(T) * parameters.batch_size * parameters.num_heads * (parameters.past_sequence_length + parameters.kv_sequence_length) * parameters.head_size);
  }
  size_t fmha_buffer_bytes = 0;
  if (use_memory_efficient_attention && MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float))) {
    fmha_buffer_bytes = (parameters.batch_size * parameters.sequence_length * parameters.num_heads * parameters.head_size * sizeof(float));
  }
  auto k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
  auto v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
  auto fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());
#else
  constexpr bool use_memory_efficient_attention = false;
  auto k_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
  auto v_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
  auto fmha_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());
#endif

  std::vector<int64_t> present_dims;
  int present_buff_seqlen = past_seq_len == nullptr ? parameters.present_sequence_length : parameters.max_sequence_length;
  if (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    present_dims = {
        parameters.batch_size, present_buff_seqlen, parameters.kv_num_heads, parameters.head_size};
  } else {  // BNSH
    present_dims = {
        parameters.batch_size, parameters.kv_num_heads, present_buff_seqlen, parameters.head_size};
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
  if (seqlens_k_buffer != nullptr) {
    data.seqlens_k = reinterpret_cast<int*>(seqlens_k_buffer.get());
  }
  if (k_buffer != nullptr) {
    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
  }
  if (fmha_buffer != nullptr) {
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
