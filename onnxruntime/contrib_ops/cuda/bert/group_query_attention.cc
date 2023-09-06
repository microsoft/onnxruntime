// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cuda/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GroupQueryAttention,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GroupQueryAttention<T>);

// REGISTER_KERNEL_TYPED(float) // TODO(aciddelgado): support regular float?
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : CudaKernel(info),
      fused_fp16_cross_attention_kernel_(nullptr),
      cumulated_sequence_length_q_cache_(),
      cumulated_sequence_length_kv_cache_() {
  int64_t num_heads = 0;
  int64_t num_heads_k = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("num_heads_k", &num_heads_k).IsOK() && num_heads_k > 0 && num_heads % num_heads_k == 0);
  num_heads_ = static_cast<int>(num_heads);
  num_heads_k_ = static_cast<int>(num_heads_k);
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 1) == 1;

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

#if USE_FLASH_ATTENTION
  disable_flash_attention_ = sizeof(T) != 2 ||
                             ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFlashAttention, false);
  min_seq_len_for_flash_attention_packed_qkv_ = ParseEnvironmentVariableWithDefault<int>(
      attention::kMinSeqLenForFlashAttentionPackedQKV,
      attention::kDefaultMinSeqLenForFlashAttentionPackedQKV);
#else
  disable_flash_attention_ = true;
  min_seq_len_for_flash_attention_packed_qkv_ = 0;
#endif

  // Allocate cache buffers
  constexpr size_t cache_bytes = sizeof(int32_t) * (static_cast<size_t>(kCumulatedSequenceLengthCacheMaxBatchSize) + 1);
  cumulated_sequence_length_q_cache_.buffer = GetTransientScratchBuffer<void>(cache_bytes);
  cumulated_sequence_length_q_cache_.max_batch_size = kCumulatedSequenceLengthCacheMaxBatchSize;
  cumulated_sequence_length_kv_cache_.buffer = GetTransientScratchBuffer<void>(cache_bytes);
  cumulated_sequence_length_kv_cache_.max_batch_size = kCumulatedSequenceLengthCacheMaxBatchSize;
}

template <typename T>
Status MultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(6); // TODO(aciddelgado): support past kv??
  const Tensor* past_value = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs<Tensor>(query,
                                                                        key,
                                                                        value,
                                                                        bias,
                                                                        past_key,
                                                                        past_value,
                                                                        &parameters,
                                                                        num_heads_,
                                                                        num_heads_k_,
                                                                        scale_,
                                                                        device_prop.maxThreadsPerBlock));
  parameters.is_unidirectional = is_unidirectional_;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.kv_hidden_size); // TODO(accidelgado): kv_hidden_size has kv_num_heads, is this correct output shape!?
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      parameters.batch_size, parameters.num_heads, parameters.total_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  MHARunner* fused_runner = nullptr;

  const FusedMultiHeadCrossAttentionKernel* fused_cross_attention_kernel = nullptr;

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.kv_num_heads);
#else
  constexpr bool use_flash_attention = false;
#endif

  constexpr size_t element_size = sizeof(T);
  size_t workspace_bytes = GetAttentionWorkspaceSize(element_size,
                                              parameters.batch_size,
                                              parameters.num_heads,
                                              parameters.kv_num_heads,
                                              parameters.head_size,
                                              parameters.sequence_length,
                                              parameters.kv_sequence_length,
                                              parameters.total_sequence_length,
                                              use_flash_attention);

  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  const size_t past_kv_bytes = element_size * parameters.batch_size * parameters.kv_sequence_length * parameters.kv_num_heads * parameters.head_size;
  const bool use_temp_k_v_workspace = use_flash_attention;
  auto temp_k_work_space = use_temp_k_v_workspace ? GetScratchBuffer<void>(past_kv_bytes, context->GetComputeStream()) : nullptr;
  auto temp_v_work_space = use_temp_k_v_workspace ? GetScratchBuffer<void>(past_kv_bytes, context->GetComputeStream()) : nullptr;

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = nullptr;
  data.bias = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = nullptr;
  data.mask_index_dims = gsl::span<const int64_t>();
  data.past = nullptr;
  data.past_key = (nullptr == past_key) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (nullptr == past_value) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.has_qkv_workspace = true;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.temp_k_workspace = use_temp_k_v_workspace ? reinterpret_cast<CudaT*>(temp_k_work_space.get()) : nullptr;
  data.temp_v_workspace = use_temp_k_v_workspace ? reinterpret_cast<CudaT*>(temp_v_work_space.get()) : nullptr;
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = nullptr;
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  data.fused_runner = nullptr;
  data.fused_cross_attention_kernel = fused_cross_attention_kernel;
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = false;
  data.cumulated_sequence_length_q_cache = &(this->cumulated_sequence_length_q_cache_); // TODO(aciddelgado): no need right?
  data.cumulated_sequence_length_kv_cache = &(this->cumulated_sequence_length_kv_cache_);

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
