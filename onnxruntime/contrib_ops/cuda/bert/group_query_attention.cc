// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
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

REGISTER_KERNEL_TYPED(float)
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
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads_k).IsOK() && num_heads_k > 0);
  num_heads_ = static_cast<int>(num_heads);
  num_heads_k_ = static_cast<int>(num_heads_k);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  disable_fused_self_attention_ = sizeof(T) != 2 ||
                                  ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedSelfAttention, false);

  enable_trt_flash_attention_ = sizeof(T) == 2 &&
                                !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);

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

#if USE_MEMORY_EFFICIENT_ATTENTION
  disable_memory_efficient_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif

  disable_fused_cross_attention_ = sizeof(T) != 2 ||
                                   ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedCrossAttention, false);

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
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  // TODO: new function or mod this one?
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      key_padding_mask,
                                                                      relative_position_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      nullptr,  // past_seq_len
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      false,  // past_present_share_buffer
                                                                      false,  // dmmha_packing
                                                                      device_prop.maxThreadsPerBlock));
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
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

  bool is_mask_1d_seq_len = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  const bool pass_key_value_as_past = (parameters.pass_past_in_kv && nullptr != key && nullptr != value);

#if USE_FLASH_ATTENTION
  // Exclude this case since PrepareQkv will convert the format to BNSH.
  bool past_no_bias = (pass_key_value_as_past || past_key != nullptr || present_key != nullptr) && bias == nullptr;
#endif

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             !past_no_bias &&
                             nullptr == relative_position_bias &&
                             nullptr == key_padding_mask &&
                             parameters.head_size == parameters.v_head_size &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.num_heads);
  // When input is packed QKV format, TensorRT kernel might be faster than flash attention when sequence length <= 512.
  if (use_flash_attention && key == nullptr && value == nullptr &&
      parameters.sequence_length < min_seq_len_for_flash_attention_packed_qkv_) {
    use_flash_attention = false;
  }
#else
  constexpr bool use_flash_attention = false;
#endif

  size_t workspace_bytes;
  constexpr size_t element_size = sizeof(T);
  // TODO: change this for different kv num head
  workspace_bytes = GetAttentionWorkspaceSize(element_size,
                                              parameters.batch_size,
                                              parameters.num_heads,
                                              parameters.head_size,
                                              parameters.v_head_size,
                                              parameters.sequence_length,
                                              parameters.kv_sequence_length,
                                              parameters.total_sequence_length,
                                              fused_runner,
                                              use_flash_attention,
                                              use_fused_cross_attention,
                                              use_memory_efficient_attention);
  // }

  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  // TODO: past k and v?
  const size_t past_k_bytes = element_size * parameters.batch_size * parameters.kv_sequence_length * parameters.num_heads_k * parameters.head_size;
  const size_t past_v_bytes = element_size * parameters.batch_size * parameters.kv_sequence_length * parameters.num_heads_k * parameters.v_head_size;
  const bool use_temp_k_v_workspace = parameters.pass_past_in_kv || use_memory_efficient_attention || use_flash_attention;
  auto temp_k_work_space = use_temp_k_v_workspace ? GetScratchBuffer<void>(past_k_bytes, context->GetComputeStream()) : nullptr;
  auto temp_v_work_space = use_temp_k_v_workspace ? GetScratchBuffer<void>(past_v_bytes, context->GetComputeStream()) : nullptr;

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = nullptr;
  data.bias = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = (nullptr == key || parameters.pass_past_in_kv) ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (nullptr == value || parameters.pass_past_in_kv) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = (nullptr == key_padding_mask) ? nullptr : key_padding_mask->Data<int>();
  data.mask_index_dims = (nullptr == key_padding_mask) ? gsl::span<const int64_t>() : key_padding_mask->Shape().GetDims();
  data.past = nullptr;
  data.past_key = pass_key_value_as_past  ? reinterpret_cast<const CudaT*>(key->Data<T>())
                  : (nullptr == past_key) ? nullptr
                                          : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = pass_key_value_as_past    ? reinterpret_cast<const CudaT*>(value->Data<T>())
                    : (nullptr == past_value) ? nullptr
                                              : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.relative_position_bias = (nullptr == relative_position_bias) ? nullptr : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.temp_k_workspace = use_temp_k_v_workspace ? reinterpret_cast<CudaT*>(temp_k_work_space.get()) : nullptr;
  data.temp_v_workspace = use_temp_k_v_workspace ? reinterpret_cast<CudaT*>(temp_v_work_space.get()) : nullptr;
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = nullptr;
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  data.use_flash_attention = use_flash_attention;
  data.cumulated_sequence_length_q_cache = &(this->cumulated_sequence_length_q_cache_);
  data.cumulated_sequence_length_kv_cache = &(this->cumulated_sequence_length_kv_cache_);

  cublasHandle_t cublas = GetCublasHandle(context);

  // TODO: QkvToContext in new group impl file or adapt existing?
  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
