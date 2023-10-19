// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MultiHeadAttention,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MultiHeadAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const OpKernelInfo& info)
    : CudaKernel(info),
      fused_fp16_cross_attention_kernel_(nullptr),
      cumulated_sequence_length_q_cache_(),
      cumulated_sequence_length_kv_cache_() {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

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

#if USE_FLASH_ATTENTION || USE_MEMORY_EFFICIENT_ATTENTION
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
  // Allocate buffers
  size_t softmax_lse_accum_bytes = 0;
  size_t out_accum_bytes = 0;
  if (use_flash_attention) {
    // split kv buffers
    parameters.num_splits = onnxruntime::flash::num_splits_heuristic(
        parameters.batch_size, parameters.sequence_length, parameters.kv_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount, 128);
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
  }
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());  // nullptr
  auto out_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());          // nullptr
#endif

  bool use_fused_cross_attention = !use_flash_attention &&
                                   !disable_fused_cross_attention_ &&
                                   nullptr == key_padding_mask &&
                                   nullptr == relative_position_bias &&
                                   (nullptr == past_key && nullptr == past_value && !parameters.pass_past_in_kv) &&
                                   key != nullptr &&
                                   (value != nullptr || bias == nullptr) &&  // TODO: new kernel for adding bias to packed KV
                                   parameters.hidden_size == parameters.v_hidden_size &&
                                   has_fused_cross_attention_kernel(sm, parameters.head_size,
                                                                    parameters.kv_sequence_length);
  if (use_fused_cross_attention) {
    if (fused_fp16_cross_attention_kernel_ == nullptr) {
      fused_fp16_cross_attention_kernel_ = get_fused_cross_attention_kernels(sm);
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    // The kernel has no limit on sequence length, and this checks whether the kernel has been loaded.
    if (fused_fp16_cross_attention_kernel_->isValid(sequence_length)) {
      fused_cross_attention_kernel = fused_fp16_cross_attention_kernel_;
    }
  }

  bool use_fused_runner = !use_flash_attention &&
                          !disable_fused_self_attention_ &&
                          fused_cross_attention_kernel == nullptr &&
                          nullptr == relative_position_bias &&
                          (value != nullptr || key == nullptr) &&
                          (nullptr == past_key && nullptr == past_value && !parameters.pass_past_in_kv) &&
                          (nullptr == key_padding_mask || is_mask_1d_seq_len) &&
                          parameters.hidden_size == parameters.v_hidden_size &&
                          parameters.sequence_length == parameters.kv_sequence_length &&
                          FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                             enable_trt_flash_attention_, false);
  if (use_fused_runner) {
    // Here we assume that num_heads and head_size does not change for a MultiHeadAttention node.
    if (nullptr == fused_fp16_runner_.get()) {
      constexpr bool is_unidirectional = false;
      std::call_once(fused_fp16_runner_created_, [&]() {
        fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm, is_unidirectional,
                                                          enable_trt_flash_attention_, parameters.scale);
      });
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
    if (fused_fp16_runner_->isValid(S)) {
      fused_runner = fused_fp16_runner_.get();
    }
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  bool is_long_sequence = sizeof(T) == 2 ||  // sequence length threshold is 0 for FP16
                          parameters.sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32 ||
                          parameters.kv_sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32;

  bool is_good_for_rpb = relative_position_bias != nullptr && parameters.sequence_length % (4 * sizeof(T)) == 0;

  bool use_memory_efficient_attention = !use_flash_attention &&
                                        fused_runner == nullptr &&
                                        fused_cross_attention_kernel == nullptr &&
                                        !disable_memory_efficient_attention_ &&
                                        (parameters.head_size & 7) == 0 &&
                                        (parameters.v_head_size & 7) == 0 &&
                                        is_long_sequence &&
                                        !past_no_bias &&
                                        (relative_position_bias == nullptr || is_good_for_rpb) &&
                                        (nullptr == key_padding_mask || parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) &&
                                        has_memory_efficient_attention(sm, sizeof(T) == 2);
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  // When packed kv or packed qkv is used, there is no needed for add bias transpose thus no qkv workspace.
  // TODO(tianleiwu): flash attention or memory efficient attention might not need qkv workspace sometime.
  bool no_qkv_workspace = nullptr == value &&
                          (use_fused_cross_attention || (nullptr != fused_runner && nullptr == key)) &&
                          nullptr == key_padding_mask &&
                          nullptr == bias;

  size_t workspace_bytes;
  constexpr size_t element_size = sizeof(T);
  if (no_qkv_workspace) {
    workspace_bytes = (parameters.batch_size > kCumulatedSequenceLengthCacheMaxBatchSize) ? 2 * GetSequenceOffsetSize(parameters.batch_size, true) : 0;
  } else {
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
  }

  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  const size_t past_k_bytes = element_size * parameters.batch_size * parameters.kv_sequence_length * parameters.num_heads * parameters.head_size;
  const size_t past_v_bytes = element_size * parameters.batch_size * parameters.kv_sequence_length * parameters.num_heads * parameters.v_head_size;
  const bool use_temp_k_v_workspace = parameters.pass_past_in_kv || use_memory_efficient_attention || use_flash_attention;
  auto temp_k_work_space = use_temp_k_v_workspace ? GetScratchBuffer<void>(past_k_bytes, context->GetComputeStream()) : nullptr;
  auto temp_v_work_space = use_temp_k_v_workspace ? GetScratchBuffer<void>(past_v_bytes, context->GetComputeStream()) : nullptr;

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.bias = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = (nullptr == key || parameters.pass_past_in_kv) ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (nullptr == value || parameters.pass_past_in_kv) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = (nullptr == key_padding_mask) ? nullptr : key_padding_mask->Data<int>();
  data.mask_index_dims = (nullptr == key_padding_mask) ? gsl::span<const int64_t>() : key_padding_mask->Shape().GetDims();
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
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = fused_cross_attention_kernel;
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  data.cumulated_sequence_length_q_cache = &(this->cumulated_sequence_length_q_cache_);
  data.cumulated_sequence_length_kv_cache = &(this->cumulated_sequence_length_kv_cache_);
  if (softmax_lse_accum_buffer != nullptr) {
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
  }
  if (out_accum_buffer != nullptr) {
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
