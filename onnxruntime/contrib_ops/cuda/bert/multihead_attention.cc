// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"

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
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
  ORT_ENFORCE(!is_unidirectional_,
              "MHA support CUDA kernel does not Unidirectional. Consider using Attention or GQA instead.");

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_fused_self_attention_ = sizeof(T) != 2 || !kernel_options_->UseTrtFusedAttention();
  enable_trt_flash_attention_ = sizeof(T) == 2 && kernel_options_->UseTrtFlashAttention();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

  disable_memory_efficient_attention_ = !kernel_options_->UseEfficientAttention();

  disable_fused_cross_attention_ = sizeof(T) != 2 || !kernel_options_->UseTrtCrossAttention();

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
  const Tensor* attention_bias = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  parameters.use_tf32 = UseTF32();

  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      key_padding_mask,
                                                                      attention_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      nullptr,  // past_seq_len
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      is_unidirectional_,
                                                                      false,  // past_present_share_buffer
                                                                      kMultiHeadAttention,
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

  int num_past = static_cast<int>(past_key != nullptr) + static_cast<int>(past_value != nullptr);
  int num_present = static_cast<int>(present_key != nullptr) + static_cast<int>(present_value != nullptr);
  if (num_past == 0 && num_present == 0) {
    // It is valid case without past state.
  } else if ((num_past == 2 && num_present == 2) || (num_past == 0 && num_present == 2)) {
    if (parameters.qkv_format == AttentionQkvFormat::QKV_BSN3H) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Inputs 'past_key', 'past_value', 'present_key' and 'present_value' shall be empty for packed QKV format");
    }

    if (parameters.qkv_format == AttentionQkvFormat::Q_KV_BSNH_BSN2H) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Inputs 'past_key', 'past_value', 'present_key' and 'present_value' shall be empty for packed KV format");
    }

    if (parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Inputs 'past_key', 'past_value', 'present_key' and 'present_value' shall be empty for cross attention");
    }
  } else {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Inputs 'past_key', 'past_value', 'present_key' and 'present_value' shall be all provided, "
        "or all empty, or only present_key and present_value are provided");
  }

  MHARunner* fused_runner = nullptr;
  const FusedMultiHeadCrossAttentionKernel* fused_cross_attention_kernel = nullptr;

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             nullptr == attention_bias &&
                             nullptr == key_padding_mask &&
                             parameters.head_size == parameters.v_head_size &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.num_heads);
  // When input is packed QKV format, TensorRT kernel might be faster than flash attention when sequence length <= 512.
  if (use_flash_attention && parameters.qkv_format == AttentionQkvFormat::QKV_BS3NH &&
      parameters.sequence_length < kernel_options_->MinSeqLenForFlashAttentionPackedQkv()) {
    use_flash_attention = false;
  }
  // Allocate buffers
  size_t softmax_lse_accum_bytes = 0;
  size_t out_accum_bytes = 0;
  if (use_flash_attention) {
    using namespace std;
    auto [num_splits, slse_accum_bytes, o_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.kv_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount);
    parameters.num_splits = static_cast<int>(num_splits);
    softmax_lse_accum_bytes = slse_accum_bytes;
    out_accum_bytes = o_accum_bytes;
  }
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());  // nullptr
  auto out_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());          // nullptr
#endif

  bool is_mask_none_or_1d_k_len = parameters.mask_type == AttentionMaskType::MASK_NONE ||
                                  parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
  bool use_fused_cross_attention =
      !use_flash_attention &&
      !disable_fused_cross_attention_ &&
      nullptr == key_padding_mask &&
      nullptr == attention_bias &&
      nullptr == past_key && nullptr == present_key &&
      (parameters.qkv_format == Q_K_V_BSNH || (parameters.qkv_format == Q_KV_BSNH_BSN2H && bias == nullptr)) &&
      parameters.hidden_size == parameters.v_hidden_size &&
      has_fused_cross_attention_kernel(sm, parameters.head_size,
                                       parameters.kv_sequence_length);
  if (use_fused_cross_attention) {
    if (fused_fp16_cross_attention_kernel_ == nullptr) {
      std::call_once(fused_cross_init_once_flag_, [&]() {
        fused_fp16_cross_attention_kernel_ = get_fused_cross_attention_kernels(sm);
      });
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    // The kernel has no limit on sequence length, and this checks whether the kernel has been loaded.
    if (fused_fp16_cross_attention_kernel_->isValid(sequence_length)) {
      fused_cross_attention_kernel = fused_fp16_cross_attention_kernel_;
    }
  }

  bool use_fused_runner =
      !use_flash_attention &&
      !disable_fused_self_attention_ &&
      fused_cross_attention_kernel == nullptr &&
      nullptr == attention_bias &&
      (parameters.qkv_format == Q_K_V_BSNH || parameters.qkv_format == QKV_BSN3H) &&
      nullptr == past_key && nullptr == present_key &&
      is_mask_none_or_1d_k_len &&
      parameters.hidden_size == parameters.v_hidden_size &&
      parameters.sequence_length == parameters.kv_sequence_length &&  // self attention only for fused runner
      FusedMHARunnerFP16v2::IsSupported(sm, parameters.head_size, sequence_length,
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
    const int normalized_seq_len = fused_fp16_runner_->NormalizeSequenceLength(sequence_length);
    if (fused_fp16_runner_->IsValid(normalized_seq_len)) {
      fused_runner = fused_fp16_runner_.get();
    }
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  int length_threshold = this->kernel_options_->MinSeqLenForEfficientAttentionFp32();
  bool is_long_sequence = std::is_same<T, MLFloat16>::value ||  // sequence length threshold is 0 for FP16
                          parameters.sequence_length >= length_threshold ||
                          parameters.kv_sequence_length >= length_threshold;

  bool use_memory_efficient_attention =
      !use_flash_attention &&
      fused_runner == nullptr &&
      fused_cross_attention_kernel == nullptr &&
      !disable_memory_efficient_attention_ &&
      is_long_sequence &&
      // Check whether the attention bias alignment is good for memory efficient attention.
      (attention_bias == nullptr || parameters.sequence_length % (4 * sizeof(T)) == 0) &&
      (nullptr == key_padding_mask || parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) &&
      has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value,
                                     parameters.head_size, parameters.v_head_size);
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.bias = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = (nullptr == key) ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (nullptr == value) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = (nullptr == key_padding_mask) ? nullptr : key_padding_mask->Data<int>();
  data.mask_index_dims = (nullptr == key_padding_mask) ? gsl::span<const int64_t>() : key_padding_mask->Shape().GetDims();
  data.past_key = (nullptr == past_key) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (nullptr == past_value) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  if (nullptr != attention_bias) {
    data.attention_bias = reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
  }
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = fused_cross_attention_kernel;
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;

  // Cache of cumulated sequence length that could help when sequence length does not change (for example, image model).
  // The cache will be initialized only once, and become readonly after that.
  if ((data.fused_cross_attention_kernel != nullptr || data.fused_runner != nullptr) && data.mask_index == nullptr) {
    cudaStream_t stream = Stream(context);
    data.cumulated_sequence_length_q_cache = this->cumulated_sequence_length_q_cache_.TryGet(
        parameters.batch_size, parameters.sequence_length, stream);

    if (data.fused_cross_attention_kernel != nullptr) {
      data.cumulated_sequence_length_kv_cache = this->cumulated_sequence_length_kv_cache_.TryGet(
          parameters.batch_size, parameters.kv_sequence_length, stream);
    }
  }

  const bool no_qkv_workspace = NoQkvWorkspace(parameters, data);
  size_t workspace_bytes = GetAttentionWorkspaceSize(sizeof(T),
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
                                                     use_memory_efficient_attention,
                                                     no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;

  data.allow_debug_info = kernel_options_->AllowDebugInfo();
  if (softmax_lse_accum_buffer != nullptr) {
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
  }
  if (out_accum_buffer != nullptr) {
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }

  if (data.allow_debug_info) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_trt_cross_attention = fused_cross_attention_kernel != nullptr;
    debug_info.use_efficient_attention = use_memory_efficient_attention;
    if (fused_fp16_runner_ != nullptr) {
      debug_info.SetTrtFusedKernel(is_unidirectional_, enable_trt_flash_attention_, sequence_length);
    }
    debug_info.Print("MultiHeadAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);

    data.PrintDebugInfo();
  }

  cublasHandle_t cublas = GetCublasHandle(context);
  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
