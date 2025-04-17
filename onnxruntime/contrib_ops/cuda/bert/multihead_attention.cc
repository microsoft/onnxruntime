// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/fastertransformer_decoder_attention/decoder_masked_multihead_attention_impl.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/lean_attention/lean_api.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, QK)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      MultiHeadAttention,                                          \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##QK,                                                    \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("QK", DataTypeImpl::GetTensorType<QK>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 8),                 \
      MultiHeadAttention<T, QK>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(float, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)

template <typename T, typename QK>
MultiHeadAttention<T, QK>::MultiHeadAttention(const OpKernelInfo& info)
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

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_fused_self_attention_ = sizeof(T) != 2 || !kernel_options_->UseTrtFusedAttention();
  enable_trt_flash_attention_ = sizeof(T) == 2 && kernel_options_->UseTrtFlashAttention();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

#if USE_LEAN_ATTENTION
  enable_lean_attention_ = sizeof(T) == 2 && kernel_options_->UseLeanAttention();
#endif

  disable_memory_efficient_attention_ = !kernel_options_->UseEfficientAttention();

  disable_fused_cross_attention_ = sizeof(T) != 2 || !kernel_options_->UseTrtCrossAttention();

  enable_cudnn_flash_attention_ = sizeof(T) == 2 && kernel_options_->UseCudnnFlashAttention();

  disable_decoder_attention_ = !kernel_options_->UseDecoderAttention();

  // Allocate cache buffers
  constexpr size_t cache_bytes = sizeof(int32_t) * (static_cast<size_t>(kCumulatedSequenceLengthCacheMaxBatchSize) + 1);
  cumulated_sequence_length_q_cache_.buffer = GetTransientScratchBuffer<void>(cache_bytes);
  cumulated_sequence_length_q_cache_.max_batch_size = kCumulatedSequenceLengthCacheMaxBatchSize;
  cumulated_sequence_length_kv_cache_.buffer = GetTransientScratchBuffer<void>(cache_bytes);
  cumulated_sequence_length_kv_cache_.max_batch_size = kCumulatedSequenceLengthCacheMaxBatchSize;
}

template <typename T, typename QK>
Status MultiHeadAttention<T, QK>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* attention_bias = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);
  const Tensor* past_sequence_length = context->Input<Tensor>(8);
  const Tensor* cache_indirection = context->Input<Tensor>(9);

  bool past_present_share_buffer = past_key != nullptr && past_sequence_length != nullptr;
  if (past_key != nullptr && past_sequence_length != nullptr && cache_indirection != nullptr) {
    ORT_ENFORCE(past_present_share_buffer);
  }

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
                                                                      cache_indirection,
                                                                      past_sequence_length,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      is_unidirectional_,
                                                                      past_present_share_buffer,
                                                                      kMultiHeadAttention,
                                                                      device_prop.maxThreadsPerBlock));
  DUMP_STRING_INIT();
  DUMP_STRING("Batch size = ", parameters.batch_size);
  DUMP_STRING("Sequence length = ", parameters.sequence_length);
  DUMP_STRING("Past sequence length = ", parameters.past_sequence_length);
  DUMP_STRING("KV sequence length = ", parameters.kv_sequence_length);
  DUMP_STRING("Total sequence length = ", parameters.total_sequence_length);
  DUMP_STRING("Max sequence length = ", parameters.max_sequence_length);
  DUMP_STRING("Hidden size = ", parameters.hidden_size);
  DUMP_STRING("Head size = ", parameters.head_size);
  DUMP_STRING("Num heads = ", parameters.num_heads);
  DUMP_STRING("Buffer sharing = ", (parameters.past_present_share_buffer == true));
  DUMP_STRING("QKV format = ", parameters.qkv_format);

  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      parameters.batch_size, parameters.num_heads, parameters.max_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  std::vector<int64_t> output_qk_dims{
      parameters.batch_size, parameters.num_heads, parameters.sequence_length, parameters.total_sequence_length};
  TensorShape output_qk_shape(output_qk_dims);
  Tensor* output_qk = context->Output(3, output_qk_shape);

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

  AttentionKernelType kernel_type = AttentionKernelType::AttentionKernel_Default;
  cudaStream_t stream = Stream(context);

  bool use_decoder_masked_multihead_attention = false;
  if (cache_indirection != nullptr) {
    bool use_dmmha_self_attention = parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH &&
                                    parameters.past_present_share_buffer &&
                                    parameters.past_sequence_length > 0;
    bool use_dmmha_cross_attention = parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH &&
                                     past_key == nullptr && past_value == nullptr && nullptr != past_sequence_length &&
                                     parameters.past_sequence_length != *((*past_sequence_length).template Data<int32_t>());
    use_decoder_masked_multihead_attention = !disable_decoder_attention_ &&
                                             (std::is_same<T, float>::value || std::is_same<T, MLFloat16>::value) &&
                                             (use_dmmha_self_attention || use_dmmha_cross_attention) &&
                                             parameters.sequence_length == 1 &&
                                             parameters.head_size == parameters.v_head_size &&
                                             (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING || parameters.mask_type == AttentionMaskType::MASK_NONE) &&
                                             nullptr != past_sequence_length && nullptr != cache_indirection &&
                                             has_decoder_masked_multihead_attention(sm, parameters.head_size);
  }
  DUMP_STRING("Use DMMHA = ", (use_decoder_masked_multihead_attention == true));
  if (use_decoder_masked_multihead_attention) {
    // Kernel only works for token generation with beam search
    kernel_type = AttentionKernelType::AttentionKernel_DecoderAttention;

    // No production use-case will incur this copy cost as the implementation of
    // DecoderMaskedMultiHeadAttention is written in such a way that the past and present buffers
    // must be shared to have parity in the outputs.
    // This is just to circumvent the OpTester's limitation of not being able to bind a specific
    // buffer to inputs/outputs.
    auto* past_key_data = (past_key == nullptr) ? nullptr : past_key->Data<T>();
    auto* past_value_data = (past_value == nullptr) ? nullptr : past_value->Data<T>();
    auto* present_key_data = (present_key == nullptr) ? nullptr : present_key->MutableData<T>();
    auto* present_value_data = (present_value == nullptr) ? nullptr : present_value->MutableData<T>();

    if (present_key_data != past_key_data) {
      DUMP_STRING("Copying past_key to present_key for OpTester");
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_key_data, past_key_data, past_key->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, stream));
    }
    if (present_value_data != past_value_data) {
      DUMP_STRING("Copying past_value to present_value for OpTester");
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_value_data, past_value_data, past_value->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, stream));
    }
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;

#if USE_LEAN_ATTENTION || USE_FLASH_ATTENTION
  size_t softmax_lse_bytes = 0;
  size_t softmax_lse_accum_bytes = 0;
  size_t out_accum_bytes = 0;
#endif

#if USE_LEAN_ATTENTION
  // Lean attention only supports token-generation phase with sequence_length == 1.
  bool use_lean_attention = enable_lean_attention_ &&
                            parameters.sequence_length == 1 &&
                            parameters.past_sequence_length > 0 &&
                            nullptr == attention_bias &&
                            nullptr == key_padding_mask &&
                            parameters.head_size == parameters.v_head_size &&
                            onnxruntime::lean::is_supported(device_prop,
                                                            parameters.head_size,
                                                            parameters.num_heads,
                                                            parameters.num_heads);

  size_t sync_flag_bytes = 0;
  DUMP_STRING("Use lean attn = ", (use_lean_attention == true));
  if (use_lean_attention) {
    softmax_lse_bytes = onnxruntime::lean::get_softmax_lse_size(parameters.sequence_length,
                                                                parameters.batch_size,
                                                                parameters.num_heads);

    auto [num_splits, slse_accum_bytes, o_accum_bytes, sflag_bytes, griddimz, max_tiles_tb, hload_tbs, tiles_per_head] = onnxruntime::lean::get_num_splits_and_buffer_sizes(
        parameters.batch_size,
        parameters.sequence_length,
        parameters.total_sequence_length,
        parameters.num_heads,  // q heads
        parameters.num_heads,  // kv heads
        parameters.head_size,
        device_prop.multiProcessorCount,
        parameters.is_unidirectional);

    data.num_splits = static_cast<int>(num_splits);
    data.grid_dim_z = static_cast<int>(griddimz);
    data.max_tiles_per_tb = static_cast<int>(max_tiles_tb);
    data.high_load_tbs = static_cast<int>(hload_tbs);
    data.tiles_per_head = static_cast<int>(tiles_per_head);
    softmax_lse_accum_bytes = slse_accum_bytes;
    out_accum_bytes = o_accum_bytes;
    sync_flag_bytes = sflag_bytes;
    kernel_type = AttentionKernelType::AttentionKernel_LeanAttention;
  }

  auto lean_sync_flag_buffer = GetScratchBuffer<void>(sync_flag_bytes, context->GetComputeStream());
  data.lean_sync_flag = reinterpret_cast<int*>(lean_sync_flag_buffer.get());
#else
  constexpr bool use_lean_attention = false;
#endif

#if USE_FLASH_ATTENTION
  bool use_flash_attention = kernel_type == AttentionKernelType::AttentionKernel_Default &&
                             !disable_flash_attention_ &&
                             nullptr == attention_bias &&
                             nullptr == key_padding_mask &&
                             nullptr == past_sequence_length &&
                             nullptr == cache_indirection &&
                             nullptr == output_qk &&
                             parameters.head_size == parameters.v_head_size &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.num_heads);
  // When input is packed QKV format, TensorRT kernel might be faster than flash attention when sequence length <= 512.
  DUMP_STRING("Use flash attn = ", (use_flash_attention == true));
  if (use_flash_attention && parameters.qkv_format == AttentionQkvFormat::QKV_BS3NH &&
      parameters.sequence_length < kernel_options_->MinSeqLenForFlashAttentionPackedQkv()) {
    use_flash_attention = false;
  }

  // Allocate buffers
  if (use_flash_attention) {
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length,
                                                                 parameters.batch_size,
                                                                 parameters.num_heads);

    using namespace std;
    auto [num_splits, slse_accum_bytes, o_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.total_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount);
    data.num_splits = static_cast<int>(num_splits);
    softmax_lse_accum_bytes = slse_accum_bytes;
    out_accum_bytes = o_accum_bytes;
    kernel_type = AttentionKernelType::AttentionKernel_FlashAttention;
  }
#else
  constexpr bool use_flash_attention = false;
#endif

#if USE_LEAN_ATTENTION || USE_FLASH_ATTENTION
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());
  if (use_flash_attention || use_lean_attention) {
    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }
#endif

  bool is_mask_none_or_1d_k_len = parameters.mask_type == AttentionMaskType::MASK_NONE ||
                                  parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
  bool use_cudnn_sdpa = kernel_type == AttentionKernelType::AttentionKernel_Default &&
                        enable_cudnn_flash_attention_ &&
                        is_mask_none_or_1d_k_len &&
                        onnxruntime::cudnn_sdpa::is_supported(device_prop,
                                                              parameters.num_heads,              // num_heads_q
                                                              parameters.num_heads,              // num_heads_kv
                                                              parameters.head_size,              // head_size_qk
                                                              parameters.v_head_size,            // head_size_v
                                                              parameters.sequence_length,        // seq_len_q
                                                              parameters.total_sequence_length,  // seq_len_kv
                                                              is_unidirectional_);
  DUMP_STRING("Use cuDNN SDPA = ", (use_cudnn_sdpa == true));
  if (use_cudnn_sdpa) {
    kernel_type = AttentionKernelType::AttentionKernel_CudnnFlashAttention;
  }

  bool use_fused_cross_attention =
      kernel_type == AttentionKernelType::AttentionKernel_Default &&
      !disable_fused_cross_attention_ &&
      !is_unidirectional_ &&
      nullptr == key_padding_mask &&
      nullptr == attention_bias &&
      nullptr == past_key &&
      nullptr == past_sequence_length &&
      nullptr == cache_indirection &&
      nullptr == present_key &&
      nullptr == output_qk &&
      (parameters.qkv_format == Q_K_V_BSNH || (parameters.qkv_format == Q_KV_BSNH_BSN2H && bias == nullptr)) &&
      parameters.hidden_size == parameters.v_hidden_size &&
      has_fused_cross_attention_kernel(sm, parameters.head_size, parameters.kv_sequence_length);

  DUMP_STRING("Use fused cross attn = ", (use_fused_cross_attention == true));
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
      kernel_type = AttentionKernelType::AttentionKernel_TrtFusedCrossAttention;
    }
  }

  bool use_fused_runner =
      kernel_type == AttentionKernelType::AttentionKernel_Default &&
      !disable_fused_self_attention_ &&
      !is_unidirectional_ &&
      nullptr == attention_bias &&
      (parameters.qkv_format == Q_K_V_BSNH || parameters.qkv_format == QKV_BSN3H) &&
      nullptr == past_key && nullptr == past_sequence_length && nullptr == cache_indirection &&
      nullptr == present_key && nullptr == output_qk &&
      is_mask_none_or_1d_k_len &&
      parameters.hidden_size == parameters.v_hidden_size &&
      parameters.sequence_length == parameters.kv_sequence_length &&  // self attention only for fused runner
      FusedMHARunnerFP16v2::IsSupported(sm, parameters.head_size, sequence_length,
                                        enable_trt_flash_attention_, is_unidirectional_);

  DUMP_STRING("Use fused runner = ", (use_fused_runner == true));
  if (use_fused_runner) {
    // Here we assume that num_heads and head_size does not change for a MultiHeadAttention node.
    if (nullptr == fused_fp16_runner_.get()) {
      std::call_once(fused_fp16_runner_created_, [&]() {
        fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                          enable_trt_flash_attention_, parameters.scale);
      });
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    const int normalized_seq_len = fused_fp16_runner_->NormalizeSequenceLength(sequence_length);
    if (fused_fp16_runner_->IsValid(normalized_seq_len)) {
      fused_runner = fused_fp16_runner_.get();
      // could also be AttentionKernel_TrtFlashAttention, but we don't classify it here.
      kernel_type = AttentionKernelType::AttentionKernel_TrtFusedAttention;
    }
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  int length_threshold = this->kernel_options_->MinSeqLenForEfficientAttentionFp32();
  bool is_long_sequence = std::is_same<T, MLFloat16>::value ||  // sequence length threshold is 0 for FP16
                          parameters.sequence_length >= length_threshold ||
                          parameters.kv_sequence_length >= length_threshold;

  bool use_memory_efficient_attention =
      kernel_type == AttentionKernelType::AttentionKernel_Default &&
      !disable_memory_efficient_attention_ &&
      is_long_sequence &&
      // Check whether the attention bias alignment is good for memory efficient attention.
      (attention_bias == nullptr || parameters.sequence_length % (4 * sizeof(T)) == 0) &&
      (nullptr == key_padding_mask || parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) &&
      nullptr == past_sequence_length && nullptr == cache_indirection && nullptr == output_qk &&
      has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value,
                                     parameters.head_size, parameters.v_head_size);
  DUMP_STRING("Use memory efficient attention = ", (use_memory_efficient_attention == true));
  if (use_memory_efficient_attention) {
    kernel_type = AttentionKernelType::AttentionKernel_CutlassMemoryEfficientAttention;
  }
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  if (kernel_type == AttentionKernelType::AttentionKernel_Default) {
    kernel_type = AttentionKernelType::AttentionKernel_Unfused;
  }

  typedef typename ToCudaType<QK>::MappedType CudaQK;
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
  if (nullptr != cache_indirection) {
    data.cache_indirection = reinterpret_cast<const int32_t*>(cache_indirection->Data<int32_t>());
  }
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  if (nullptr != output_qk) {
    data.output_qk = reinterpret_cast<CudaQK*>(output_qk->MutableData<QK>());
  }
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = fused_cross_attention_kernel;
  data.use_flash_attention = use_flash_attention;
  data.use_lean_attention = use_lean_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  data.use_decoder_masked_multihead_attention = use_decoder_masked_multihead_attention;
  data.kernel_type = kernel_type;
  data.allocator = Info().GetAllocator(OrtMemType::OrtMemTypeDefault);

  // Cache of cumulated sequence length that could help when sequence length does not change (for example, image model).
  // The cache will be initialized only once, and become readonly after that.
  if ((data.fused_cross_attention_kernel != nullptr || data.fused_runner != nullptr) && data.mask_index == nullptr) {
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
                                                     use_lean_attention,
                                                     use_fused_cross_attention,
                                                     use_memory_efficient_attention,
                                                     use_cudnn_sdpa,
                                                     no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;

  data.allow_debug_info = kernel_options_->AllowDebugInfo();

  // For past-present buffer sharing.
  if (parameters.past_present_share_buffer) {
    std::vector<int64_t> seqlens_k(parameters.batch_size, parameters.total_sequence_length - 1);
    size_t seqlens_k_bytes = 0;
    seqlens_k_bytes = sizeof(int) * parameters.batch_size;
    auto seqlens_k_buffer = GetScratchBuffer<void>(seqlens_k_bytes, context->GetComputeStream());
    if (seqlens_k_buffer != nullptr) {
      data.seqlens_k_total = reinterpret_cast<int*>(seqlens_k_buffer.get());
      CUDA_RETURN_IF_ERROR(cudaMemcpy(data.seqlens_k_total, seqlens_k.data(), seqlens_k_bytes, cudaMemcpyHostToDevice));
    }
  }

  if (data.allow_debug_info) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_lean_attention = use_lean_attention;
    debug_info.use_cudnn_flash_attention = use_cudnn_sdpa;
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
  cudnnHandle_t cudnn = GetCudnnHandle(context);
  DUMP_STRING("Run QkvToContext from MHA CUDA");
  return QkvToContext<CudaT, CudaQK>(
      device_prop, cublas, cudnn, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
