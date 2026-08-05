// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kPastSequenceLengthInputIndex = 6;
constexpr int kPastInputIndex = 4;
constexpr int kPresentOutputIndex = 1;

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Attention,                                                               \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, false) {
  kernel_options_ = this->GetAttentionKernelOptions();

  constexpr bool kIsFp16 = std::is_same<T, MLFloat16>::value;
  constexpr bool kIsBf16 = std::is_same<T, BFloat16>::value;
  constexpr bool kIs16bit = kIsFp16 || kIsBf16;

  // We only support FP16 for TRT fused/flash attention.
  disable_fused_self_attention_ = !kIsFp16 || !kernel_options_->UseTrtFusedAttention();
  enable_trt_flash_attention_ = kIsFp16 && kernel_options_->UseTrtFlashAttention();

  disable_memory_efficient_attention_ = kIsBf16 || !kernel_options_->UseEfficientAttention();

  disable_flash_attention_ = !kIs16bit || !kernel_options_->UseFlashAttention();

  enable_cudnn_flash_attention_ = kIs16bit && kernel_options_->UseCudnnFlashAttention();
  auto_enable_cudnn_flash_attention_ = kIs16bit && kernel_options_->AllowCudnnFlashAttentionAuto();
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  auto ort_stream = GetOrtStream(context);

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(kPastInputIndex);
  const Tensor* attention_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  parameters.use_tf32 = UseTF32();

  // Use the second dimension from weight for bias to get q_hidden_size when bias is nullptr
  std::vector<int64_t> bias_dims{weights->Shape().GetDims()[1]};
  const TensorShape bias_shape{bias_dims};
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias != nullptr ? bias->Shape() : bias_shape,
                                  mask_index,
                                  past,
                                  attention_bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock,
                                  past_seq_len));
  assert(parameters.sequence_length == parameters.kv_sequence_length);  // self attention

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      2, parameters.batch_size, parameters.num_heads,
      past_present_share_buffer_ ? parameters.max_sequence_length : parameters.total_sequence_length,
      parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(kPresentOutputIndex, present_shape);

  MHARunner* fused_runner = nullptr;

  // Check whether we can use fused kernel
  const int sm = device_prop.major * 10 + device_prop.minor;
  const bool is_mask_1d_seq_len = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;

  const bool cudnn_sdpa_enabled = enable_cudnn_flash_attention_ ||
                                  (auto_enable_cudnn_flash_attention_ && sm >= 90);
  // attention_bias is safe here because the no-cache self-attention path has
  // sequence_length == total_sequence_length, so cuDNN uses top-left causal alignment.
  const bool cudnn_sdpa_supported = cudnn_sdpa_enabled &&
                                    (parameters.mask_type == AttentionMaskType::MASK_NONE ||
                                     is_mask_1d_seq_len) &&
                                    nullptr == past &&
                                    nullptr == present &&
                                    onnxruntime::cudnn_sdpa::is_stable() &&
                                    onnxruntime::cudnn_sdpa::is_supported(device_prop,
                                                                          parameters.num_heads,
                                                                          parameters.num_heads,
                                                                          parameters.head_size,
                                                                          parameters.v_head_size,
                                                                          parameters.sequence_length,
                                                                          parameters.total_sequence_length,
                                                                          is_unidirectional_);

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             !cudnn_sdpa_supported &&
                             (nullptr == attention_bias) &&
                             nullptr == past &&
                             nullptr == present &&
                             parameters.hidden_size == parameters.v_hidden_size &&
                             nullptr == mask_index &&
                             onnxruntime::flash::is_supported<T>(device_prop,
                                                                 parameters.head_size,
                                                                 parameters.num_heads,
                                                                 parameters.num_heads);
  // When input is packed QKV format, TensorRT kernel might be faster when sequence length <= 512.
  if (use_flash_attention && parameters.sequence_length < kernel_options_->MinSeqLenForFlashAttentionPackedQkv()) {
    use_flash_attention = false;
  }
  // Allocate buffers
  size_t softmax_lse_bytes = 0;
  size_t softmax_lse_accum_bytes = 0;
  size_t out_accum_bytes = 0;
  if (use_flash_attention) {
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(sequence_length, batch_size, parameters.num_heads);

    using namespace std;
    auto [num_splits, slse_accum_bytes, o_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.total_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount);
    data.num_splits = static_cast<int>(num_splits);
    softmax_lse_accum_bytes = slse_accum_bytes;
    out_accum_bytes = o_accum_bytes;
  }
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, GetComputeStream(context));
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, GetComputeStream(context));
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, GetComputeStream(context));
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_buffer = GetScratchBuffer<void>(0, GetComputeStream(context));
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(0, GetComputeStream(context));  // nullptr
  auto out_accum_buffer = GetScratchBuffer<void>(0, GetComputeStream(context));          // nullptr
#endif

  if (!use_flash_attention && !is_unidirectional_) {  // BERT
    bool use_fused_runner = !cudnn_sdpa_supported &&
                            !disable_fused_self_attention_ &&
                            (nullptr == mask_index || is_mask_1d_seq_len) &&
                            nullptr == past &&
                            nullptr == present &&
                            nullptr == attention_bias &&
                            parameters.hidden_size == parameters.v_hidden_size &&
                            FusedMHARunnerFP16v2::IsSupported(sm, parameters.head_size, sequence_length,
                                                              enable_trt_flash_attention_);

    if (use_fused_runner) {
      // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
      if (nullptr == fused_fp16_runner_.get()) {
        std::call_once(fused_fp16_runner_created_, [&]() {
          fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm,
                                                            enable_trt_flash_attention_, parameters.scale);
        });
      }

      // In case some kernel not loaded due to shared memory limit, we need to double check here.
      const int normalized_seq_len = fused_fp16_runner_->NormalizeSequenceLength(sequence_length);
      if (fused_fp16_runner_->IsValid(normalized_seq_len)) {
        fused_runner = fused_fp16_runner_.get();
      }
    }
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  bool use_memory_efficient_attention =
      !use_flash_attention &&
      !cudnn_sdpa_supported &&
      fused_runner == nullptr &&
      !disable_memory_efficient_attention_ &&
      nullptr == past &&
      nullptr == present &&
      (nullptr == mask_index || parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) &&
      (sizeof(T) == 2 || parameters.sequence_length >= this->kernel_options_->MinSeqLenForEfficientAttentionFp32()) &&
      (nullptr == attention_bias || parameters.sequence_length % (4 * sizeof(T)) == 0) &&
      has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value, parameters.head_size, parameters.v_head_size);

#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_efficient_attention = use_memory_efficient_attention;
    debug_info.use_cudnn_flash_attention = cudnn_sdpa_supported;
    if (fused_runner != nullptr) {
      debug_info.SetTrtFusedKernel(enable_trt_flash_attention_, sequence_length);
    }

    debug_info.Print("Attention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  int m = batch_size * sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  IAllocatorUniquePtr<void> gemm_buffer = GetScratchBuffer<void>(static_cast<size_t>(m * n) * sizeof(T), GetComputeStream(context));

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop, UseTF32()));

  constexpr size_t element_size = sizeof(T);
  constexpr bool use_fused_cross_attention = false;
  constexpr bool use_lean_attention = false;
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
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
                                                   cudnn_sdpa_supported,
                                                   false);
  IAllocatorUniquePtr<void> work_space = GetScratchBuffer<void>(workSpaceSize, GetComputeStream(context));

  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  if (nullptr != bias) {
    data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  }
  if (nullptr != mask_index) {
    data.mask_index = mask_index->Data<int>();
    data.mask_index_dims = mask_index->Shape().GetDims();
  }
  if (nullptr != past) {
    data.past = reinterpret_cast<const CudaT*>(past->Data<T>());
  }
  if (nullptr != attention_bias) {
    data.attention_bias = reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
  }
  data.has_qkv_workspace = true;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workSpaceSize;
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  if (nullptr != present) {
    data.present = reinterpret_cast<CudaT*>(present->MutableData<T>());
  }
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  if (cudnn_sdpa_supported) {
    data.kernel_type = AttentionKernelType::AttentionKernel_CudnnFlashAttention;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&data.allocator));
  }
  if (softmax_lse_buffer != nullptr) {
    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
  }

  if (softmax_lse_accum_buffer != nullptr) {
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
  }
  if (out_accum_buffer != nullptr) {
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }

  cudnnHandle_t cudnn = GetCudnnHandle(context);
  return QkvToContext<CudaT>(device_prop, cublas, cudnn, ort_stream.get(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
