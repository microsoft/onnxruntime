// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

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

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, false) {
  disable_fused_self_attention_ = sizeof(T) != 2 ||
                                  ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedSelfAttention, false);

  enable_trt_flash_attention_ = sizeof(T) == 2 &&
                                !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);

  enable_fused_causal_attention_ = sizeof(T) == 2 &&
                                   ParseEnvironmentVariableWithDefault<bool>(attention::kEnableFusedCausalAttention, false);

#if USE_FLASH_ATTENTION
  disable_memory_efficient_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(kPastInputIndex);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  // Use the second dimension from weight for bias to get q_hidden_size when bias is nullptr
  std::vector<int64_t> bias_dims{weights->Shape().GetDims()[1]};
  const TensorShape bias_shape{bias_dims};
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias != nullptr ? bias->Shape() : bias_shape,
                                  mask_index,
                                  past,
                                  relative_position_bias,
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
  int sm = device_prop.major * 10 + device_prop.minor;
  bool is_mask_1d_seq_len = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
  bool is_mask_1d_key_seq_len_start = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START;

  if (is_unidirectional_ && enable_fused_causal_attention_) {  // GPT
    // GPT fused kernels requires left side padding. mask can be:
    //     none (no padding), 1D sequence lengths or 2d mask.
    // Fused kernels don't support different sequence lengths of q and kv, so only apply to the first token
    // where past state is empty.
    bool is_mask_2d_key_padding = parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING;
    bool use_causal_fused_runner = (nullptr == mask_index || is_mask_1d_seq_len || is_mask_2d_key_padding) &&
                                   nullptr == relative_position_bias &&
                                   parameters.past_sequence_length == 0 &&
                                   parameters.hidden_size == parameters.v_hidden_size &&
                                   FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                                      enable_trt_flash_attention_, true);
    if (use_causal_fused_runner) {
      // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
      if (nullptr == fused_fp16_runner_.get()) {
        fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                          enable_trt_flash_attention_, parameters.scale);
      }

      // Here we assume all causal kernels can be loaded into shared memory. TODO: add a function to check.
      fused_runner = fused_fp16_runner_.get();
    }
  } else {  // BERT
    bool use_fused_runner = !disable_fused_self_attention_ &&
                            (nullptr == mask_index || is_mask_1d_seq_len) &&
                            nullptr == past &&
                            nullptr == present &&
                            nullptr == relative_position_bias &&
                            parameters.hidden_size == parameters.v_hidden_size &&
                            FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                               enable_trt_flash_attention_, false);

    if (use_fused_runner) {
      // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
      if (nullptr == fused_fp16_runner_.get()) {
        fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                          enable_trt_flash_attention_, parameters.scale);
      }

      // In case some kernel not loaded due to shared memory limit, we need to double check here.
      const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
      if (fused_fp16_runner_->isValid(S)) {
        fused_runner = fused_fp16_runner_.get();
      }
    }
  }

#if USE_FLASH_ATTENTION
  bool is_good_for_rpb = relative_position_bias != nullptr && parameters.sequence_length % (4 * sizeof(T)) == 0;
  bool use_memory_efficient_attention = fused_runner == nullptr &&
                                        !disable_memory_efficient_attention_ &&
                                        (nullptr == mask_index || is_mask_1d_key_seq_len_start) &&
                                        nullptr == past &&
                                        nullptr == present &&
                                        (nullptr == relative_position_bias || is_good_for_rpb) &&
                                        (sizeof(T) == 2 ||  // sequence length threshold is 0 in FP16
                                         parameters.sequence_length >= attention::kMinSequenceLengthForMemoryEfficientAttentionFp32) &&
                                        has_memory_efficient_attention(sm, sizeof(T) == 2);
#else
  constexpr bool use_memory_efficient_attention = false;
  ORT_UNUSED_PARAMETER(is_mask_1d_key_seq_len_start);
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  IAllocatorUniquePtr<T> gemm_buffer;
  int m = batch_size * sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  constexpr size_t element_size = sizeof(T);
  constexpr bool use_fused_cross_attention = false;
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner,
                                                   use_fused_cross_attention,
                                                   use_memory_efficient_attention);
  auto work_space = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  data.bias = nullptr == bias ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = nullptr;
  data.key = nullptr;
  data.value = nullptr;
  data.mask_index = (nullptr == mask_index) ? nullptr : mask_index->Data<int>();
  data.mask_index_dims = (nullptr == mask_index) ? gsl::span<const int64_t>() : mask_index->Shape().GetDims();
  data.past = (nullptr == past) ? nullptr : reinterpret_cast<const CudaT*>(past->Data<T>());
  data.past_key = nullptr;
  data.past_value = nullptr;
  data.relative_position_bias = (nullptr == relative_position_bias) ? nullptr : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.has_qkv_workspace = true;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = (nullptr == present) ? nullptr : reinterpret_cast<CudaT*>(present->MutableData<T>());
  data.present_key = nullptr;
  data.present_value = nullptr;
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = nullptr;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  data.cumulated_sequence_length_q_cache = nullptr;
  data.cumulated_sequence_length_kv_cache = nullptr;

  return QkvToContext<CudaT>(device_prop, cublas, Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
