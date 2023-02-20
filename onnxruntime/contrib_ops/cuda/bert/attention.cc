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
constexpr int kPackingTokenOffsetInputIndex = 7;
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
MHARunner* Attention<T>::TryGetFusedMHARunner(const AttentionParameters& parameters, int sm) const {
  MHARunner* fused_runner = nullptr;

  // Check whether we can use fused kernel
  bool is_mask_1d_seq_len = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  if (is_unidirectional_) {  // GPT
    // GPT fused kernels requires left side padding. mask can be:
    //     none (no padding), 1D sequence lengths or 2d mask.
    // Fused kernels don't support different sequence lengths of q and kv, so only apply to the first token
    // where past state is empty.
    bool is_mask_2d_key_padding = parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING;
    bool use_causal_fused_runner = !disable_fused_runner_ &&
                                   (!parameters.has_mask_index || is_mask_1d_seq_len || is_mask_2d_key_padding) &&
                                   !parameters.has_relative_position_bias &&
                                   parameters.past_sequence_length == 0 &&
                                   parameters.hidden_size == parameters.v_hidden_size &&
                                   FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, parameters.sequence_length,
                                                                      enable_trt_flash_attention_, true);
    if (use_causal_fused_runner) {
      // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
      if (nullptr == fused_fp16_runner_.get()) {
        fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                          enable_trt_flash_attention_, parameters.scale));
      }

      // Here we assume all causal kernels can be loaded into shared memory. TODO: add a function to check.
      fused_runner = fused_fp16_runner_.get();
    }
  } else {  // BERT
    bool use_fused_runner = !disable_fused_runner_ &&
                            (!parameters.has_mask_index || is_mask_1d_seq_len) &&
                            !parameters.has_past &&
                            !parameters.has_present &&
                            !parameters.has_relative_position_bias &&
                            parameters.hidden_size == parameters.v_hidden_size &&
                            FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, parameters.sequence_length,
                                                               enable_trt_flash_attention_, false);

    if (use_fused_runner) {
      // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
      if (nullptr == fused_fp16_runner_.get()) {
        fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                          enable_trt_flash_attention_, parameters.scale));
      }

      // In case some kernel not loaded due to shared memory limit, we need to double check here.
      const int S = fused_fp16_runner_->getSFromMaxSeqLen(parameters.sequence_length);
      if (fused_fp16_runner_->isValid(S)) {
        fused_runner = fused_fp16_runner_.get();
      }
    }
  }

  return fused_runner;
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, false) {
  disable_fused_runner_ = sizeof(T) != 2 ||
                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedAttention, false);

  enable_trt_flash_attention_ = sizeof(T) == 2 &&
                                !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);

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
  const Tensor* packing_token_offset = context->Input<Tensor>(kPackingTokenOffsetInputIndex);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  relative_position_bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock,
                                  past_seq_len,
                                  packing_token_offset));
  assert(parameters.sequence_length == parameters.kv_sequence_length);  // self attention

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  Tensor* output = nullptr;
  if (is_packing_mode_) {
    TensorShapeVector output_shape{parameters.total_token_count, parameters.v_hidden_size};
    output = context->Output(0, output_shape);
  } else {
    TensorShapeVector output_shape{batch_size, sequence_length, parameters.v_hidden_size};
    output = context->Output(0, output_shape);
  }

  std::vector<int64_t> present_dims{
      2, parameters.batch_size, parameters.num_heads,
      past_present_share_buffer_ ? parameters.max_sequence_length : parameters.total_sequence_length,
      parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(kPresentOutputIndex, present_shape);
  parameters.has_present = nullptr != present;

  int sm = device_prop.major * 10 + device_prop.minor;
  MHARunner* fused_runner = TryGetFusedMHARunner(parameters, sm);

#if USE_FLASH_ATTENTION
  bool use_memory_efficient_attention = fused_runner == nullptr &&
                                        !disable_memory_efficient_attention_ &&
                                        !parameters.has_mask_index &&  // TODO: support 1D mask
                                        !parameters.has_past &&
                                        !parameters.has_present &&
                                        !parameters.has_relative_position_bias &&
                                        (sizeof(T) == 2 ||  // sequence length threshold is 0 in FP16
                                         parameters.sequence_length >= attention::kMinSequenceLengthForMemoryEfficientAttentionFp32) &&
                                        has_memory_efficient_attention(sm, sizeof(T) == 2);
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  IAllocatorUniquePtr<T> gemm_buffer;
  int m = batch_size * sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;

  size_t gemm_buffer_size = static_cast<size_t>(m) * n;
  if (is_packing_mode_) {
    m = parameters.total_token_count;
    gemm_buffer_size += static_cast<size_t>(m) * n;
    gemm_buffer_size += batch_size * sequence_length * parameters.v_hidden_size;  // output buffer
  }
  gemm_buffer = GetScratchBuffer<T>(gemm_buffer_size, context->GetComputeStream());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  CudaT* gemm_buffer_data = reinterpret_cast<CudaT*>(gemm_buffer.get());
  CudaT* output_buffer = reinterpret_cast<CudaT*>(output->MutableData<T>());
  if (is_packing_mode_) {
    const int32_t* token_offset_data = packing_token_offset->Data<int32_t>();
    CudaT* gemm_buffer_data_padding = gemm_buffer_data + m * n;
    LaunchRestorePadding(
        gemm_buffer_data_padding,
        reinterpret_cast<CudaT*>(gemm_buffer.get()),
        token_offset_data,
        parameters.total_token_count,
        n,
        batch_size,
        sequence_length,
        Stream(context));
    gemm_buffer_data = gemm_buffer_data_padding;

    output_buffer = gemm_buffer_data_padding + batch_size * sequence_length * n;
  }

  constexpr size_t element_size = sizeof(T);
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner,
                                                   use_memory_efficient_attention);
  auto work_space = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = gemm_buffer_data;
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = nullptr;
  data.key = nullptr;
  data.value = nullptr;
  data.mask_index = (nullptr == mask_index) ? nullptr : mask_index->Data<int>();
  data.mask_index_dims = (nullptr == mask_index) ? gsl::span<const int64_t>() : mask_index->Shape().GetDims();
  data.past = (nullptr == past) ? nullptr : reinterpret_cast<const CudaT*>(past->Data<T>());
  data.relative_position_bias = (nullptr == relative_position_bias) ? nullptr : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = output_buffer;
  data.present = (nullptr == present) ? nullptr : reinterpret_cast<CudaT*>(present->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = nullptr;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  // data.token_offset_data = token_offset_data;

  Status status = QkvToContext<CudaT>(device_prop, cublas, Stream(context), parameters, data);
  if (is_packing_mode_) {
    const int32_t* token_offset_data = packing_token_offset->Data<int32_t>();
    LaunchRemovePadding(reinterpret_cast<CudaT*>(output->MutableData<T>()),
                        output_buffer,
                        token_offset_data,
                        parameters.total_token_count,
                        parameters.v_hidden_size,
                        Stream(context));
  }

  return status;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
