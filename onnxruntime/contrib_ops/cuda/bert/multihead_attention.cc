// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

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
    : CudaKernel(info), fused_fp16_cross_attention_kernel_(nullptr) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);

  disable_fused_runner_ = sizeof(T) != 2 ||
                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedAttention, false);

  enable_trt_flash_attention_ = sizeof(T) == 2 &&
                                !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);

#if USE_FLASH_ATTENTION
  disable_memory_efficient_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif

  disable_fused_cross_attention_ = sizeof(T) != 2 || ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedCrossAttention, false);
}

template <typename T>
Status MultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      key_padding_mask,
                                                                      relative_position_bias,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      device_prop.maxThreadsPerBlock));

  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  MHARunner* fused_runner = nullptr;

  const FusedMultiHeadCrossAttentionKernel* fused_cross_attention_kernel = nullptr;

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;

  bool is_mask_1d_seq_len = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  bool use_fused_cross_attention = !disable_fused_cross_attention_ &&
                                   nullptr == key_padding_mask &&
                                   nullptr == relative_position_bias &&
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

  bool use_fused_runner = !disable_fused_runner_ &&
                          fused_cross_attention_kernel == nullptr &&
                          nullptr == relative_position_bias &&
                          value != nullptr &&  // fused runner requires packed qkv instead of packed kv
                          (nullptr == key_padding_mask || is_mask_1d_seq_len) &&
                          parameters.hidden_size == parameters.v_hidden_size &&
                          parameters.sequence_length == parameters.kv_sequence_length &&
                          FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                             enable_trt_flash_attention_, false);
  if (use_fused_runner) {
    // Here we assume that num_heads and head_size does not change for a MultiHeadAttention node.
    if (nullptr == fused_fp16_runner_.get()) {
      constexpr bool is_unidirectional = false;
      fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(
          num_heads_, parameters.head_size, sm, is_unidirectional, enable_trt_flash_attention_, parameters.scale));
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
    if (fused_fp16_runner_->isValid(S)) {
      fused_runner = fused_fp16_runner_.get();
    }
  }

#if USE_FLASH_ATTENTION
  bool is_long_sequence = sizeof(T) == 2 ||  // sequence length threshold is 0 for FP16
                          parameters.sequence_length >= attention::kMinSequenceLengthForMemoryEfficientAttentionFp32 ||
                          parameters.kv_sequence_length >= attention::kMinSequenceLengthForMemoryEfficientAttentionFp32;

  bool use_memory_efficient_attention = fused_runner == nullptr &&
                                        fused_cross_attention_kernel == nullptr &&
                                        !disable_memory_efficient_attention_ &&
                                        is_long_sequence &&
                                        nullptr == key_padding_mask &&  // TODO: support 1D mask
                                        nullptr == relative_position_bias &&
                                        has_memory_efficient_attention(sm, sizeof(T) == 2);
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

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
  data.gemm_buffer = nullptr;
  data.bias = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (nullptr == value) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = (nullptr == key_padding_mask) ? nullptr : key_padding_mask->Data<int>();
  data.mask_index_dims = (nullptr == key_padding_mask) ? gsl::span<const int64_t>() : key_padding_mask->Shape().GetDims();
  data.past = nullptr;
  data.relative_position_bias = (nullptr == relative_position_bias) ? nullptr : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = nullptr;
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = fused_cross_attention_kernel;
  data.use_memory_efficient_attention = use_memory_efficient_attention;

  cublasHandle_t cublas = GetCublasHandle(context);
  return QkvToContext<CudaT>(
      device_prop, cublas, Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
