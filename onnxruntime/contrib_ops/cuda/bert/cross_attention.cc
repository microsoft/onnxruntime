// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
//#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/cross_attention.h"
//#include "contrib_ops/cuda/bert/bert_padding.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      CrossAttention,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      CrossAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
CrossAttention<T>::CrossAttention(const OpKernelInfo& info) : CudaKernel(info), CrossAttentionBase(info) {
  disable_fused_runner_ = sizeof(T) != 2 ||
                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedAttention, false);

  enable_flash_attention_ = sizeof(T) == 2 &&
                            ParseEnvironmentVariableWithDefault<bool>(attention::kEnableFlashAttention, true);
}

template <typename T>
Status CrossAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(query,
                                  key,
                                  value,
                                  bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  MHARunner* fused_runner = nullptr;
#ifndef ENABLE_TRAINING  // Only enable fused kernel on non-training builds
  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;

  bool use_fused_runner = !disable_fused_runner_ &&
                          parameters.hidden_size == parameters.v_hidden_size &&
                          parameters.sequence_length == parameters.kv_sequence_length &&
                          FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                             enable_flash_attention_, false);

  if (use_fused_runner) {
    // Here we assume that num_heads, head_size and is_unidirectional does not change for an CrossAttention node.
    if (nullptr == fused_fp16_runner_.get()) {
      constexpr bool is_unidirectional = false;
      fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(
          num_heads_, parameters.head_size, sm, is_unidirectional, enable_flash_attention_));
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
    if (fused_fp16_runner_->isValid(S)) {
      fused_runner = fused_fp16_runner_.get();
    }
  }
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  constexpr size_t element_size = sizeof(T);
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner);
  auto work_space = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = nullptr;
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = nullptr;
  data.mask_index_dims = gsl::span<const int64_t>();
  data.past = nullptr;
  data.extra_add_qk = nullptr;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = nullptr;

  return QkvToContext<CudaT>(
    device_prop, cublas, Stream(context), parameters, data, reinterpret_cast<void*>(fused_runner), false);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
