// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/decoder/decoder_masked_multihead_attention.h"
#include "contrib_ops/cuda/decoder/fastertransformer_decoder_attention/decoder_masked_multihead_attention_impl.h"

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
      DecoderMaskedMultiheadAttention,                                         \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex), \
      DecoderMaskedMultiheadAttention<T>);

REGISTER_KERNEL_TYPED(float)
//REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status DecoderMaskedMultiheadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(kPastInputIndex);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);

  auto& device_prop = GetDeviceProp();
  DecoderMaskedMultiheadAttentionParams parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  relative_position_bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock,
                                  past_seq_len));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  if (sequence_length != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input sequence length should be 1 to use DecoderMaskedMultiheadAttention");
  }

  if (!past_present_share_buffer_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Past Present share buffer must be turned on 1 to use DecoderMaskedMultiheadAttention");
  }

  // TODO(hasesh): Is there any value supporting this case ?
  if (parameters.head_size != parameters.v_head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QK head size should be same as V head size to use DecoderMaskedMultiheadAttention");
  }

  // TODO(hasesh): In future, we may support CrossAttention. Currently, this kernel only supports SelfAttention.
  if (parameters.sequence_length != parameters.kv_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "DecoderMaskedMultiheadAttention only supports self attention currently");
  }

  if (parameters.mask_type != AttentionMaskType::MASK_2D_KEY_PADDING &&
      parameters.mask_type != AttentionMaskType::MASK_NONE) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DecoderMaskedMultiheadAttention only supports no mask or 2D key "
                           "padding mask of shape [batch, total_seq_length] currently");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Present input will have the same shape as the past input
  Tensor* present = context->Output(kPresentOutputIndex, past->Shape());

  // Sanity check: Past/Present buffers should be the same to use this optimized kernel
  // The user of this kernel/ORT framework should ensure this.
  ORT_ENFORCE(present->MutableData<T>() == past->Data<T>());

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  IAllocatorUniquePtr<T> gemm_buffer;
  int m = batch_size * sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // QKV Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrices into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Update the q, k, and v buffers
  parameters.q = gemm_buffer.get();
  parameters.k = reinterpret_cast<CudaT*>(gemm_buffer.get()) + parameters.hidden_size;
  parameters.v = reinterpret_cast<CudaT*>(gemm_buffer.get()) + 2 * parameters.hidden_size;

  // Update the q, k, and v bias
  const T* bias_data = bias->Data<T>();
  parameters.q_bias = const_cast<T*>(bias_data);
  parameters.k_bias = const_cast<T*>(bias_data + parameters.hidden_size);
  parameters.v_bias = const_cast<T*>(bias_data + 2 * parameters.hidden_size);

  // Half of the past/present buffer correspond to K - the other half is V.
  auto k_size = present->Shape().Size() / 2;
  parameters.k_cache = present->MutableDataRaw();
  parameters.v_cache = present->MutableData<T>() + k_size;
  parameters.out = output->MutableDataRaw();

  // Mask
  if (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
    parameters.mask = mask_index->Data<int32_t>();
  }

  switch (parameters.head_size) {
    case 64:
      mmha_launch_kernel<T, 64>(parameters, Stream(context));
      break;

    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unsupported head size in DecoderMaskedMultiheadAttention");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
