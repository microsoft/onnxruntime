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
      DecoderMaskedSelfAttention,                                              \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex), \
      DecoderMaskedSelfAttention<T>);

REGISTER_KERNEL_TYPED(float)
//REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status DecoderMaskedSelfAttention<T>::ComputeInternal(OpKernelContext* context) const {
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
  assert(parameters.sequence_length == parameters.kv_sequence_length);  // self attention

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  // TODO: Send graceful error Status back
  ORT_ENFORCE(sequence_length == 1);
  ORT_ENFORCE(past_present_share_buffer_);

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      2, parameters.batch_size, parameters.num_heads, parameters.max_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(kPresentOutputIndex, present_shape);

  // Past/Present buffers should be shared
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

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Update the q,k, and v buffers
  parameters.q = gemm_buffer.get();
  parameters.k = reinterpret_cast<CudaT*>(gemm_buffer.get()) + parameters.hidden_size;
  parameters.v = reinterpret_cast<CudaT*>(gemm_buffer.get()) + 2 * parameters.hidden_size;

  cudaDeviceSynchronize();
  std::vector<float> before(768 * 3, 0);
  cudaMemcpy(before.data(), gemm_buffer.get(), 768 * 3 * 4, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  const T* bias_data = bias->Data<T>();
  parameters.q_bias = const_cast<T*>(bias_data);
  parameters.k_bias = const_cast<T*>(bias_data + parameters.hidden_size);
  parameters.v_bias = const_cast<T*>(bias_data + 2 * parameters.hidden_size);

  parameters.k_cache = present->MutableDataRaw();

  cudaDeviceSynchronize();
  std::vector<float> past_host(1 * 2 * 4 * 64, 0);
  cudaMemcpy(past_host.data(), past->DataRaw(), 1 * 2 * 4 * 64 * 4, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  switch (parameters.head_size) {
  case 64:
      mmha_launch_kernel<T, 64>(parameters, Stream(context));
      break;

  default:
      ORT_THROW("Unsupported head size");
  }

  cudaDeviceSynchronize();
  std::vector<float> after(768 * 3, 0);
  cudaMemcpy(after.data(), gemm_buffer.get(), 768 * 3 * 4, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaDeviceSynchronize();
  std::vector<float> present_host(1 * 2 * 4 * 64, 0);
  cudaMemcpy(present_host.data(), present->MutableDataRaw(), 1 * 2 * 4 * 64 * 4, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
