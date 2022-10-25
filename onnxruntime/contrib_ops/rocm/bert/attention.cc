// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/attention.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : RocmKernel(info), AttentionBase(info, true, true) {}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* extra_add_qk = context->Input<Tensor>(5);

  const Tensor* key = context->Input<Tensor>(6);
  const Tensor* value = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights == nullptr ? nullptr : &(weights->Shape()),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  extra_add_qk,
                                  key,
                                  value,
                                  nullptr,
                                  device_prop.maxThreadsPerBlock));

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int input_hidden_size = static_cast<int>(shape[2]);

  // Note: Scenario where q_hidden_size == k_hidden_size != v_hidden_size is not supported in ROCM EP
  // bias shape (3 * hidden_size)
  const auto& bias_shape = bias->Shape();
  int hidden_size = static_cast<int>(bias_shape[0]) / 3;

  int head_size = hidden_size / num_heads_;

  TensorShapeVector output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  int past_sequence_length = 0;
  Tensor* present = GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length);

  rocblas_handle rocblas = RocblasHandle();
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(batch_size * sequence_length * 3 * hidden_size * element_size);

  typedef typename ToHipType<T>::MappedType HipT;
  HipT one = ToHipType<T>::FromFloat(1.0f);
  HipT zero = ToHipType<T>::FromFloat(0.0f);

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
      rocblas, rocblas_operation_none, rocblas_operation_none, n, m, 1, &one,
      reinterpret_cast<const HipT*>(bias->Data<T>()), n,
      GetConstOnes<HipT>(m), 1,
      &zero, reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  // Gemm, note that ROCM assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
      rocblas, rocblas_operation_none, rocblas_operation_none, n, m, k, &one,
      reinterpret_cast<const HipT*>(weights->Data<T>()), n,
      reinterpret_cast<const HipT*>(input->Data<T>()), k,
      &one, reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size,
                                                   sequence_length, past_sequence_length);

  auto work_space = GetScratchBuffer<void>(workSpaceSize);
  return LaunchAttentionKernel(
      device_prop,
      Stream(),
      rocblas,
      element_size,
      batch_size,
      sequence_length,
      num_heads_,
      head_size,
      past_sequence_length,
      is_unidirectional_,
      reinterpret_cast<const void*>(gemm_buffer.get()),
      nullptr == mask_index ? nullptr : mask_index->Data<int>(),
      nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
      nullptr == past ? nullptr : past->Data<T>(),
      nullptr == extra_add_qk ? nullptr : extra_add_qk->Data<T>(),
      work_space.get(),
      output->MutableData<T>(),
      nullptr == present ? nullptr : present->MutableData<T>());
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
