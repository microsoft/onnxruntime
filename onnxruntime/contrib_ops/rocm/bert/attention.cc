// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/attention.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

constexpr int kPastSequenceLengthInputIndex = 6;
constexpr int kPastInputIndex = 4;
constexpr int kPresentOutputIndex = 1;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : RocmKernel(info), AttentionBase(info, true) {}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);

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
                                  past_seq_len));
  ORT_ENFORCE(parameters.sequence_length == parameters.kv_sequence_length);  // self attention 

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      2, parameters.batch_size, parameters.num_heads,
      parameters.past_present_share_buffer ? parameters.max_sequence_length : parameters.total_sequence_length,
      parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(kPresentOutputIndex, present_shape);

  rocblas_handle rocblas = GetRocblasHandle(context);
  constexpr size_t element_size = sizeof(T);

  int m = parameters.batch_size * parameters.sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  typedef typename ToHipType<T>::MappedType HipT;
  namespace blas = rocm::tunable::blas;

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  ORT_RETURN_IF_ERROR(blas::column_major::Gemm(
      GetTuningContext(), Stream(context), rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      n, m, 1,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(bias->Data<T>()), n,
      GetConstOnes<HipT>(m, Stream(context)), 1,
      /*beta=*/0.0f,
      reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  // result(N, M) = 1 * weights x input + 1 x B.
  ORT_RETURN_IF_ERROR(blas::column_major::Gemm(
      GetTuningContext(), Stream(context), rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      n, m, k,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(weights->Data<T>()), n,
      reinterpret_cast<const HipT*>(input->Data<T>()), k,
      /*beta=*/1.0f,
      reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, 
                                                   parameters.batch_size,
                                                   parameters.num_heads, 
                                                   parameters.head_size,
                                                   parameters.sequence_length,
                                                   parameters.past_sequence_length); 

  auto work_space = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());
  return LaunchAttentionKernel(
      device_prop,
      GetTuningContext(),
      Stream(context),
      rocblas,
      element_size,
      parameters.batch_size,
      parameters.sequence_length,
      parameters.num_heads,
      parameters.head_size,
      parameters.past_sequence_length,
      parameters.is_unidirectional,
      reinterpret_cast<const void*>(gemm_buffer.get()),
      nullptr == mask_index ? nullptr : mask_index->Data<int>(),
      nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
      parameters.mask_filter_value,
      nullptr == past ? nullptr : past->Data<T>(),
      nullptr == relative_position_bias ? nullptr : relative_position_bias->Data<T>(),
      work_space.get(),
      output->MutableData<T>(),
      nullptr == present ? nullptr : present->MutableData<T>());
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
