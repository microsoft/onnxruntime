// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/fast_gelu.h"

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/miopen_common.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "contrib_ops/rocm/bert/elementwise.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, input->Shape());

  int64_t input_length = input->Shape().Size();
  if (input_length == 0) {
    return Status::OK();
  }
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();
  typedef typename ToHipType<T>::MappedType HipT;

  const HipT* input_buffer = reinterpret_cast<const HipT*>(input->Data<T>());
  const HipT* bias_buffer = (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->Data<T>()) : nullptr;
  return LaunchElementwiseKernel<functor::FastGeLU, HipT>(
      GetTuningContext(), context->GetComputeStream(),
      input_buffer, static_cast<int>(input_length),
      bias_buffer, static_cast<int>(bias_length),
      reinterpret_cast<HipT*>(output->MutableData<T>()));
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
