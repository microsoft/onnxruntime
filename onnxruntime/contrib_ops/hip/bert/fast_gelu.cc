// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/hip/miopen_common.h"
#include "core/framework/tensorprotoutils.h"
#include "fast_gelu.h"
#include "fast_gelu_impl.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"

namespace onnxruntime {
namespace contrib {
namespace hip {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kHipExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : HipKernel(op_kernel_info) {
}

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, input->Shape());

  int64_t input_length = input->Shape().Size();
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();
  typedef typename ToHipType<T>::MappedType HipT;
  if (!LaunchFastGeluKernel<HipT>(GetDeviceProp(),
                                   nullptr,
                                   static_cast<int>(input_length),
                                   static_cast<int>(bias_length),
                                   reinterpret_cast<const HipT*>(input->template Data<T>()),
                                   (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->template Data<T>()) : nullptr,
                                   reinterpret_cast<HipT*>(output->template MutableData<T>()))) {
    HIP_CALL(hipGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  //namespace hip
}  // namespace contrib
}  // namespace onnxruntime
