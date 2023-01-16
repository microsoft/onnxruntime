// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "fast_gelu.h"
#include "fast_gelu_impl.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  const TransformerOptions* options = TransformerOptions::GetInstance();
  use_half2_ = !options->DisableHalf2();
}

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
  typedef typename ToCudaType<T>::MappedType CudaT;

  return LaunchFastGeluKernel<CudaT>(GetDeviceProp(),
                                     Stream(context),
                                     static_cast<int>(input_length),
                                     static_cast<int>(bias_length),
                                     reinterpret_cast<const CudaT*>(input->Data<T>()),
                                     (nullptr != bias) ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
                                     reinterpret_cast<CudaT*>(output->MutableData<T>()),
                                     use_half2_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
