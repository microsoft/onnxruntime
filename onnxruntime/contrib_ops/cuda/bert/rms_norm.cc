// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "rms_norm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      RmsNormalization,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RmsNormalization<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
RmsNormalization<T>::RmsNormalization(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK() && epsilon_ >= 0);
}

template <typename T>
Status RmsNormalization<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* weight = ctx->Input<Tensor>(1);

  Tensor* output = ctx->Output(0, input->Shape());
  typedef typename ToCudaType<T>::MappedType CudaT;

  auto input_shape = input->Shape();
  if (input_shape.NumDimensions() > 2) {
    input_shape[0] *= input_shape[1];
    input_shape[1] = input_shape[2];
  }
  LaunchRMSNormKernel<CudaT>(
      Stream(ctx),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      reinterpret_cast<const CudaT*>(weight->Data<T>()),
      epsilon_,
      input_shape.GetDims().data());

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
