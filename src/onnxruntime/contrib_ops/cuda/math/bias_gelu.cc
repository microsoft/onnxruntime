// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_gelu.h"

#include "contrib_ops/cuda/math/bias_gelu_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BiasGelu, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    BiasGelu);

template <typename T>
void BiasGelu::KernelLaunchDispatcher<T>::operator()(cudaStream_t stream, int64_t input_size, int64_t bias_size,
                                                     const Tensor& X, const Tensor& B, Tensor& Y) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  LaunchBiasGeluKernel<CudaT>(stream, input_size, bias_size, reinterpret_cast<const CudaT*>(X.template Data<T>()),
                              reinterpret_cast<const CudaT*>(B.template Data<T>()),
                              reinterpret_cast<CudaT*>(Y.template MutableData<T>()));
}

Status BiasGelu::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X);
  const auto* B = context->Input<Tensor>(1);
  ORT_ENFORCE(B);

  const auto& input_shape = X->Shape();
  const auto& bias_shape = B->Shape();
  ORT_ENFORCE(input_shape.NumDimensions() >= 1 && bias_shape.NumDimensions() == 1 &&
                  input_shape.GetDims().back() == bias_shape.GetDims().back(),
              "B must be 1-dimensional and match the last dimension of X.");

  auto* Y = context->Output(0, input_shape);
  ORT_ENFORCE(Y);

  const auto input_size = input_shape.Size();
  const auto bias_size = bias_shape.Size();
  utils::MLTypeCallDispatcher<MLFloat16, float, double, BFloat16> dispatcher{X->GetElementType()};
  dispatcher.Invoke<KernelLaunchDispatcher>(Stream(context), input_size, bias_size, *X, *B, *Y);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
