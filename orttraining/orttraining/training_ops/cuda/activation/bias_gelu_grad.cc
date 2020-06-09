// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/bias_gelu_grad.h"

#include "core/common/common.h"
#include "orttraining/training_ops/cuda/activation/bias_gelu_grad_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BiasGeluGrad_dX,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double>())
        .MayInplace(0, 0),
    BiasGeluGrad_dX<false>);

ONNX_OPERATOR_KERNEL_EX(
    BiasFastGeluGrad_dX,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double>())
        .MayInplace(0, 0),
    BiasGeluGrad_dX<true>);

namespace {
template <typename T>
struct BiasGeluGradDxDispatcher {
  void operator()(
      int64_t input_size, int64_t bias_size,
      const Tensor& dY, const Tensor& X, const Tensor& B,
      Tensor& dX) const {
    using CudaT = typename ToCudaType<T>::MappedType;
    LaunchBiasGeluGradDxKernel(
        input_size, bias_size,
        reinterpret_cast<const CudaT*>(dY.template Data<T>()),
        reinterpret_cast<const CudaT*>(X.template Data<T>()),
        reinterpret_cast<const CudaT*>(B.template Data<T>()),
        reinterpret_cast<CudaT*>(dX.template MutableData<T>()));
  }
};

template <typename T>
struct BiasGeluApproximationGradDxDispatcher {
  void operator()(
      int64_t input_size, int64_t bias_size,
      const Tensor& dY, const Tensor& X, const Tensor& B,
      Tensor& dX) const {
    using CudaT = typename ToCudaType<T>::MappedType;
    LaunchBiasGeluApproximationGradDxKernel(
        input_size, bias_size,
        reinterpret_cast<const CudaT*>(dY.template Data<T>()),
        reinterpret_cast<const CudaT*>(X.template Data<T>()),
        reinterpret_cast<const CudaT*>(B.template Data<T>()),
        reinterpret_cast<CudaT*>(dX.template MutableData<T>()));
  }
};
}  // namespace

template <bool use_approximation>
Status BiasGeluGrad_dX<use_approximation>::ComputeInternal(OpKernelContext* context) const {
  const auto* dY = context->Input<Tensor>(0);
  ORT_ENFORCE(dY);
  const auto* X = context->Input<Tensor>(1);
  ORT_ENFORCE(X);
  const auto* B = context->Input<Tensor>(2);
  ORT_ENFORCE(B);

  const auto& input_shape = X->Shape();
  ORT_ENFORCE(input_shape == dY->Shape(), "dY and X must have the same shape.");
  const auto& bias_shape = B->Shape();
  ORT_ENFORCE(
      input_shape.NumDimensions() >= 1 && bias_shape.NumDimensions() == 1 && input_shape.GetDims().back() == bias_shape.GetDims().back(),
      "B must be 1-dimensional and match the last dimension of X.");

  auto* dX = context->Output(0, dY->Shape());
  ORT_ENFORCE(dX);

  const auto input_size = input_shape.Size(), bias_size = bias_shape.Size();
  ORT_ENFORCE(input_size > 0 && bias_size > 0, "dY, X, and B sizes must be greater than 0.");

  if (use_approximation) {
    utils::MLTypeCallDispatcher<
        BiasGeluApproximationGradDxDispatcher,
        MLFloat16, float, double>
        dispatcher{X->GetElementType()};
    dispatcher.Invoke(input_size, bias_size, *dY, *X, *B, *dX);
  } else {
    utils::MLTypeCallDispatcher<
        BiasGeluGradDxDispatcher,
        MLFloat16, float, double>
        dispatcher{X->GetElementType()};
    dispatcher.Invoke(input_size, bias_size, *dY, *X, *B, *dX);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
