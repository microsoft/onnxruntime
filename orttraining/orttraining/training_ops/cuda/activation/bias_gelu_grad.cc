// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/bias_gelu_grad.h"

#include "core/common/common.h"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"
#include "orttraining/training_ops/cuda/activation/bias_gelu_grad_impl.h"

namespace onnxruntime {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define ALL_IEEE_FLOAT_DATA_TYPES MLFloat16, float, double, BFloat16
#else
#define ALL_IEEE_FLOAT_DATA_TYPES MLFloat16, float, double
#endif

ONNX_OPERATOR_KERNEL_EX(
    BiasGeluGrad_dX,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<ALL_IEEE_FLOAT_DATA_TYPES>())
        .MayInplace(0, 0),
    BiasGeluGrad_dX<gelu_computation_mode::Default>);

ONNX_OPERATOR_KERNEL_EX(
    BiasFastGeluGrad_dX,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<ALL_IEEE_FLOAT_DATA_TYPES>())
        .MayInplace(0, 0),
    BiasGeluGrad_dX<gelu_computation_mode::Approximation>);

template <typename GeluComputationMode>
template <typename T>
void BiasGeluGrad_dX<GeluComputationMode>::KernelLaunchDispatcher<T>::operator()(
    cudaStream_t stream,
    int64_t input_size, int64_t bias_size,
    const Tensor& dY, const Tensor& X, const Tensor& B,
    Tensor& dX) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  LaunchBiasGeluGradDxKernel<CudaT, GeluComputationMode>(
      stream,
      input_size, bias_size,
      reinterpret_cast<const CudaT*>(dY.template Data<T>()),
      reinterpret_cast<const CudaT*>(X.template Data<T>()),
      reinterpret_cast<const CudaT*>(B.template Data<T>()),
      reinterpret_cast<CudaT*>(dX.template MutableData<T>()));
}

template <typename GeluComputationMode>
Status BiasGeluGrad_dX<GeluComputationMode>::ComputeInternal(OpKernelContext* context) const {
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
      input_shape.NumDimensions() >= 1 && bias_shape.NumDimensions() == 1 &&
          input_shape.GetDims().back() == bias_shape.GetDims().back(),
      "B must be 1-dimensional and match the last dimension of X.");

  auto* dX = context->Output(0, input_shape);
  ORT_ENFORCE(dX);

  const auto input_size = input_shape.Size(), bias_size = bias_shape.Size();

  utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> dispatcher{X->GetElementType()};
  dispatcher.Invoke<KernelLaunchDispatcher>(Stream(), input_size, bias_size, *dY, *X, *B, *dX);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
