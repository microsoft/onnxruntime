// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/activation/activations_grad.h"

#include "core/common/gsl.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "core/common/eigen_common_wrapper.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif

#include "core/util/math_cpuonly.h"

#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    GeluGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    GeluGrad<float, gelu_computation_mode::Default>);

ONNX_OPERATOR_KERNEL_EX(
    FastGeluGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    GeluGrad<float, gelu_computation_mode::Approximation>);

ONNX_OPERATOR_KERNEL_EX(
    BiasGeluGrad_dX,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasGeluGrad_dX<float, gelu_computation_mode::Default>);

ONNX_OPERATOR_KERNEL_EX(
    BiasFastGeluGrad_dX,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasGeluGrad_dX<float, gelu_computation_mode::Approximation>);

namespace {
template <typename T>
Status ComputeGeluGradDX(gsl::span<const T> dY, gsl::span<const T> X, gsl::span<T> dX,
                         gelu_computation_mode::Default) {
  static constexpr T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2 * 0.5);

  ConstEigenVectorArrayMap<T> X_array(X.data(), X.size());
  ConstEigenVectorArrayMap<T> dY_array(dY.data(), dY.size());
  EigenVectorArrayMap<T> dX_array(dX.data(), dX.size());

  dX_array = dY_array * (0.5f * ((X_array * static_cast<T>(M_SQRT1_2)).erf() + 1.0f) +
                         X_array * kAlpha * (-0.5f * X_array * X_array).exp());

  return Status::OK();
}

template <typename T>
Status ComputeGeluGradDX(gsl::span<const T> dY, gsl::span<const T> X, gsl::span<T> dX,
                         gelu_computation_mode::Approximation) {
  static constexpr T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
  static constexpr T kGamma = static_cast<T>(0.044715f);
  static constexpr T kBeta = static_cast<T>(kGamma * kAlpha * 3.0f);

  //
  // Commented out EIGEN implentation due to EIGEN bug.
  // On Windows Release build with GPU enabled, kAlpha * EIGEN_X below would produce pure 0
  // result, even though neither kAlpha nor EIGEN_X is zero.
  // Given that CPU kernel is mostly for conformance check, where performance is not of high
  // priority, to workaround this bug, use a for loop and avoid using EIGEN library.
  //
  // EIGEN_X_VAR(xm);
  // EIGEN_DY_VAR(dy);

  // const auto x_cube = EIGEN_X.cube();
  // const auto tanh_result = ((T)kAlpha * (EIGEN_X + kGamma * x_cube)).tanh();
  // const auto sech_sqr_result = 1 - (tanh_result * tanh_result);

  // EIGEN_DX = dy * (0.5f * (tanh_result + sech_sqr_result * (kAlpha * xm + kBeta * x_cube) + 1));
  //
  const T* dY_data = dY.data();
  const T* X_data = X.data();
  T* dX_data = dX.data();
  int64_t elem_count = X.size();
  for (auto i = 0; i < elem_count; ++i) {
    const auto x_val = X_data[i];
    const auto x_cube = x_val * x_val * x_val;
    T tanh_result = std::tanh(kAlpha * x_val + kAlpha * kGamma * x_cube);
    T sech_sqr_result = 1 - (tanh_result * tanh_result);
    dX_data[i] = (dY_data[i]) * (0.5f * (tanh_result + sech_sqr_result * (kAlpha * x_val + kBeta * x_cube) + 1));
  }
  return Status::OK();
}
}  // namespace

template <typename T, typename GeluComputationMode>
Status GeluGrad<T, GeluComputationMode>::Compute(OpKernelContext* context) const {
  const auto* dY = context->Input<Tensor>(0);
  ORT_ENFORCE(dY);
  const auto* X = context->Input<Tensor>(1);
  ORT_ENFORCE(X);
  Tensor* dX = context->Output(0, X->Shape());
  ORT_ENFORCE(dX);

  ORT_RETURN_IF_ERROR((ComputeGeluGradDX<T>(
      dY->template DataAsSpan<T>(), X->template DataAsSpan<T>(), dX->template MutableDataAsSpan<T>(),
      GeluComputationMode{})));

  return Status::OK();
}

template <typename T, typename GeluComputationMode>
Status BiasGeluGrad_dX<T, GeluComputationMode>::Compute(OpKernelContext* context) const {
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

  const auto input_size = narrow<Eigen::Index>(input_shape.Size()),
             bias_size = narrow<Eigen::Index>(bias_shape.Size());

  // X + B, broadcasting
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  auto X_plus_B_buffer = IAllocator::MakeUniquePtr<T>(allocator, input_size);

  // these are column-major array maps
  ConstEigenArrayMap<T> X_array(X->template Data<T>(), bias_size, input_size / bias_size);
  ConstEigenVectorArrayMap<T> B_vector(B->template Data<T>(), bias_size);
  EigenArrayMap<T> X_plus_B_array(X_plus_B_buffer.get(), bias_size, input_size / bias_size);

  X_plus_B_array = X_array.colwise() + B_vector;

  // dX
  const auto biased_X_span = gsl::make_span<const T>(X_plus_B_buffer.get(), narrow<size_t>(X->Shape().Size()));
  ORT_RETURN_IF_ERROR((ComputeGeluGradDX<T>(
      dY->template DataAsSpan<T>(), biased_X_span, dX->template MutableDataAsSpan<T>(),
      GeluComputationMode{})));

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
