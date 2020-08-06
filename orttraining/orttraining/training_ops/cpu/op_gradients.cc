// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_gradients.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    SinGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SinGrad<float>);

template <typename T>
Status SinGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& X = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, X.Shape());
  MakeEigenArrayMap<float>(dX) = MakeEigenArrayMap<float>(dY) * MakeEigenArrayMap<float>(X).cos();
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    ReluGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReluGrad<float>);

template <typename T>
Status ReluGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& X = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, dY.Shape());

  EigenVectorArrayMap<float>(dX.template MutableData<T>(), dX.Shape().Size()) =
      (ConstEigenVectorArrayMap<float>(X.template Data<T>(), X.Shape().Size()) > T(0))
          .select(ConstEigenVectorArrayMap<float>(dY.template Data<T>(), dY.Shape().Size()), T(0));

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SoftmaxGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxGrad<float>);

template <typename T>
Status SoftmaxGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  const TensorShape input_shape{Y.Shape()};
  auto& dX = *context->Output(0, Y.Shape());

  auto axis = HandleNegativeAxis(axis_, Y.Shape().NumDimensions());

  size_t N = input_shape.SizeToDimension(axis);
  size_t D = input_shape.SizeFromDimension(axis);

  if (N == 0) {
    return Status::OK();
  }

  std::vector<float> scale_(N);
  std::vector<float> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0
  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  float* scaledata = scale_.data();
  const float* Ydata = Y.template Data<float>();
  const float* dYdata = dY.template Data<float>();
  float* dXdata = dX.template MutableData<float>();

  gsl::copy(gsl::make_span(dYdata, nd), gsl::make_span(dXdata, nd));

  for (size_t i = 0; i < N; ++i) {
    math::Dot<float, CPUMathUtil>(d, Ydata + i * d, dYdata + i * d,
                                  scaledata + i, nullptr);
  }

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, n, d, 1, -1,
                    scaledata, sum_multiplier_.data(), 1,
                    dXdata, tp);

  math::Mul<float, CPUMathUtil>(gsl::narrow_cast<int>(Y.Shape().Size()), dXdata, Ydata, dXdata, nullptr);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
  LogSoftmaxGrad,
  kMSDomain,
  1,
  kCpuExecutionProvider,
  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
  LogSoftmaxGrad<float>);

template <typename T>
Status LogSoftmaxGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  const TensorShape input_shape{Y.Shape()};
  auto& dX = *context->Output(0, Y.Shape());

  auto axis = HandleNegativeAxis(axis_, Y.Shape().NumDimensions());

  size_t N = input_shape.SizeToDimension(axis);
  size_t D = input_shape.SizeFromDimension(axis);

  if (N == 0) {
    return Status::OK();
  }

  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);

  const float* Ydata = Y.template Data<float>();
  const float* dYdata = dY.template Data<float>();
  float* dXdata = dX.template MutableData<float>();

  std::vector<float> eY(nd);
  float* eYdata = eY.data();
  
  // dX_ai = d(log Y_ai) - [sum_j d(log Y_aj)] exp(log Y_ai)
  gsl::copy(gsl::make_span(dYdata, nd), gsl::make_span(dXdata, nd));
  math::Exp<float, CPUMathUtil>(nd, Ydata, eYdata, nullptr);
  for (size_t i = 0; i < N; ++i) {
    float sdY;
    math::Sum<float, CPUMathUtil>(d, dYdata + i*d, &sdY, nullptr, nullptr);
    math::Axpy<float, CPUMathUtil>(d, -sdY, eYdata + i*d, dXdata + i*d, nullptr);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
