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
#include "core/providers/cpu/tensor/transpose.h"
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

ONNX_OPERATOR_KERNEL_EX(
    ReluGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
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

ONNX_OPERATOR_KERNEL_EX(
    SoftmaxGrad_13,
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

  size_t rank = input_shape.NumDimensions();
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));

  size_t N = input_shape.SizeToDimension(axis);
  size_t D = input_shape.SizeFromDimension(axis);

  if (N == 0) {
    return Status::OK();
  }

  bool is_transpose_required = opset_ >= 13 && axis != (rank - 1);

  std::unique_ptr<Tensor> transposed_dY;
  std::unique_ptr<Tensor> transposed_Y;
  std::vector<int64_t> transposed_input_dims;
  std::unique_ptr<Tensor> intermediate_output;  // output that the softmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  if (is_transpose_required) {
    AllocatorPtr alloc;
    auto status = context->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

    std::iota(std::begin(permutation), std::end(permutation), 0);

    // swap the innermost dim with the dim corresponding to axis
    permutation[axis] = rank - 1;
    permutation[rank - 1] = axis;

    transposed_input_dims.reserve(rank);
    for (auto e : permutation) {
      transposed_input_dims.push_back(input_shape[e]);
    }
    N = TensorShape(transposed_input_dims).SizeToDimension(rank - 1);
    D = TensorShape(transposed_input_dims).SizeFromDimension(rank - 1);

    // Allocate a temporary tensor to hold transposed input
    auto temp_input0 = Tensor::Create(Y.DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutation, Y, *temp_input0));
    transposed_Y = std::move(temp_input0);

    auto temp_input1 = Tensor::Create(Y.DataType(), TensorShape(transposed_input_dims), alloc);
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutation, dY, *temp_input1));
    transposed_dY = std::move(temp_input1);

    // Allocate memory for the intermediate output
    intermediate_output = Tensor::Create(dX.DataType(), TensorShape(transposed_input_dims), alloc);
  }

  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);
  const float* Ydata = is_transpose_required ? transposed_Y->template Data<T>() : Y.template Data<float>();
  const float* dYdata = is_transpose_required ? transposed_dY->template Data<T>() : dY.template Data<float>();
  float* dXdata = is_transpose_required ? intermediate_output->template MutableData<T>() : dX.template MutableData<float>();

  gsl::copy(gsl::make_span(dYdata, nd), gsl::make_span(dXdata, nd));
  if (is_logsoftmaxgrad_) {
    std::vector<float> eY(nd);
    float* eYdata = eY.data();

    // dX_ai = d(log Y_ai) - [sum_j d(log Y_aj)] exp(log Y_ai)
    gsl::copy(gsl::make_span(dYdata, nd), gsl::make_span(dXdata, nd));
    math::Exp<float, CPUMathUtil>(nd, Ydata, eYdata, nullptr);
    for (size_t i = 0; i < N; ++i) {
      float sdY;
      math::Sum<float, CPUMathUtil>(d, dYdata + i * d, &sdY, nullptr);
      math::Axpy<float, CPUMathUtil>(d, -sdY, eYdata + i * d, dXdata + i * d, nullptr);
    }
  } else {
    std::vector<float> scale_(N);
    std::vector<float> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0
    float* scaledata = scale_.data();
    for (size_t i = 0; i < N; ++i) {
      math::Dot<float, CPUMathUtil>(d, Ydata + i * d, dYdata + i * d,
                                    scaledata + i, nullptr);
    }

    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    math::Gemm<float>(CblasNoTrans, CblasNoTrans, n, d, 1, -1,
                      scaledata, sum_multiplier_.data(), 1,
                      dXdata, tp);

    math::Mul<float, CPUMathUtil>(gsl::narrow_cast<int>(Y.Shape().Size()), dXdata, Ydata, dXdata, nullptr);
  }
  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutation, *intermediate_output, dX));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    LogSoftmaxGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxGrad<float>);

ONNX_OPERATOR_KERNEL_EX(
    LogSoftmaxGrad_13,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SoftmaxGrad<float>);

ONNX_OPERATOR_KERNEL_EX(
    SigmoidGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SigmoidGrad<float>);

template <typename T>
Status SigmoidGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, dY.Shape());
  EigenVectorArrayMap<float> dx = EigenVectorArrayMap<float>(dX.template MutableData<T>(), dX.Shape().Size());
  ConstEigenVectorArrayMap<float> y = ConstEigenVectorArrayMap<float>(Y.template Data<T>(), Y.Shape().Size());
  ConstEigenVectorArrayMap<float> dy = ConstEigenVectorArrayMap<float>(dY.template Data<T>(), dY.Shape().Size());
  dx = dy * y * (1 - y);
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    TanhGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    TanhGrad<float>);

template <typename T>
Status TanhGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  auto& Y = *context->Input<Tensor>(1);
  auto& dX = *context->Output(0, dY.Shape());
  EigenVectorArrayMap<float> dx = EigenVectorArrayMap<float>(dX.template MutableData<T>(), dX.Shape().Size());
  ConstEigenVectorArrayMap<float> y = ConstEigenVectorArrayMap<float>(Y.template Data<T>(), Y.Shape().Size());
  ConstEigenVectorArrayMap<float> dy = ConstEigenVectorArrayMap<float>(dY.template Data<T>(), dY.Shape().Size());
  dx = dy * (1 - y * y);
  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
