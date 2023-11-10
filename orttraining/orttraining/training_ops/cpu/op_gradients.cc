// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/op_gradients.h"

#include "core/common/gsl.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include <unsupported/Eigen/SpecialFunctions>

namespace onnxruntime {
namespace contrib {

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

  auto dY_span = dY.template DataAsSpan<T>();
  auto X_span = X.template DataAsSpan<T>();
  auto dX_span = dX.template MutableDataAsSpan<T>();

  EigenVectorArrayMap<float>(dX_span.data(), dX_span.size()) =
      (ConstEigenVectorArrayMap<float>(X_span.data(), X_span.size()) > T(0))
          .select(ConstEigenVectorArrayMap<float>(dY_span.data(), dY_span.size()), T(0));

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

  size_t N = narrow<size_t>(input_shape.SizeToDimension(axis));
  size_t D = narrow<size_t>(input_shape.SizeFromDimension(axis));

  if (N == 0) {
    return Status::OK();
  }

  bool is_transpose_required = opset_ >= 13 && axis != (rank - 1);

  Tensor transposed_dY;
  Tensor transposed_Y;
  TensorShapeVector transposed_input_dims;
  Tensor intermediate_output;  // output that the softmax implementation will write into while using transposed input
  InlinedVector<size_t> permutation(rank);

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
    N = narrow<size_t>(TensorShape(transposed_input_dims).SizeToDimension(rank - 1));
    D = narrow<size_t>(TensorShape(transposed_input_dims).SizeFromDimension(rank - 1));

    // Allocate a temporary tensor to hold transposed input
    auto temp_input0 = Tensor(Y.DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutation, Y, temp_input0));
    transposed_Y = std::move(temp_input0);

    auto temp_input1 = Tensor(Y.DataType(), TensorShape(transposed_input_dims), alloc);
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutation, dY, temp_input1));
    transposed_dY = std::move(temp_input1);

    // Allocate memory for the intermediate output
    intermediate_output = Tensor(dX.DataType(), TensorShape(transposed_input_dims), alloc);
  }

  const int n = gsl::narrow_cast<int>(N);
  const int d = gsl::narrow_cast<int>(D);
  const int nd = gsl::narrow_cast<int>(N * D);
  const float* Ydata = is_transpose_required ? transposed_Y.template Data<T>() : Y.template Data<float>();
  const float* dYdata = is_transpose_required ? transposed_dY.template Data<T>() : dY.template Data<float>();
  float* dXdata = is_transpose_required ? intermediate_output.template MutableData<T>() : dX.template MutableData<float>();

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
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutation, intermediate_output, dX));
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
  EigenVectorArrayMap<float> dx = EigenVectorArrayMap<float>(dX.template MutableData<T>(),
                                                             narrow<Eigen::Index>(dX.Shape().Size()));
  ConstEigenVectorArrayMap<float> y = ConstEigenVectorArrayMap<float>(Y.template Data<T>(),
                                                                      narrow<Eigen::Index>(Y.Shape().Size()));
  ConstEigenVectorArrayMap<float> dy = ConstEigenVectorArrayMap<float>(dY.template Data<T>(),
                                                                       narrow<Eigen::Index>(dY.Shape().Size()));
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
  EigenVectorArrayMap<float> dx = EigenVectorArrayMap<float>(dX.template MutableData<T>(),
                                                             narrow<Eigen::Index>(dX.Shape().Size()));
  ConstEigenVectorArrayMap<float> y = ConstEigenVectorArrayMap<float>(Y.template Data<T>(),
                                                                      narrow<Eigen::Index>(Y.Shape().Size()));
  ConstEigenVectorArrayMap<float> dy = ConstEigenVectorArrayMap<float>(dY.template Data<T>(),
                                                                       narrow<Eigen::Index>(dY.Shape().Size()));
  dx = dy * (1 - y * y);
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(QuickGeluGrad, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        QuickGeluGrad<float>);

template <typename T>
Status QuickGeluGrad<T>::Compute(OpKernelContext* context) const {
  auto& dY = *context->Input<Tensor>(0);
  const T* dY_data = dY.template Data<T>();
  auto& X = *context->Input<Tensor>(1);
  const T* X_data = X.template Data<T>();
  auto& dX = *context->Output(0, dY.Shape());
  T* dX_data = dX.template MutableData<T>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = dY.Shape().Size();
  constexpr int64_t length_per_task = 4096;  // this number comes from FastGelu.
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
  concurrency::ThreadPool::TryBatchParallelFor(
      tp, static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const T* p_dy = dY_data + start;
        const T* p_x = X_data + start;
        T* p_dx = dX_data + start;
        int64_t count = std::min(length_per_task, elem_count - start);
        for (int64_t i = 0; i < count; i++) {
          p_dx[i] = p_x[i] * alpha_;
        }

        MlasComputeLogistic(p_dx, p_dx, narrow<size_t>(count));

        for (int64_t i = 0; i < count; i++) {
          p_dx[i] = p_dy[i] * p_dx[i] * (1.f + alpha_ * p_x[i] * (1.f - p_dx[i]));
        }
      },
      0);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
