// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/softmax.h"
#include "core/providers/cpu/tensor/transpose.h"
#include <vector>
#include <numeric>

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    1,
    10,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    11,
    12,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    13,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    11,
    12,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    13,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// opset-12 and below
template <typename T>
Status Softmax<T>::ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                               concurrency::ThreadPool* thread_pool) const {
  const auto& X_shape = input.Shape();
  const size_t N = onnxruntime::narrow<size_t>(X_shape.SizeToDimension(axis));
  const size_t D = onnxruntime::narrow<size_t>(X_shape.SizeFromDimension(axis));

  return SoftmaxCPU<T>(N, D, input.Data<T>(), output.MutableData<T>(), log_softmax_, thread_pool);
}

// opset-13 and above
template <typename T>
Status Softmax<T>::ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis,
                                      concurrency::ThreadPool* thread_pool, OpKernelContext* ctx) const {
  const auto& X_shape = input.Shape();
  size_t rank = X_shape.NumDimensions();

  bool is_transpose_required = false;
  Tensor transposed_input;
  std::vector<int64_t> transposed_input_dims;
  Tensor intermediate_output;  // output that the softmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  // The "semantic" meaning of axis has changed in opset-13.
  // Please compare: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
  // with https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softmax-11 for detailed explanations
  // To account for the opset-13 behavior, our plan will be to transpose the "axis" dim to the innermost dim
  // and perform softmax and then reverse the transpose. We can skip the transposing aspect if the axis is already
  // the innermost dim
  if (axis != (rank - 1)) {
    is_transpose_required = true;
  }

  if (is_transpose_required) {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

    std::iota(std::begin(permutation), std::end(permutation), 0);

    // swap the innermost dim with the dim corresponding to axis
    permutation[axis] = rank - 1;
    permutation[rank - 1] = axis;

    transposed_input_dims.reserve(rank);
    for (auto e : permutation) {
      transposed_input_dims.push_back(X_shape[e]);
    }

    // Allocate a temporary tensor to hold transposed input
    Tensor temp_input(input.DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, input, temp_input));
    transposed_input = std::move(temp_input);

    // Allocate memory for the intermediate output
    Tensor temp_output(output.DataType(), TensorShape(transposed_input_dims), alloc);
    intermediate_output = std::move(temp_output);
  }

  const size_t N = onnxruntime::narrow<size_t>(is_transpose_required ? TensorShape(transposed_input_dims).SizeToDimension(rank - 1) : X_shape.SizeToDimension(rank - 1));
  const size_t D = onnxruntime::narrow<size_t>(is_transpose_required ? TensorShape(transposed_input_dims).SizeFromDimension(rank - 1) : X_shape.SizeFromDimension(rank - 1));

  ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(N, D,
                                    is_transpose_required ? transposed_input.Data<T>() : input.Data<T>(),
                                    is_transpose_required ? intermediate_output.MutableData<T>() : output.MutableData<T>(),
                                    log_softmax_, thread_pool));

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, intermediate_output, output));
  }

  return Status::OK();
}

// compute method of Softmax
template <typename T>
Status Softmax<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  size_t rank = X_shape.NumDimensions();
  auto* Y = ctx->Output(0, X_shape);

  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }

  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  if (opset_ < 13) {
    return ComputeImpl(*X, *Y, axis, thread_pool);
  } else {
    return ComputeImplOpset13(*X, *Y, axis, thread_pool, ctx);
  }
}

}  // namespace onnxruntime
