// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/hardmax.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/math.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {

template <>
Status Hardmax<float>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  size_t rank = X_shape.NumDimensions();
  Tensor* Y = ctx->Output(0, X_shape);

  // special case when there is a dim value of 0 in the shape.
  if (X_shape.Size() == 0)
    return Status::OK();

  // handle negative and enforce axis is valid
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));

  bool is_transpose_required = false;
  Tensor transposed_input;
  std::vector<int64_t> transposed_input_dims;
  Tensor intermediate_output;  // output that the hardmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  // The "semantic" meaning of axis has changed in opset-13.
  // Please compare: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax
  // with https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Hardmax-11 for detailed explanations
  // To account for the opset-13 behavior, our plan will be to transpose the "axis" dim to the innermost dim
  // and perform softmax and then reverse the transpose. We can skip the transposing aspect if the axis is already
  // the innermost dim
  if (opset_ >= 13 && axis != (rank - 1)) {
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
    Tensor temp_input(X->DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, *X, temp_input));
    transposed_input = std::move(temp_input);

    // Allocate memory for the intermediate output
    Tensor temp_output(Y->DataType(), TensorShape(transposed_input_dims), alloc);
    intermediate_output = std::move(temp_output);
  }

  size_t tmp_N = is_transpose_required ? TensorShape(transposed_input_dims).SizeToDimension(rank - 1) : X_shape.SizeToDimension(axis);
  size_t tmp_D = is_transpose_required ? TensorShape(transposed_input_dims).SizeFromDimension(rank - 1) : X_shape.SizeFromDimension(axis);

  // Math::RowwiseMax expects int N and D.
  if (tmp_N * tmp_D > INT32_MAX || tmp_N > INT32_MAX || tmp_D > INT32_MAX) {
    std::ostringstream ss;
    ss << "Hardmax inputs N, D and N * D must be < " << INT32_MAX << ". N=" << tmp_N << ", D=" << tmp_D;
    std::string msg = ss.str();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, msg);
  }

  const int N = gsl::narrow_cast<int>(tmp_N);
  const int D = gsl::narrow_cast<int>(tmp_D);

  std::vector<float> rowmax_(N);
  float* rowmax_data = rowmax_.data();

  const float* X_data = nullptr;
  float* Y_data = nullptr;

  if (is_transpose_required) {  // use intermediate buffers to compute the hardmax values
    X_data = transposed_input.Data<float>();
    Y_data = intermediate_output.MutableData<float>();
  } else {  // use the node input/output directly
    X_data = X->Data<float>();
    Y_data = Y->MutableData<float>();
  }

  math::RowwiseMax<float, CPUMathUtil>(N, D, X_data, rowmax_data, nullptr);

  // Even if we had to transpose the input, it is safe to go with X_shape.Size() which computes
  // the size of the buffer from the original input's shape as even if we do transpose, the size
  // of the transposed buffer will be the same as the original input's buffer
  math::Set<float, CPUMathUtil>(X_shape.Size(), 0.f, Y_data, &CPUMathUtil::Instance());

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      if (X_data[i * D + j] == rowmax_data[i]) {
        Y_data[i * D + j] = 1;
        break;
      }
    }
  }

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, intermediate_output, *Y));
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Hardmax,
    1,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Hardmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Hardmax,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Hardmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_KERNEL(
    Hardmax,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Hardmax<float>);

}  // namespace onnxruntime
