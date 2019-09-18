// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/det.h"
#include "core/util/math_cpuonly.h"

using namespace onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Det,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Det<float>);

template <typename T>
Status Det::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  const auto& X_shape = X->Shape();
  int X_num_dims = X_shape.size();

  // input validation
  if (X_shape.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Got empty shape for the input tensor");
  }
  if (X_shape.size() < 2) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor should have a rank of at least 2");
  }
  if (X_shape[X_num_dims - 1] != X_shape[X_num_dims - 2]) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Matrix dimensions are not equal. Square matrix is expected");
  }

  const auto* X_data = X->Data();
  int matrix_dim = X_shape[X_num_dims - 1];

  auto get_determinant = [= matrix_dim](const T* matrix_ptr) -> T {
    auto one_eigen_mat = ConstEigenMatrixMapRowMajor<T>(matrix_ptr, matrix_dim, matrix_dim);
    return one_eigen_mat.determinant();
  };

  if (X_num_dims == 2) {
    auto* Y = context->Output(0, {});  // as per spec output should be a scalar when input is 2D
    auto* Y_data = Y->template MutableData<T>();
    *Y_data = get_determinant(X_data);
  } else {
    int batch_size = X_shape[0];
    int num_matrix_elems = matrix_dim * matrix_dim;
    auto* Y = context->Output(0, {batch_size});
    auto* Y_data = Y->template MutableData<T>();
    for (int b = 0; b < batch_size; ++b) {
      const auto* one_matrix = X_data[b * num_matrix_elems];
      *Y_data++ = get_determinant(one_matrix);
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
