// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/det.h"
#include "core/util/math_cpuonly.h"
//TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif

using namespace onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Det,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Det<float>);

template <typename T>
Status Det<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  const auto& X_shape = X->Shape();
  int X_num_dims = static_cast<int>(X_shape.NumDimensions());

  // input validation
  if (X_num_dims < 2) {  // this is getting capture by shape inference code as well
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor should have a rank of at least 2");
  }
  if (X_shape[X_num_dims - 1] != X_shape[X_num_dims - 2]) {  // this is getting capture by shape inference code as well
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Matrix dimensions are not equal. Square matrix is expected");
  }

  const auto* X_data = X->Data<T>();
  int matrix_dim = static_cast<int>(X_shape[X_num_dims - 1]);

  auto get_determinant = [matrix_dim](const T* matrix_ptr) -> T {
    auto one_eigen_mat = ConstEigenMatrixMapRowMajor<T>(matrix_ptr, matrix_dim, matrix_dim);
    return one_eigen_mat.determinant();
  };

  if (X_num_dims == 2) {
    auto* Y = context->Output(0, {});  // as per spec output should be a scalar when input is 2D
    auto* Y_data = Y->template MutableData<T>();
    *Y_data = get_determinant(X_data);
  } else {
    // calculate batch size and output shape
    std::vector<int64_t> output_shape;
    output_shape.reserve(X_num_dims - 2);
    int64_t batch_size = 1;
    for (int i = 0; i < X_num_dims - 2; ++i) {
      batch_size *= X_shape[i];
      output_shape.push_back(X_shape[i]);
    }

    int num_matrix_elems = matrix_dim * matrix_dim;
    auto* Y = context->Output(0, output_shape);
    auto* Y_data = Y->template MutableData<T>();
    for (int b = 0; b < static_cast<int>(batch_size); ++b) {  // can be parallelized if need to
      const T* one_matrix = X_data + (b * num_matrix_elems);
      *Y_data++ = get_determinant(one_matrix);
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
