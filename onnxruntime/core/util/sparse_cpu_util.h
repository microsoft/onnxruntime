// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <core/framework/op_kernel.h>
#include <Eigen/SparseCore>
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace sparse_util {

template <class T>
using SparseMatrixRow = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template <class T>
Status ConvertDenseToEigenSparse(const Tensor& tensor,
                                 bool transpose, int32_t expected_kernel_type,
                                 SparseMatrixRow<T>& result) {
  if (tensor.GetElementType() != expected_kernel_type) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, " Input data type: ", tensor.GetElementType(),
                           " does not match expected: ", expected_kernel_type);
  }

  const auto& shape = tensor.Shape();
  const auto num_dims = shape.NumDimensions();
  if (num_dims > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2");
  }

  // XXX: We assume simple matrix for testing purpose
  // MxKxN
  int64_t K = 0;
  int64_t N = 0;
  if (num_dims == 2) {
    K = shape[num_dims - 2];
    N = shape[num_dims - 1];
  } else {
    K = shape[0];
    N = 1;
  }

  ConstEigenMatrixMapRowMajor<T> initializer_matrix(tensor.Data<T>(), K, N);

  if (transpose) {
    result = initializer_matrix.transpose().sparseView();
  } else {
    result = initializer_matrix.sparseView();
  }

  return Status::OK();
}

// XXX: Parallel batching
// XXX: Speedup algorithm and parallelize it
template <class T>
Status ComputeWithSparseWeight(const T* left_data,
                               T* output_data,
                               const MatMulComputeHelper& helper,
                               const sparse_util::SparseMatrixRow<T>& sparse_mat,
                               bool transa) {
  const auto M = helper.M();
  const auto K = helper.K();
  const auto N = helper.N();

  const size_t batches = helper.OutputOffsets().size();

  for (size_t i = 0; i < batches; ++i) {
    ConstEigenMatrixMapRowMajor<T> left_matrix(left_data + helper.LeftOffsets()[i], M, K);
    auto output_matrix = EigenMatrixMapRowMajor<T>(output_data + helper.OutputOffsets()[i], M, N);
    // Tried:
    // 1) multiply sparse initializer to dense and then transpose
    // 2) convert sparse initializer to dense and then multiply dense to dense
    // 3) Fastest convert dense to sparse multiply by sparse_initialzier
    if (transa) {
      // output_matrix = (sparse_info.sparse_initializer_ * left_matrix).transpose();
      // output_matrix = left_matrix.transpose() * dense_initializer;
      output_matrix = left_matrix.transpose().sparseView() * sparse_mat;
    } else {
      // output_matrix = (sparse_info.sparse_initializer_ * left_matrix.transpose()).transpose();
      // output_matrix = left_matrix * dense_initializer;
      output_matrix = left_matrix.sparseView() * sparse_mat;
    }
  }
  return Status::OK();
}

Status ComputeWithSparseWeight(const float* left_data,
                               float* output_data,
                               const MatMulComputeHelper& helper,
                               float alpha,
                               const sparse_util::SparseMatrixRow<float>& sparse_mat,
                               bool transa);

// TODO: Make sure we can convert from CSR buffers to Eigen Sparse Matrix

}  // namespace sparse_util
}  // namespace onnxruntime
