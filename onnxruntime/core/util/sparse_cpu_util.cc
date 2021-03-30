// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sparse_cpu_util.h"

namespace onnxruntime {
namespace sparse_util {
Status ComputeWithSparseWeight(const float* left_data,
                               float* output_data,
                               const MatMulComputeHelper& helper,
                               float alpha,
                               const sparse_util::SparseMatrixRow<float>& sparse_mat,
                               bool transa) {
  const auto M = helper.M();
  const auto K = helper.K();
  const auto N = helper.N();

  const size_t batches = helper.OutputOffsets().size();

  for (size_t i = 0; i < batches; ++i) {
    ConstEigenMatrixMapRowMajor<float> left_matrix(left_data + helper.LeftOffsets()[i], M, K);
    auto output_matrix = EigenMatrixMapRowMajor<float>(output_data + helper.OutputOffsets()[i], M, N);
    // Tried:
    // 1) multiply sparse initializer to dense and then transpose
    // 2) convert sparse initializer to dense and then multiply dense to dense
    // 3) Fastest convert dense to sparse multiply by sparse_initialzier
    if (transa) {
      // output_matrix = (sparse_info.sparse_initializer_ * left_matrix).transpose();
      // output_matrix = left_matrix.transpose() * dense_initializer;
      output_matrix = left_matrix.transpose().sparseView() * alpha * sparse_mat;
    } else {
      // output_matrix = (sparse_info.sparse_initializer_ * left_matrix.transpose()).transpose();
      // output_matrix = left_matrix * dense_initializer;
      output_matrix = left_matrix.sparseView() * alpha * sparse_mat;
    }
  }
  return Status::OK();
}

}
}
