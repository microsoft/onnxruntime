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
namespace matmul_sparse {

template <class T>
using SparseMatrixRow = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template <class T>
struct MatMulSparseInfo {
  SparseMatrixRow<T> sparse_initializer_;
  TensorShape shape_;
};

template <class T>
Status PrePack(const Tensor& tensor, const OpKernel::PrepackParam& prepack_param,
               bool transb, int32_t expected_kernel_type,
               std::unique_ptr<MatMulSparseInfo<T>>& sparse_info, bool& is_packed) {
  is_packed = false;
  if (tensor.GetElementType() != expected_kernel_type) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, prepack_param.name + " : wrong data type for the constant initializer");
  }

  const auto& right_shape = tensor.Shape();
  const auto right_num_dims = right_shape.NumDimensions();
  if (right_num_dims > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2");
  }

  if (right_num_dims == 1) {
    transb = false;
  }

  // XXX: We assume simple matrix for testing purpose
  // MxKxN
  int64_t K = 0;
  int64_t N = 0;
  if (right_num_dims == 2) {
    K = right_shape[right_num_dims - 2];
    N = right_shape[right_num_dims - 1];
  } else {
    K = right_shape[0];
    N = 1;
  }

  ConstEigenMatrixMapRowMajor<T> initializer_matrix(tensor.Data<T>(), K, N);
  auto sp_info = onnxruntime::make_unique<MatMulSparseInfo<T>>();

  if (transb) {
    sp_info->sparse_initializer_ = initializer_matrix.transpose().sparseView();
  } else {
    sp_info->sparse_initializer_ = initializer_matrix.sparseView();
  }

  sp_info->sparse_initializer_.makeCompressed();
  sp_info->shape_ = tensor.Shape();
  sparse_info = std::move(sp_info);
  is_packed = true;
  return Status::OK();
}

template <class T>
static Status Compute(OpKernelContext* ctx, const MatMulSparseInfo<T>& sparse_info,
                      bool transa, bool transb) {
  const Tensor* left = ctx->Input<Tensor>(0);
  const auto& left_shape = left->Shape();
  
  const auto& right_shape = sparse_info.shape_;
  const bool trans_a = transa && left->Shape().NumDimensions() != 1;
  const bool trans_b = transb && right_shape.NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_shape, right_shape, trans_a, trans_b));
  if (helper.OutputShape().Size() == 0)
    return Status::OK();

  const auto M = helper.M();
  const auto K = helper.K();
  const auto N = helper.N();

  const auto* left_data = left->Data<T>();
  Tensor* output = ctx->Output(0, helper.OutputShape());
  auto* output_data = output->MutableData<T>();

  const size_t max_len = helper.OutputOffsets().size();
  // auto dense_initializer = sparse_info.sparse_initializer_.toDense();
  for (size_t i = 0; i < max_len; ++i) {
    ConstEigenMatrixMapRowMajor<T> left_matrix(left_data + helper.LeftOffsets()[i], M, K);
    auto output_matrix = EigenMatrixMapRowMajor<T>(output_data + helper.OutputOffsets()[i], M, N);
    if (transa) {
      // output_matrix = (sparse_info.sparse_initializer_ * left_matrix).transpose();
      // output_matrix = left_matrix.transpose() * dense_initializer;
      output_matrix = left_matrix.transpose().sparseView() * sparse_info.sparse_initializer_;
    } else {
      // output_matrix = (sparse_info.sparse_initializer_ * left_matrix.transpose()).transpose();
      // output_matrix = left_matrix * dense_initializer;
      output_matrix = left_matrix.sparseView() * sparse_info.sparse_initializer_;
    }
  }
  return Status::OK();
}
}  // namespace matmul_sparse
}  // namespace onnxruntime
