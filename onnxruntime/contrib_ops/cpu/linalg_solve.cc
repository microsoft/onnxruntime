// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "contrib_ops/cpu/linalg_solve.h"
#include "core/framework/framework_common.h"
#include "core/framework/tensorprotoutils.h"
#include <Eigen/Dense>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
  LinalgSolve,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
  LinalgSolve<float>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    LinalgSolve,
    1,
    double,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    LinalgSolve<double>);

template <typename T>
void solve(const T* a_data, const T* b_data, T* x_data, bool left, int64_t n, int64_t k) {
  const Eigen::StorageOptions option = Eigen::RowMajor;
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> b_matrix(b_data, narrow<size_t>(n), narrow<size_t>(k));
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> x_matrix(x_data, narrow<size_t>(n), narrow<size_t>(k));

  if (left) {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> a_matrix(a_data, narrow<size_t>(n), narrow<size_t>(n));
    Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(a_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x_matrix = svd.solve(b_matrix);
  } else {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> a_matrix(a_data, narrow<size_t>(k), narrow<size_t>(k));
    auto a_matrix_transposed = a_matrix.transpose();
    auto b_matrix_transposed = b_matrix.transpose();
    Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(a_matrix_transposed, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option> x_matrix_transposed_result = svd.solve(b_matrix_transposed);
    x_matrix = x_matrix_transposed_result.transpose();
  }
}

#pragma warning(disable : 4189)
template <typename T>
Status LinalgSolve<T>::Compute(OpKernelContext* context) const {
  Status status = Status::OK();
  const Tensor* A = context->Input<Tensor>(0);
  const TensorShape& a_shape = A->Shape();
  assert(a_shape.NumDimensions() == 2 || a_shape.NumDimensions() == 3);
  bool has_batch = a_shape.NumDimensions() == 3;
  const Tensor* B = context->Input<Tensor>(1);
  const TensorShape& b_shape = B->Shape();

  int64_t batch = has_batch ? a_shape[0] : 1, n = 1, k = 1;
  bool b_as_a_vector = b_shape.NumDimensions() == 1;
  bool broadcast;
  int64_t n_or_k = a_shape[1];
  if (left_) {
    n = n_or_k;
  } else {
    k = n_or_k;
  }

  if (has_batch) {
    ORT_ENFORCE(a_shape[1] == a_shape[2], "A should be square matrix: ", a_shape);
    if (b_shape.NumDimensions() == 1) {
      b_as_a_vector = true;
      broadcast = true;
    } else if (b_shape.NumDimensions() == 2) {
      if (b_shape[0] == a_shape[0] && b_shape[1] == a_shape[1]) {  // A has shape (*, n/k, n/k) and B has shape(*, n/k)
        b_as_a_vector = true;
        broadcast = false;
      } else if (left_ && b_shape[0] == a_shape[1]) {  // A has shape (*, n, n) and B has shape (n, k)
        b_as_a_vector = false;
        broadcast = true;
        k = b_shape[1];
      } else if (!left_ && b_shape[1] == a_shape[1]) {  // A has shape (*, k, k) and B has shape (n, k)
        b_as_a_vector = false;
        broadcast = true;
        n = b_shape[0];
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "B shape does not mach A shape.", b_shape, a_shape);
      }
    } else { // b_shape.NumDimensions() == 3
      ORT_ENFORCE(b_shape[0] == a_shape[0], "A and B shall have the same batch size");
      b_as_a_vector = false;
      broadcast = false;
      if (left_) { // A: (*, n, n), B: (*, n, k)
        ORT_ENFORCE(b_shape[1] == a_shape[1], "A and B shall have matching size at dim 1: ", b_shape[1], "vs", a_shape[1]);
        k = b_shape[2];
      } else {  // A: (*, k, k), B: (*, n, k)
        ORT_ENFORCE(b_shape[2] == a_shape[2], "A and B shall have matching size at dim 2: ", b_shape[2], "vs", a_shape[2]);
        n = b_shape[1];
      }
    }
  } else { // !has_batch
    ORT_ENFORCE(a_shape[0] == a_shape[1], "A should be square matrix: ", a_shape);
    broadcast = false;
    if (b_shape.NumDimensions() == 1) { // A: (n/k. n/k), B: (n/k,)
      ORT_ENFORCE(b_shape[0] == a_shape[0], "A and B shall have matching size at dim 2: ", b_shape[2], "vs", a_shape[2]);
      b_as_a_vector = true;
    } else if (b_shape.NumDimensions() == 2) { // A: (n/k. n/k), B: (n, k)
      b_as_a_vector = false;
      if (left_) { // A: (n, n), B: (n, k)
        k = b_shape[1];
      } else { // A: (k, k), B: (n, k)
        n = b_shape[0];
      }
    }
  }

  std::vector<int64_t> x_dims;
  if (has_batch) {
    x_dims.push_back(batch);
  }
  if (b_as_a_vector) {
    if (left_) {
      x_dims.push_back(n);
    } else {
      x_dims.push_back(k);
    }      
  } else {
    x_dims.push_back(n);
    x_dims.push_back(k);
  }
  TensorShape x_shape(x_dims);
  Tensor* X = context->Output(0, x_shape);

  if (batch == 1) {
    const T* a_data = A->Data<T>();
    const T* b_data = B->Data<T>();
    T* x_data = X->MutableData<T>();
    solve(a_data, b_data, x_data, left_, n, k);
  } else {
    int64_t a_single_batch_size = a_shape.SizeFromDimension(a_shape.NumDimensions() - 2);
    int64_t b_single_batch_size = broadcast ? 0 : b_shape.SizeFromDimension(1);
    int64_t x_single_batch_size = x_shape.SizeFromDimension(1);
    std::function<void(ptrdiff_t)> fn = [&](ptrdiff_t batch_num) {
      const T* a_data = A->Data<T>() + batch_num * a_single_batch_size;
      const T* b_data = B->Data<T>() + (broadcast ? 0 : batch_num * b_single_batch_size);
      T* x_data = X->MutableData<T>() + batch_num * x_single_batch_size;
      solve(a_data, b_data, x_data, left_, n, k);
    };
    concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), narrow<size_t>(batch), std::move(fn), 0);
  }

  return status;
}
}  // namespace contrib
}  // namespace onnxruntime
