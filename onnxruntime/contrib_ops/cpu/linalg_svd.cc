// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "contrib_ops/cpu/linalg_svd.h"
#include "core/framework/framework_common.h"
#include "core/framework/tensorprotoutils.h"
#include <Eigen/Dense>
#include <Eigen/SVD>


using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    LinalgSVD,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LinalgSVD<float>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    LinalgSVD,
    1,
    double,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    LinalgSVD<double>);

template <typename T>
void compute_svd(const T* a_data, T* u_data, T* s_data, T* v_data, int64_t m, int64_t n, int64_t k, bool full_matrices) {
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 1>> a_map(a_data, narrow<size_t>(m), narrow<size_t>(n));

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u_map(u_data, m, full_matrices ? m : k);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> s_map(s_data, k, 1);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> v_map(v_data, narrow<size_t>(full_matrices ? n : k), narrow<size_t>(n));

  // Compute the SVD
  unsigned int computationOptions = full_matrices ? (Eigen::ComputeFullU | Eigen::ComputeFullV) : (Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> svd(a_map, computationOptions);

  // Assign the computed matrices to the pre-allocated memory
  u_map = svd.matrixU();
  s_map = svd.singularValues();
  v_map = svd.matrixV().transpose();
}

template <typename T>
Status LinalgSVD<T>::Compute(OpKernelContext* context) const {
  Status status = Status::OK();
  const Tensor* A = context->Input<Tensor>(0);
  const TensorShape& a_shape = A->Shape();
  int64_t dimensions = A->Shape().NumDimensions();
  ORT_ENFORCE(dimensions == 2 || dimensions == 3, "data must be 2D or 3D tensor");

  int64_t batch = 1, m, n, k;
  m = a_shape[dimensions - 2];
  n = a_shape[dimensions - 1];
  k = std::min(m, n);

  TensorShape u_shape, s_shape, v_shape;
  if (dimensions == 3) {
    batch = a_shape[0];
    u_shape = {batch, m, full_matrices_ ? m : k};
    s_shape = {batch, k};
    v_shape = {batch, full_matrices_ ? n : k, n};
  } else {
    u_shape = {m, full_matrices_ ? m : k};
    s_shape = {k};
    v_shape = {full_matrices_ ? n : k, n};
  }
  Tensor* U = context->Output(0, u_shape);
  Tensor* S = context->Output(1, s_shape);
  Tensor* V = context->Output(2, v_shape);

  int64_t a_single_batch_size = A->Shape().SizeFromDimension(dimensions - 2);
  int64_t u_single_batch_size = U->Shape().SizeFromDimension(dimensions - 2);
  int64_t s_single_batch_size = S->Shape().SizeFromDimension(S->Shape().NumDimensions() - 1);
  int64_t v_single_batch_size = V->Shape().SizeFromDimension(dimensions - 2);

  std::function<void(ptrdiff_t)> fn = [&](ptrdiff_t batch_num) {
    const T* a_data = A->template Data<T>() + batch_num * a_single_batch_size;
    T* u_data = U->template MutableData<T>() + batch_num * u_single_batch_size;
    T* s_data = S->template MutableData<T>() + batch_num * s_single_batch_size;
    T* v_data = V->template MutableData<T>() + batch_num * v_single_batch_size;

    compute_svd(a_data, u_data, s_data, v_data, m, n, k, full_matrices_);
  };

  concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), narrow<size_t>(batch), std::move(fn), 0);
  return status;
}
}  // namespace contrib
}  // namespace onnxruntime
