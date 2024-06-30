// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "contrib_ops/cpu/linalg_cholesky.h"
#include "core/framework/framework_common.h"
#include "core/framework/tensorprotoutils.h"
#include <Eigen/Dense>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
  LinalgCholesky,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LinalgCholesky<float>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    LinalgCholesky,
    1,
    double,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    LinalgCholesky<double>);

template <typename T>
Status cholesky(const T* a_data, T* l_data, int64_t n, bool upper) {
  const Eigen::StorageOptions option = Eigen::RowMajor;
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> a_matrix(a_data, narrow<size_t>(n), narrow<size_t>(n));
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> l_matrix(l_data, narrow<size_t>(n), narrow<size_t>(n));
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, option>> lltOfA(a_matrix);
  if (lltOfA.info() == Eigen::Success) {
    if (upper) {
      l_matrix = lltOfA.matrixU();
    } else {
      l_matrix = lltOfA.matrixL();
    }
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input matrix A is not decomposable with Cholesky.");
  }
}

#pragma warning(disable : 4189)
template <typename T>
Status LinalgCholesky<T>::Compute(OpKernelContext* context) const {
  const Tensor* A = context->Input<Tensor>(0);
  const TensorShape& a_shape = A->Shape();
  int64_t a_rank = a_shape.NumDimensions();
  assert(a_rank == 2 || a_rank == 3);
  int64_t batch = a_rank == 2 ? 1 : a_shape[0];

  assert(a_shape[a_rank - 1] == a_shape[a_rank - 2]);

  Tensor* L = context->Output(0, a_shape);

  if (batch == 1) {
    return cholesky(A->Data<T>(), L->MutableData<T>(), a_shape[a_rank - 1], upper_);
  } else {
    std::mutex status_mutex;
    Status summary_status = Status::OK();
    int64_t single_batch_size = a_shape.SizeFromDimension(a_rank - 2);
    std::function<void(ptrdiff_t)> fn = [&](ptrdiff_t batch_num) {
      const T* a_data = A->Data<T>() + batch_num * single_batch_size;
      T* l_data = L->MutableData<T>() + batch_num * single_batch_size;
      Status status = cholesky(a_data, l_data, a_shape[a_rank - 1], upper_);
      if (!status.IsOK()) {
        // let the main function return any unsuccessful status
        std::lock_guard<std::mutex> lock(status_mutex);
        summary_status = status;
      }
    };
    concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), narrow<size_t>(batch), std::move(fn), 0);
    return summary_status;
  }
}
}  // namespace contrib
}  // namespace onnxruntime
