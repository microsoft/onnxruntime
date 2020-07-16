// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "triu.h"
#include <functional>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Triu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<float, double>()),
    Triu);

template <typename T>
struct Triu::ComputeImpl {

  void get_triu(const T* X_data, T* Y_data, int64_t matrix_h, int64_t matrix_w, int64_t k_val) const {
    auto input_mat = ConstEigenMatrixMapRowMajor<T>(
	      X_data,
	      matrix_h,
	      matrix_w);
    auto output_mat = EigenMatrixMapRowMajor<T>(
              Y_data,
              matrix_h,
              matrix_w);

    output_mat = input_mat;
    for (int64_t i = -1 * matrix_h; i < k_val; i++){
      output_mat.diagonal(i).array() = static_cast<T>(0);
    }
  }

  void operator()(const Tensor* X, Tensor* Y, int64_t k_val) const {
    const auto& X_shape = X->Shape();
    int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());

    const auto* X_data = X->Data<T>();
    int64_t matrix_h = static_cast<int64_t>(X_shape[X_num_dims - 2]);
    int64_t matrix_w = static_cast<int64_t>(X_shape[X_num_dims - 1]);

    if (X_num_dims == 2) {
      auto* Y_data = Y->template MutableData<T>();
      get_triu(X_data, Y_data, matrix_h, matrix_w, k_val);
    } else {
      // calculate batch size and output shape
      int64_t batch_size = 1;
      for (int64_t i = 0; i < X_num_dims - 2; ++i) {
        batch_size *= X_shape[i];
      }

      int64_t num_matrix_elems = matrix_h * matrix_w;
      auto* Y_data = Y->template MutableData<T>();
      for (int64_t b = 0; b < batch_size; b++) {  // can be parallelized if need to
        auto X_batch_data = X_data + (b * num_matrix_elems);
        auto Y_batch_data = Y_data + (b * num_matrix_elems);
        get_triu(X_batch_data, Y_batch_data, matrix_h, matrix_w, k_val);
      }
    }
  }
};

Status Triu::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* k = ctx->Input<Tensor>(1);

  int64_t k_val = 0;
  if (k) {
    ORT_ENFORCE(IsScalarOr1ElementVector(k), "k should be a 1-D or 0-D tensor.");
    k_val = *(k->template Data<int64_t>());
  }

  const auto& X_shape = X->Shape();
  auto* Y = ctx->Output(0, X->Shape());

  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());
  // input validation
  if (X_num_dims < 2) {  // this is getting capture by shape inference code as well
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor should have a rank of at least 2");
  }

  utils::MLTypeCallDispatcher<ComputeImpl, float, double> t_disp(ctx->Input<Tensor>(0)->GetElementType());
  t_disp.Invoke(X, Y, k_val);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
