// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "trilu.h"
#include <functional>

using namespace onnxruntime::common;

namespace onnxruntime {
#ifndef DISABLE_CONTRIB_OPS
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Trilu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double, int64_t>()),
    Trilu);
} // namespace contrib
#endif

ONNX_OPERATOR_KERNEL_EX(
    Trilu,
    kOnnxDomain,
    14,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double, int64_t>()),
    Trilu);

template <typename T>
static Status TriluImpl(const Tensor* X, Tensor* Y, int64_t k_val, bool up) {
  const auto& X_shape = X->Shape();
  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());

  const auto* X_data = reinterpret_cast<const T*>(X->DataRaw());
  int64_t matrix_h = static_cast<int64_t>(X_shape[X_num_dims - 2]);
  int64_t matrix_w = static_cast<int64_t>(X_shape[X_num_dims - 1]);

  int64_t batch_size = 1;
  for (int64_t i = 0; i < X_num_dims - 2; ++i) {
    batch_size *= X_shape[i];
  }

  int64_t num_matrix_elems = matrix_h * matrix_w;
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());
  for (int64_t b = 0; b < batch_size; b++) {  // can be parallelized if need to
    auto X_batch_data = X_data + (b * num_matrix_elems);
    auto Y_batch_data = Y_data + (b * num_matrix_elems);

    auto input_mat = ConstEigenMatrixMapRowMajor<T>(X_batch_data, matrix_h, matrix_w);
    auto output_mat = EigenMatrixMapRowMajor<T>(Y_batch_data, matrix_h, matrix_w);

    if (X_batch_data != Y_batch_data) {
      output_mat = input_mat;
    }

    if (up) {
      int64_t start_i = k_val > 0 ? 0 : 1 - k_val;
      for (int64_t i = start_i; i < matrix_h; i++) {
        for (int64_t j = 0; j < i + k_val && j < matrix_w; j++) {
          output_mat(i, j) = 0;
        }
      }
    } else {
      int64_t end_i = std::min(matrix_h, matrix_w - k_val);
      for (int64_t i = 0; i < end_i; i++) {
        for (int64_t j = std::max(static_cast<int64_t>(0), i + k_val + 1); j < matrix_w; j++) {
          output_mat(i, j) = 0;
        }
      }
    }
  }
  return Status::OK();
}

Status Trilu::Compute(OpKernelContext* ctx) const {
  Status status;
  const auto* X = ctx->Input<Tensor>(0);
  const auto* k = ctx->Input<Tensor>(1);

  bool up = upper_;
  int64_t k_val = 0;
  if (k) {
    ORT_ENFORCE(IsScalarOr1ElementVector(k), "k should be a 1-D or 0-D tensor.");
    k_val = *(k->template Data<int64_t>());
  }

  const auto& X_shape = X->Shape();
  auto* Y = ctx->Output(0, X_shape);

  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());
  // input validation
  if (X_num_dims < 2) {  // this is getting capture by shape inference code as well
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor should have a rank of at least 2");
  }

  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      status = TriluImpl<float>(X, Y, k_val, up);
      break;
    case sizeof(double):
      status = TriluImpl<double>(X, Y, k_val, up);
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

}  // namespace onnxruntime
