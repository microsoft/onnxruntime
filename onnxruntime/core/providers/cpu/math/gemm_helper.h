// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor_shape.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

class GemmHelper {
 public:
  static Status Create(const TensorShape& left, bool trans_left, const TensorShape& right, bool trans_right, const TensorShape& bias,
                       std::unique_ptr<GemmHelper>& out) {
    GSL_SUPPRESS(r .11)
    out = std::unique_ptr<GemmHelper>(new GemmHelper());
    // dimension check
    if (left.NumDimensions() != 2 && left.NumDimensions() != 1) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Gemm: left matrix must be 1D or 2D");
    }
    if (right.NumDimensions() != 2) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Gemm: right matrix must be 2D");
    }

    if (trans_left) {
      out->M_ = left.NumDimensions() == 2 ? narrow<ptrdiff_t>(left[1]) : narrow<ptrdiff_t>(left[0]);
      out->K_ = left.NumDimensions() == 2 ? narrow<ptrdiff_t>(left[0]) : 1;
    } else {
      out->M_ = left.NumDimensions() == 2 ? narrow<ptrdiff_t>(left[0]) : 1;
      out->K_ = left.NumDimensions() == 2 ? narrow<ptrdiff_t>(left[1])
                                          : narrow<ptrdiff_t>(left[0]);
    }

    int k_dim;
    if (trans_right) {
      out->N_ = narrow<ptrdiff_t>(right[0]);
      k_dim = 1;
    } else {
      out->N_ = narrow<ptrdiff_t>(right[1]);
      k_dim = 0;
    }

    if (right[k_dim] != out->K_)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GEMM: Dimension mismatch, W: ",
                             right.ToString(),
                             " K: " + std::to_string(out->K_),
                             " N:" + std::to_string(out->N_));

    if (!IsValidBroadcast(bias, out->M_, out->N_))
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Gemm: Invalid bias shape for broadcast");

    // it is possible the input is empty tensor, for example the output of roipool in fast rcnn.
    if (out->M_ < 0 || out->K_ < 0 || out->N_ < 0) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Dims cannot be negative");
    }
    return Status::OK();
  }

  ptrdiff_t M() const { return M_; }
  ptrdiff_t N() const { return N_; }
  ptrdiff_t K() const { return K_; }

 private:
  static bool IsValidBroadcast(const TensorShape& bias_shape, ptrdiff_t M, ptrdiff_t N) {
    // valid shapes are (,) , (1, N) , (M, 1) , (M, N)
    if (bias_shape.NumDimensions() > 2)
      return false;
    // shape is (1,) or (1, 1), or (,)
    if (bias_shape.Size() == 1)
      return true;
    // valid bias_shape (s) are (N,) or (1, N) or (M, 1) or (M, N),
    // In last case no broadcasting needed, so don't fail it
    return ((bias_shape.NumDimensions() == 1 && bias_shape[0] == N) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == M && (bias_shape[1] == 1 || bias_shape[1] == N)) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == 1 && bias_shape[1] == N));
  }

 private:
  GemmHelper() = default;
  ptrdiff_t M_;
  ptrdiff_t K_;
  ptrdiff_t N_;
};

template <typename T>
void GemmBroadcastBias(ptrdiff_t M, ptrdiff_t N, T beta,
                       _In_opt_ const T* c_data, _In_opt_ const TensorShape* c_shape,
                       _Out_writes_(M* N) T* y_data) {
  // Broadcast the bias as needed if bias is given
  if (beta != 0 && c_data != nullptr) {
    ORT_ENFORCE(c_shape != nullptr, "c_shape is required if c_data is provided");
    auto output_mat = EigenMatrixMapRowMajor<T>(y_data, M, N);
    if (c_shape->Size() == 1) {
      // C is (), (1,) or (1, 1), set the scalar
      output_mat.setConstant(*c_data);
    } else if (c_shape->NumDimensions() == 1 || (*c_shape)[0] == 1) {
      // C is (N,) or (1, N)
      output_mat.rowwise() = ConstEigenVectorMap<T>(c_data, N).transpose();
    } else if ((*c_shape)[1] == 1) {
      // C is (M, 1)
      output_mat.colwise() = ConstEigenVectorMap<T>(c_data, M);
    } else {
      // C is (M, N), no broadcast needed.
      output_mat = ConstEigenMatrixMapRowMajor<T>(c_data, M, N);
    }
  }
}

}  // namespace onnxruntime
