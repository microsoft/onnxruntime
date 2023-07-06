// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

class GroupedGemmHelper {
 public:
  GroupedGemmHelper(const TensorShape& left, bool trans_left, const TensorShape& right, bool trans_right, const TensorShape& bias_shape, const TensorShape& msizes_shape) {
    // dimension check
    ORT_ENFORCE(left.NumDimensions() == 2);
    ORT_ENFORCE(right.NumDimensions() == 2);
    ORT_ENFORCE(msizes_shape.NumDimensions() == 2);

    for (size_t i = 0; i != left.NumDimensions(); ++i) {
      ORT_ENFORCE(left[i] >= 0);
      ORT_ENFORCE(left[i] <= std::numeric_limits<ptrdiff_t>::max());
    }

    for (size_t i = 0; i != right.NumDimensions(); ++i) {
      ORT_ENFORCE(right[i] >= 0);
      ORT_ENFORCE(right[i] <= std::numeric_limits<ptrdiff_t>::max());
    }

    if (trans_left) {
      M_ = static_cast<ptrdiff_t>(left[1]);
      K_ = static_cast<ptrdiff_t>(left[0]);
    } else {
      M_ = static_cast<ptrdiff_t>(left[0]);
      K_ = static_cast<ptrdiff_t>(left[1];)
    }

    int k_dim;
    if (trans_right) {
      N_ = static_cast<ptrdiff_t>(right[0]);
      k_dim = 1;
    } else {
      N_ = static_cast<ptrdiff_t>(right[1]);
      k_dim = 0;
    }

    num_matrix_ = msizes_shape[0];

    if (right[k_dim] != K_)
      status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                "GEMM: Dimension mismatch, W: ",
                                right.ToString(),
                                " K: " + std::to_string(K_),
                                " N:" + std::to_string(N_));
    // check bias shape
    if (!IsValidBroadcast(bias_shape, msizes_shape)) {
      status_ = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "GroupedGemm: Invalid bias shape for broadcast");
    }

    // it is possible the input is empty tensor, for example the output of roipool in fast rcnn.
    ORT_ENFORCE(M_ >= 0 && K_ > 0 && N_ >= 0);
  }

  ptrdiff_t M() const { return M_; }
  ptrdiff_t N() const { return N_; }
  ptrdiff_t K() const { return K_; }
  ptrdiff_t num_matrix const { return num_matrix_; }
  Status State() const { return status_; }

 private:
  static bool IsValidBroadcast(const TensorShape& bias_shape, const TensorShape& msizes_shape) {
    if (bias_shape.NumDimensions() != 2) {
      return false;
    }

    // valid shape is (M, N) or (m, N) where m is number of elements in msizes
    return (bias_shape[0] == M_ && bias_shape[1] == N_) ||
	    (bias_shape[0] == msizes_shape[0] && bias_shape[1] == N);
  }

  GroupedGemmHelper() = default;
  ptrdiff_t M_;
  ptrdiff_t K_;
  ptrdiff_t N_;
  ptrdiff_t num_matrix_;
  Status status_;
};

template<typename T>
class GroupedGemm final : public RocmKernel {
 public:
  GroupedGemm(const OpKernelInfo& op_kernel_info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK()); 
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;
 private:
  float alpha_;
  float beta_;
  bool trans_A_;
  bool trans_B_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
