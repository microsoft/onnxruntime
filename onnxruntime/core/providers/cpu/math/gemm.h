// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "gemm_helper.h"

namespace onnxruntime {

template <typename T_X,
          typename T_W,
          typename T_B,
          typename T_Y>
class Gemm : public OpKernel {
 public:
  Gemm(const OpKernelInfo& info) : OpKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);
    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, B->Shape());

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    int64_t K = helper.K();
    auto Y = context->Output(0, TensorShape({M, N}));
    // if input is emtpy tensor, return directly as nothing need to be calculated.
    if (M == 0 || N == 0)
      return Status::OK();
    T_Y* y_data = Y->template MutableData<T_Y>();

    //bias
    // Todo: we might should move this part into math::gemm to let eigen
    // have better chance to further optimize it.
    if (beta_ != 0) {
      auto output_mat = EigenMatrixMapRowMajor<T_Y>(
          Y->template MutableData<T_Y>(),
          M,
          N);
      output_mat.setZero();

      auto& b_shape = B->Shape();
      // if B is (), (1,) or (1, 1), add the scalar
      if (b_shape.Size() == 1) {
        output_mat.array() += *(B->template Data<T_B>());
      }
      // B is (N,)
      else if (b_shape.NumDimensions() == 1) {
        auto bias_vec = ConstEigenVectorMap<T_B>(
            B->template Data<T_B>(),
            N);
        output_mat.rowwise() += bias_vec.transpose();
      } else if (b_shape.NumDimensions() == 2) {
        // B is (M, 1)
        if (b_shape[1] == 1) {
          auto bias_vec = ConstEigenVectorMap<T_B>(
              B->template Data<T_B>(),
              M);
          output_mat.colwise() += bias_vec;
        }
        // B is (1, N)
        else if (b_shape[0] == 1) {
          auto bias_vec = ConstEigenVectorMap<T_B>(
              B->template Data<T_B>(),
              N);
          output_mat.rowwise() += bias_vec.transpose();
        }
        // B is (M, N), no broadcast needed.
        else {
          auto bias_mat = ConstEigenMatrixMapRowMajor<T_B>(
              B->template Data<T_B>(),
              M,
              N);
          output_mat += bias_mat;
        }
      }
    }

    // W * x
    math::Gemm<T_X, CPUMathUtil>(
        trans_A_,
        trans_B_,
        M,
        N,
        K,
        alpha_,
        X->template Data<T_X>(),
        W->template Data<T_W>(),
        beta_,
        y_data,
        &CPUMathUtil::Instance());

    FuseActivation<T_Y>(activation_, y_data, M * N, leaky_relu_alpha_);

    return Status::OK();
  }

 private:
  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;

protected:
  // For fused gemm + activation
  std::string activation_;
  float leaky_relu_alpha_;
};

}  // namespace onnxruntime
