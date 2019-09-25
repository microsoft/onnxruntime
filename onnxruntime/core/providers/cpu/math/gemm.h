// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "gemm_helper.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {

template <typename T>
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
    auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
    concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

    const auto* X = context->Input<Tensor>(0);
    const auto* W = context->Input<Tensor>(1);
    const auto* B = context->InputCount() == 3 ? context->Input<Tensor>(2) : nullptr;
    // Bias could be missing. Treat as scalar 0 if that is the case.
    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans,
                      B != nullptr ? B->Shape() : TensorShape({}));

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    auto Y = context->Output(0, {M, N});
    // if input is emtpy tensor, return directly as nothing need to be calculated.
    if (M == 0 || N == 0)
      return Status::OK();
    T* y_data = Y->template MutableData<T>();

    if (beta_ != 0) {
      auto output_mat = EigenMatrixMapRowMajor<T>(y_data, M, N);
      // if beta is non-zero and bias is missing, we need to reset memory of the output buffer to 0
      // treating the "missing" bias as scalar 0, otherwise any junk data in the output buffer will be used as bias
      if (B == nullptr) {
        // B is treated as scalar 0
        output_mat.setConstant(0);
      } else {
        // Broadcast the bias as needed IF the bias is provided
        const auto& b_shape = B->Shape();
        const T* b_data = B->template Data<T>();
        if (b_shape.Size() == 1) {
          // B is (), (1,) or (1, 1), set the scalar
          output_mat.setConstant(*b_data);
        } else if (b_shape.NumDimensions() == 1 || b_shape[0] == 1) {
          // B is (N,) or (1, N)
          output_mat.rowwise() = ConstEigenVectorMap<T>(b_data, N).transpose();
        } else if (b_shape[1] == 1) {
          // B is (M, 1)
          output_mat.colwise() = ConstEigenVectorMap<T>(b_data, M);
        } else {
          // B is (M, N), no broadcast needed.
          output_mat = ConstEigenMatrixMapRowMajor<T>(b_data, M, N);
        }
      }
    }

    // W * x
    math::Gemm<T>(
        trans_A_,
        trans_B_,
        M,
        N,
        helper.K(),
        alpha_,
        X->template Data<T>(),
        W->template Data<T>(),
        beta_,
        y_data,
        tp);

    FuseActivation<T>(activation_, y_data, M * N, leaky_relu_alpha_);

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
