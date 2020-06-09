// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "gemm_helper.h"
#include "core/providers/cpu/activation/activations.h"

namespace onnxruntime {

template <typename T>
class Gemm : public OpKernel {
private:
    class CallWrapper{
public:
    CallWrapper(functors::ElementWiseRangedTransform<T>* b1):b(b1){}
    void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
        (*b)(first, last);
    }
private:
    functors::ElementWiseRangedTransform<T>* b;
};

 public:
  Gemm(const OpKernelInfo& info) : OpKernel(info)
  {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  static void ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                          int64_t M, int64_t N, int64_t K,
                          float alpha,
                          const T* a_data, const T* b_data,
                          float beta,
                          const T* c_data, const TensorShape* c_shape,
                          T* y_data,
                          concurrency::ThreadPool* thread_pool) {
    // if input is empty tensor, return directly as nothing need to be calculated.
    if (M == 0 || N == 0)
      return;

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

    math::Gemm<T>(trans_a, trans_b,
                  M, N, K,
                  alpha,
                  a_data,
                  b_data,
                  // ideally we need to set the output buffer contents to 0 if bias is missing,
                  // but passing 0 for beta is cheaper and it will ignore any junk in the output buffer
                  c_data != nullptr ? beta : 0,
                  y_data,
                  thread_pool);
  }

  Status Compute(OpKernelContext* context) const override {
    concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

    const auto* X = context->Input<Tensor>(0);
    const auto* W = context->Input<Tensor>(1);
    const auto* B = context->Input<Tensor>(2);
    // Bias could be missing. Treat as scalar 0 if that is the case.
    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans,
                      B != nullptr ? B->Shape() : TensorShape({}));

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    int64_t K = helper.K();

    auto Y = context->Output(0, {M, N});

    // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
    if (M == 0 || N == 0)
      return Status::OK();

    const T* b_data = B != nullptr ? B->Data<T>() : nullptr;
    const TensorShape* b_shape = B != nullptr ? &B->Shape() : nullptr;

    T* y_data = Y->MutableData<T>();

    ComputeGemm(trans_A_, trans_B_, M, N, K, alpha_, X->Data<T>(), W->Data<T>(), beta_,
                b_data, b_shape,
                y_data,
                thread_pool);

    if(activation_){
      std::unique_ptr<functors::ElementWiseRangedTransform<T>> f(activation_->Copy());
      f->input = y_data;
      f->output = y_data;
      std::ptrdiff_t total_len = static_cast<std::ptrdiff_t>(M * N);
      double cost = f->Cost();
      CallWrapper c(f.get());
      concurrency::ThreadPool::TryParallelFor(thread_pool, total_len, {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), cost}, c);
    }
    return Status::OK();
  }

 private:
  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;

 protected:
  // For fused gemm + activation  
  std::unique_ptr<functors::ElementWiseRangedTransform<T>> activation_;
};

}  // namespace onnxruntime
