// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/util/math.h"
#include "ml_common.h"

namespace onnxruntime {
namespace ml {

template <typename T>
class LinearRegressor final : public OpKernel {
 public:
  LinearRegressor(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override {
#ifndef USE_OPENMP
    concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();
#endif
    const auto* X = ctx->Input<Tensor>(0);
    if (X->Shape().NumDimensions() == 0) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input shape needs to be at least a single dimension.");
    }
    // X: [N, feature_size],coefficients_[feature_size, target]
    // X*coefficients_ : [N, target]
    int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    Tensor* Y = ctx->Output(0, TensorShape({N, targets_}));
    T* Ydata = Y->MutableData<T>();
    typename Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Unaligned> input_tensor(X->Data<T>(), N, stride);
    typename Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Unaligned> output_tensor(Y->MutableData<T>(), N, targets_);
    Eigen::array<int, 2> bcast{static_cast<int>(N), 1};
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
#ifndef USE_OPENMP
    output_tensor.device(Eigen::ThreadPoolDevice(&tp->GetHandler(), tp->NumThreads())) = input_tensor.contract(coefficients_, product_dims) + intercepts_.broadcast(bcast);
#else
    output_tensor.device(Eigen::DefaultDevice()) = input_tensor.contract(coefficients_, product_dims) + intercepts_.broadcast(bcast);
#endif
    // TODO: parallel this part
    if (post_transform_ != POST_EVAL_TRANSFORM::NONE)
      for (int64_t i = 0; i < N; i++)  // for each point
      {
        T* p = Ydata + i * targets_;
        ml::write_scores(p, targets_, post_transform_, p);
      }
    return Status::OK();
  }

 private:
  int64_t targets_;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> coefficients_;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> intercepts_;
  // Indeed, it can't be PROBIT, because the PROBIT function requires its input in [0,1] range.
  // But here we can't guarantee that.
  POST_EVAL_TRANSFORM post_transform_;
};

}  // namespace ml
}  // namespace onnxruntime
