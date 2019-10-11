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
    auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
    concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

    const auto* X = ctx->Input<Tensor>(0);
    if (X->Shape().NumDimensions() == 0) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input shape needs to be at least a single dimension.");
    }
    // X: [N, feature_size],coefficients_[target, feature_size]
    // X*coefficients_^t : [N, target]
    int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    Tensor* Y = ctx->Output(0, TensorShape({N, targets_}));
    T* Ydata = Y->MutableData<T>();
    math::MatMul<T>(static_cast<int>(N), static_cast<int>(targets_), static_cast<int>(stride), X->Data<T>(),
                    coefficients_.data(), Y->MutableData<T>(), tp);
    bool useIntercepts = intercepts_.size() == static_cast<size_t>(targets_);
    if (useIntercepts) {
      for (int64_t i = 0; i < N; i++)  // for each point
      {
        T* p = Ydata + i * targets_;
        for (int j = 0; j < targets_; j++)  // for each target
        {
          p[j] += intercepts_[j];
        }
      }
    }
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
  std::vector<T> coefficients_;
  std::vector<T> intercepts_;
  // Indeed, it can't be PROBIT
  POST_EVAL_TRANSFORM post_transform_;
};

}  // namespace ml
}  // namespace onnxruntime
