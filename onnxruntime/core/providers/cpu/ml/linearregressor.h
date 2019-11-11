// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/util/eigen_common_wrapper.h"
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
    if (X->Shape().NumDimensions() > 2) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Input shape must have no more than 2 dimensions");
    }
    // X: [N, feature_size],coefficients_[feature_size, target]
    // X*coefficients_ : [N, target]
    const auto& input_shape = X->Shape();
    int64_t stride = input_shape.NumDimensions() <= 1 ? input_shape.Size() : input_shape[1];
    int64_t N = input_shape.NumDimensions() <= 1 ? 1 : input_shape[0];
    Tensor* Y = ctx->Output(0, TensorShape({N, targets_}));
    ORT_ENFORCE(N <= std::numeric_limits<int>::max());
    T* Ydata = Y->MutableData<T>();
    typename Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Unaligned>
        input_tensor(X->Data<T>(), N, stride);
    typename Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Unaligned> output_tensor(
        Y->MutableData<T>(), N, targets_);
    if (has_intercepts_) {
      Eigen::array<int, 2> bcast{static_cast<int>(N), 1};
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
#ifndef USE_OPENMP
      if (tp == nullptr)
#endif
        output_tensor.device(Eigen::DefaultDevice()) =
            input_tensor.contract(coefficients_, product_dims) + intercepts_.broadcast(bcast);
#ifndef USE_OPENMP
      else
        output_tensor.device(Eigen::ThreadPoolDevice(&tp->GetHandler(), tp->NumThreads())) =
            input_tensor.contract(coefficients_, product_dims) + intercepts_.broadcast(bcast);
#endif
    } else {
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
#ifndef USE_OPENMP
      if (tp == nullptr)
#endif
        output_tensor.device(Eigen::DefaultDevice()) =
            input_tensor.contract(coefficients_, product_dims);
#ifndef USE_OPENMP
      else
        output_tensor.device(Eigen::ThreadPoolDevice(&tp->GetHandler(), tp->NumThreads())) =
            input_tensor.contract(coefficients_, product_dims);
#endif
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
  const int64_t targets_;
  const Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> coefficients_;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> intercepts_;
  // Indeed, it can't be PROBIT, because the PROBIT function requires its input in [0,1] range.
  // But here we can't guarantee that.
  const POST_EVAL_TRANSFORM post_transform_;
  bool has_intercepts_ = false;
};

}  // namespace ml
}  // namespace onnxruntime
