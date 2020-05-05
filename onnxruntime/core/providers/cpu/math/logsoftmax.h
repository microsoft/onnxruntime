// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl-lite.hpp"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/softmax_shared.h"

namespace onnxruntime {
template <typename T>
class LogSoftmax final : public OpKernel {
 public:
  LogSoftmax(const OpKernelInfo& info) : OpKernel{info}, axis_{1} {
    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    }
  }

  Status Compute(OpKernelContext* ctx) const override {
    concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();

    const auto* tensor_pointer = ctx->Input<Tensor>(0);
    if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const Tensor& X = *tensor_pointer;
    const TensorShape& input_shape{X.Shape()};

    Tensor* Y = ctx->Output(0, input_shape);

    // edge case. one or more dims with value of 0. nothing to do
    if (input_shape.Size() == 0)
      return Status::OK();

    const int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());

    size_t N = input_shape.SizeToDimension(axis);
    size_t D = input_shape.SizeFromDimension(axis);

    auto* Ydata = Y->template MutableData<T>();

    std::vector<T> scale_(N);
    std::vector<T> rowmax_(N);
    std::vector<T> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0

    const bool logarithmic = true;
    auto status = SoftmaxCPU(N, D, X.template Data<T>(), Ydata,
                             scale_.data(), sum_multiplier_.data(), logarithmic, rowmax_.data(), tp);

    return status;
  }

 private:
  int axis_;
};
}  // namespace onnxruntime
